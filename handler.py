from __future__ import annotations

import base64
from dataclasses import replace
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.parse import urlencode
import uuid

from PIL import Image
import requests
import runpod
from runpod.serverless.utils import rp_upload
import websocket

from instruction_parser import EditPass, InstructionParseError, parse_instruction
from workflow_builder import SAVE_IMAGE_NODE_ID, build_workflow, coerce_seed


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("qwen-v19-handler")

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_HISTORY_TIMEOUT_S = int(os.environ.get("COMFY_HISTORY_TIMEOUT_S", "1800"))
COMFY_POLL_INTERVAL_S = float(os.environ.get("COMFY_POLL_INTERVAL_S", "2"))
COMFY_STARTUP_TIMEOUT_S = int(os.environ.get("COMFY_STARTUP_TIMEOUT_S", "300"))
COMFY_STARTUP_POLL_INTERVAL_S = float(os.environ.get("COMFY_STARTUP_POLL_INTERVAL_S", "2"))
WORKFLOW_TEMPLATE_DIR = Path(__file__).resolve().parent / "workflow_templates"

QWEN_CHECKPOINT_NAME = os.environ.get("QWEN_CHECKPOINT_NAME", "Qwen-Rapid-AIO-NSFW-v19.safetensors")
DEFAULT_STEPS = int(os.environ.get("QWEN_DEFAULT_STEPS", "8"))
DEFAULT_CFG = float(os.environ.get("QWEN_DEFAULT_CFG", "1.0"))
DEFAULT_SAMPLER = os.environ.get("QWEN_DEFAULT_SAMPLER", "euler_ancestral")
DEFAULT_SCHEDULER = os.environ.get("QWEN_DEFAULT_SCHEDULER", "beta")
DEFAULT_DENOISE = float(os.environ.get("QWEN_DEFAULT_DENOISE", "0.84"))
DEFAULT_PARSER_MODEL = os.environ.get("QWEN_DEFAULT_PARSER_MODEL", "lip").strip().lower()
DEFAULT_FILENAME_PREFIX = os.environ.get("QWEN_DEFAULT_FILENAME_PREFIX", "qwen-v19-clothing/output")
DEFAULT_TARGET_WIDTH = int(os.environ.get("QWEN_DEFAULT_TARGET_WIDTH", "1024"))
DEFAULT_TARGET_HEIGHT = int(os.environ.get("QWEN_DEFAULT_TARGET_HEIGHT", "1024"))
DEFAULT_MASK_EXPAND_PIXELS = int(os.environ.get("QWEN_DEFAULT_MASK_EXPAND_PIXELS", "0"))
DEFAULT_MASK_BLEND_PIXELS = int(os.environ.get("QWEN_DEFAULT_MASK_BLEND_PIXELS", "2"))
DEFAULT_CONTEXT_EXPAND_FACTOR = float(os.environ.get("QWEN_DEFAULT_CONTEXT_EXPAND_FACTOR", "1.15"))
DEFAULT_OUTPUT_PADDING = int(os.environ.get("QWEN_DEFAULT_OUTPUT_PADDING", "32"))
DEFAULT_DEVICE_MODE = os.environ.get("QWEN_DEFAULT_DEVICE_MODE", "gpu").strip().lower()
MASK_SCOPE_NAME = "body_silhouette_to_neck"
MAX_INPUT_LONG_SIDE = int(os.environ.get("QWEN_MAX_INPUT_LONG_SIDE", "1920"))
AUTO_TARGET_LONG_SIDE = int(os.environ.get("QWEN_AUTO_TARGET_LONG_SIDE", str(MAX_INPUT_LONG_SIDE)))
AUTO_DENOISE_WIDE = float(os.environ.get("QWEN_AUTO_DENOISE_WIDE", "0.78"))
AUTO_DENOISE_STANDARD = float(os.environ.get("QWEN_AUTO_DENOISE_STANDARD", "0.82"))
LEG_REVEAL_DENOISE = float(os.environ.get("QWEN_LEG_REVEAL_DENOISE", "0.9"))
TARGET_REFINE_DENOISE = float(os.environ.get("QWEN_TARGET_REFINE_DENOISE", "0.9"))
TARGET_REFINE_STEPS_BONUS = int(os.environ.get("QWEN_TARGET_REFINE_STEPS_BONUS", "2"))
DEFAULT_HAIR_CLEANUP = os.environ.get("QWEN_DEFAULT_HAIR_CLEANUP", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_HAIR_CLEANUP_DENOISE = float(os.environ.get("QWEN_DEFAULT_HAIR_CLEANUP_DENOISE", "0.62"))
DEFAULT_HAIR_CLEANUP_STEPS = int(os.environ.get("QWEN_DEFAULT_HAIR_CLEANUP_STEPS", "6"))
VALID_OUTPUT_PADDING_VALUES = (0, 8, 16, 32, 64, 128, 256, 512)
LEG_REVEAL_KEYWORDS = (
    "shorts",
    "short short",
    "mini skirt",
    "miniskirt",
    "micro skirt",
    "bodysuit",
    "swimsuit",
    "bikini",
    "leotard",
    "romper",
    "remove dress",
    "remove skirt",
)


def comfy_url(path: str) -> str:
    return f"http://{COMFY_HOST}{path}"


def ws_url(client_id: str) -> str:
    return f"ws://{COMFY_HOST}/ws?clientId={client_id}"


def parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def strip_data_uri(data: str) -> str:
    if "," in data and data.split(",", 1)[0].startswith("data:"):
        return data.split(",", 1)[1]
    return data


def slugify(value: str) -> str:
    lowered = value.lower()
    lowered = "".join(char if char.isalnum() else "-" for char in lowered)
    while "--" in lowered:
        lowered = lowered.replace("--", "-")
    return lowered.strip("-") or "edit"


def bucket_upload_enabled() -> bool:
    required = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
    return all(os.environ.get(key) for key in required)


def normalize_image_bytes(raw_bytes: bytes) -> bytes:
    with Image.open(BytesIO(raw_bytes)) as image:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def resize_image_bytes_to_long_side(raw_bytes: bytes, max_long_side: int) -> tuple[bytes, dict[str, int | bool]]:
    with Image.open(BytesIO(raw_bytes)) as image:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")

        original_width, original_height = image.size
        original_long_side = max(original_width, original_height)
        if max_long_side <= 0 or original_long_side <= max_long_side:
            output = BytesIO()
            image.save(output, format="PNG")
            return output.getvalue(), {
                "resized": False,
                "original_width": original_width,
                "original_height": original_height,
                "width": original_width,
                "height": original_height,
            }

        scale = max_long_side / float(original_long_side)
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        output = BytesIO()
        resized.save(output, format="PNG")
        return output.getvalue(), {
            "resized": True,
            "original_width": original_width,
            "original_height": original_height,
            "width": resized_width,
            "height": resized_height,
        }


def image_size_from_bytes(raw_bytes: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(raw_bytes)) as image:
        return image.size


def load_source_image_bytes(job_input: dict[str, Any]) -> bytes:
    image_base64 = job_input.get("image_base64")
    generic_image = job_input.get("image")
    image_url = job_input.get("image_url")

    if image_base64:
        return normalize_image_bytes(base64.b64decode(strip_data_uri(str(image_base64))))

    if isinstance(generic_image, str):
        if generic_image.startswith("http://") or generic_image.startswith("https://"):
            image_url = generic_image
        else:
            return normalize_image_bytes(base64.b64decode(strip_data_uri(generic_image)))

    if image_url:
        response = requests.get(str(image_url), timeout=60)
        response.raise_for_status()
        return normalize_image_bytes(response.content)

    raise ValueError("Provide one of: image_base64, image, or image_url.")


def check_server() -> None:
    response = requests.get(comfy_url("/"), timeout=10)
    response.raise_for_status()


def wait_for_server() -> None:
    start = time.time()
    while True:
        try:
            check_server()
            LOGGER.info("ComfyUI is ready after %.1fs", time.time() - start)
            return
        except requests.RequestException as exc:
            elapsed = time.time() - start
            if elapsed >= COMFY_STARTUP_TIMEOUT_S:
                raise TimeoutError(f"Timed out waiting for ComfyUI after {COMFY_STARTUP_TIMEOUT_S}s.") from exc
            LOGGER.info("Waiting for ComfyUI on %s (%.1fs elapsed)", COMFY_HOST, elapsed)
            time.sleep(COMFY_STARTUP_POLL_INTERVAL_S)


def upload_image_bytes(filename: str, image_bytes: bytes) -> None:
    files = {
        "image": (filename, BytesIO(image_bytes), "image/png"),
        "overwrite": (None, "true"),
    }
    response = requests.post(comfy_url("/upload/image"), files=files, timeout=60)
    response.raise_for_status()


def queue_workflow(workflow: dict[str, Any], client_id: str) -> dict[str, Any]:
    payload = {"prompt": workflow, "client_id": client_id}
    response = requests.post(comfy_url("/prompt"), json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "prompt_id" not in data:
        raise RuntimeError(f"Missing prompt_id in queue response: {data}")
    return data


def get_history(prompt_id: str) -> dict[str, Any]:
    response = requests.get(comfy_url(f"/history/{prompt_id}"), timeout=60)
    response.raise_for_status()
    return response.json()


def wait_for_history_entry(prompt_id: str) -> dict[str, Any]:
    deadline = time.time() + COMFY_HISTORY_TIMEOUT_S
    while time.time() < deadline:
        history = get_history(prompt_id)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(COMFY_POLL_INTERVAL_S)
    raise TimeoutError(f"Timed out waiting for history entry for prompt {prompt_id}")


def get_image_bytes(filename: str, subfolder: str, image_type: str) -> bytes:
    query = urlencode(
        {
            "filename": filename,
            "subfolder": subfolder,
            "type": image_type,
        }
    )
    response = requests.get(comfy_url(f"/view?{query}"), timeout=60)
    response.raise_for_status()
    return response.content


def wait_for_execution(socket: websocket.WebSocket, prompt_id: str) -> None:
    deadline = time.time() + COMFY_HISTORY_TIMEOUT_S
    while time.time() < deadline:
        try:
            message = socket.recv()
        except websocket.WebSocketTimeoutException:
            continue

        if not isinstance(message, str):
            continue

        payload = json.loads(message)
        message_type = payload.get("type")
        data = payload.get("data", {})

        if message_type == "executing" and data.get("prompt_id") == prompt_id and data.get("node") is None:
            return

        if message_type == "execution_error" and data.get("prompt_id") == prompt_id:
            error_message = data.get("exception_message") or "ComfyUI reported an execution error."
            node_id = data.get("node_id")
            if node_id is not None:
                raise RuntimeError(f"Execution error at node {node_id}: {error_message}")
            raise RuntimeError(error_message)

    raise TimeoutError(f"Timed out waiting for workflow completion for prompt {prompt_id}")


def extract_output_images(prompt_history: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = prompt_history.get("outputs", {})
    save_output = outputs.get(SAVE_IMAGE_NODE_ID, {})
    images = save_output.get("images", [])
    if images:
        return images

    for node_output in outputs.values():
        images = node_output.get("images", [])
        if images:
            return images
    return []


def is_leg_reveal_edit(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in LEG_REVEAL_KEYWORDS)


def build_positive_prompt(edit_pass: EditPass) -> str:
    prompt = (
        "Regenerate only inside the strict body silhouette mask, from the neck down.\n"
        "The mask must stop at the neck: do not edit the face, head, hairline, eyes, mouth, or expression.\n"
        "Replace all visible old clothing inside the mask, including clothing near hair, shoulders, neck, arms, waist, hips, and legs.\n"
        "Keep pose, limb placement, camera framing, perspective, and body proportions as close to the source image as possible.\n"
        "Preserve the exact number of visible limbs and joints from the source image.\n"
        "Do not create extra legs, extra feet, extra knees, extra arms, or extra hands.\n"
        "If the source clothing is loose, infer hidden body proportions conservatively from visible cues without exaggeration.\n"
        "Apply the user's clothing request consistently across the whole masked body region.\n"
        "Do not leave fragments, scraps, seams, patches, or denim pieces of the original clothing unless the user explicitly asks for them.\n"
        "Keep all unmasked regions unchanged.\n"
    )
    if is_leg_reveal_edit(edit_pass.edit_text):
        prompt += (
            "This request reveals the legs. Preserve visible feet, shoes, ankles, calves, knees, and leg pose from the source.\n"
            "Where the old dress or skirt hid the upper legs, synthesize continuous natural legs connecting the hips to the visible lower legs.\n"
            "Do not erase legs, fade legs into the background, hide legs behind invisible fabric, or cut off the legs.\n"
        )
    return f"{prompt}User request: {edit_pass.edit_text}"


def build_target_refine_positive_prompt(edit_pass: EditPass) -> str:
    focus_lines = [
        f"Refine only the masked {edit_pass.category} region for this pass.",
        f"Fully replace the visible {edit_pass.category.lower()} and remove remnants of the original {edit_pass.category.lower()}.",
        "Keep anatomy, pose, limbs, hands, and the rest of the outfit stable.",
    ]
    if edit_pass.parser_field in {"upper_clothes", "dress", "coat"}:
        focus_lines.append(
            "Clean residual old-clothing fragments near hair strands, collar lines, shoulders, and neckline while keeping the hair natural."
        )
    if edit_pass.parser_field in {"pants", "skirt", "jumpsuits"}:
        focus_lines.append(
            "Ensure the lower-body garment changes clearly and completely instead of keeping the original pants or denim."
        )
    focus_lines.append(f"User request: {edit_pass.edit_text}")
    return "\n".join(focus_lines)


def build_hair_cleanup_pass(instruction: str) -> EditPass:
    return EditPass(
        category="Hair Cleanup",
        parser_field="hair",
        parser_type="lip",
        supported_parsers=("lip",),
        edit_text=instruction,
        category_negatives=(),
    )


def build_hair_cleanup_positive_prompt(instruction: str) -> str:
    return (
        "Refine only the masked hair region.\n"
        "Remove any clothing, denim, fabric, blue garment, collar, or jacket fragments visible inside the hair.\n"
        "Restore continuous natural hair strands matching the source hair color, flow, length, and texture.\n"
        "Keep face, eyes, mouth, skin, neck, shoulders, body, outfit, background, lighting, and camera framing unchanged.\n"
        f"Original user request: {instruction}"
    )


def build_hair_cleanup_negative_prompt() -> str:
    return (
        "clothing in hair, denim in hair, fabric scraps, blue fabric, jacket fragments, collar fragments, "
        "changed face, changed expression, changed hairstyle, shorter hair, missing hair, blurred hair, "
        "changed skin, changed neck, changed shoulder, changed outfit, background change"
    )


def build_negative_prompt(edit_pass: EditPass) -> str:
    base = (
        "change face, change identity, different person, different hairstyle, different expression, "
        "changed hairline, changed head position, changed head size, different pose, changed arm position, "
        "changed leg position, changed shoulder width, widened hips, enlarged chest, slimmed waist, "
        "longer legs, shorter torso, background change, lighting change, extra garments, extra limbs, "
        "duplicate body parts, duplicate leg, duplicate foot, duplicate knee, duplicate ankle, duplicate arm, "
        "extra leg, extra foot, extra knee, extra thigh, extra arm, extra hand, missing leg, missing legs, "
        "invisible leg, invisible legs, erased legs, cut off legs, floating feet, disconnected feet, warped hands, missing hair, "
        "leftover original clothing, torn fabric scraps, partial outfit change, inconsistent outfit coverage, "
        "mask bleed, edits outside masked region"
    )
    category_negatives = list(edit_pass.category_negatives)
    if is_leg_reveal_edit(edit_pass.edit_text):
        category_negatives = [negative for negative in category_negatives if negative != "legs changed"]
    extras = ", ".join(category_negatives)
    if extras:
        return f"{base}, {extras}"
    return base


def encode_output_artifact(job_id: str, filename: str, image_bytes: bytes) -> dict[str, str]:
    if bucket_upload_enabled():
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix or ".png") as tmp_file:
            tmp_file.write(image_bytes)
            temp_path = tmp_file.name
        try:
            uploaded_url = rp_upload.upload_image(job_id, temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
        return {"filename": filename, "type": "s3_url", "data": uploaded_url}

    return {
        "filename": filename,
        "type": "base64",
        "data": base64.b64encode(image_bytes).decode("utf-8"),
    }


def parse_device_mode(job_input: dict[str, Any]) -> str:
    requested = str(job_input.get("device_mode", DEFAULT_DEVICE_MODE)).strip().lower()
    if requested not in {"gpu", "cpu"}:
        raise ValueError("device_mode must be either 'gpu' or 'cpu'.")
    return requested


def snap_dimension(value: float, minimum: int = 640) -> int:
    snapped = int(round(value / 64.0) * 64)
    return max(minimum, snapped)


def snap_dimension_down(value: float, minimum: int = 512) -> int:
    snapped = int(value // 64) * 64
    return max(minimum, snapped)


def snap_output_padding(value: int | str) -> int:
    requested = int(value)
    return min(VALID_OUTPUT_PADDING_VALUES, key=lambda candidate: abs(candidate - requested))


def cap_target_to_long_side(options: dict[str, Any], max_long_side: int) -> None:
    target_width = int(options["target_width"])
    target_height = int(options["target_height"])
    target_long_side = max(target_width, target_height)
    if max_long_side <= 0 or target_long_side <= max_long_side:
        return

    scale = max_long_side / float(target_long_side)
    options["target_width"] = snap_dimension_down(target_width * scale)
    options["target_height"] = snap_dimension_down(target_height * scale)


def maybe_autosize_target(source_bytes: bytes, options: dict[str, Any]) -> None:
    if options["target_width"] != DEFAULT_TARGET_WIDTH or options["target_height"] != DEFAULT_TARGET_HEIGHT:
        cap_target_to_long_side(options, MAX_INPUT_LONG_SIDE)
        return

    source_width, source_height = image_size_from_bytes(source_bytes)
    longest_side = max(source_width, source_height)
    if longest_side <= 0:
        return

    aspect_ratio = source_width / source_height
    if 0.8 <= aspect_ratio <= 1.25:
        cap_target_to_long_side(options, MAX_INPUT_LONG_SIDE)
        return

    scale = AUTO_TARGET_LONG_SIDE / float(longest_side)
    auto_width = snap_dimension_down(source_width * scale)
    auto_height = snap_dimension_down(source_height * scale)
    options["target_width"] = auto_width
    options["target_height"] = auto_height
    cap_target_to_long_side(options, MAX_INPUT_LONG_SIDE)


def maybe_autotune_denoise(source_bytes: bytes, options: dict[str, Any]) -> None:
    if "denoise" in options["raw"]:
        return

    source_width, source_height = image_size_from_bytes(source_bytes)
    if source_height <= 0:
        return

    aspect_ratio = source_width / source_height
    target_denoise = AUTO_DENOISE_WIDE if (aspect_ratio < 0.8 or aspect_ratio > 1.25) else AUTO_DENOISE_STANDARD
    options["denoise"] = min(float(options["denoise"]), float(target_denoise))


def maybe_tune_leg_reveal_denoise(passes: list[EditPass], options: dict[str, Any]) -> None:
    if "denoise" in options["raw"]:
        return
    if any(is_leg_reveal_edit(edit_pass.edit_text) for edit_pass in passes):
        options["denoise"] = max(float(options["denoise"]), float(LEG_REVEAL_DENOISE))


def force_lip_for_body_mask(passes: list[EditPass]) -> list[EditPass]:
    coerced: list[EditPass] = []
    for edit_pass in passes:
        if edit_pass.parser_type == "lip":
            coerced.append(edit_pass)
            continue
        if "lip" in edit_pass.supported_parsers:
            coerced.append(replace(edit_pass, parser_type="lip"))
            continue
        coerced.append(edit_pass)
    return coerced


def should_run_target_refinement(edit_pass: EditPass) -> bool:
    return False


def refinement_denoise(options: dict[str, Any]) -> float:
    return max(float(options["denoise"]), float(TARGET_REFINE_DENOISE))


def refinement_steps(options: dict[str, Any]) -> int:
    return int(options["steps"]) + int(TARGET_REFINE_STEPS_BONUS)


def validate_input(job_input: dict[str, Any]) -> dict[str, Any]:
    if job_input is None:
        raise ValueError("Please provide input.")

    if isinstance(job_input, str):
        job_input = json.loads(job_input)

    instruction = str(job_input.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("Missing 'instruction'.")

    if not any(key in job_input for key in ("image", "image_base64", "image_url")):
        raise ValueError("Provide one of: image, image_base64, or image_url.")

    parser_model = str(job_input.get("parser_model", DEFAULT_PARSER_MODEL)).strip().lower()
    seed = coerce_seed(job_input.get("seed"))
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    cfg = float(job_input.get("cfg", DEFAULT_CFG))
    sampler_name = str(job_input.get("sampler_name", DEFAULT_SAMPLER)).strip()
    scheduler = str(job_input.get("scheduler", DEFAULT_SCHEDULER)).strip()
    denoise = float(job_input.get("denoise", DEFAULT_DENOISE))
    target_width = int(job_input.get("target_width", DEFAULT_TARGET_WIDTH))
    target_height = int(job_input.get("target_height", DEFAULT_TARGET_HEIGHT))
    mask_expand_pixels = int(job_input.get("mask_expand_pixels", DEFAULT_MASK_EXPAND_PIXELS))
    mask_blend_pixels = int(job_input.get("mask_blend_pixels", DEFAULT_MASK_BLEND_PIXELS))
    context_expand_factor = float(job_input.get("context_expand_factor", DEFAULT_CONTEXT_EXPAND_FACTOR))
    output_padding = snap_output_padding(job_input.get("output_padding", DEFAULT_OUTPUT_PADDING))
    checkpoint_name = str(job_input.get("checkpoint_name", QWEN_CHECKPOINT_NAME)).strip()
    hair_cleanup = parse_bool(job_input.get("hair_cleanup"), DEFAULT_HAIR_CLEANUP)
    hair_cleanup_denoise = float(job_input.get("hair_cleanup_denoise", DEFAULT_HAIR_CLEANUP_DENOISE))
    hair_cleanup_steps = int(job_input.get("hair_cleanup_steps", DEFAULT_HAIR_CLEANUP_STEPS))

    if steps < 1:
        raise ValueError("steps must be >= 1.")
    if cfg <= 0:
        raise ValueError("cfg must be > 0.")
    if denoise <= 0:
        raise ValueError("denoise must be > 0.")
    if hair_cleanup_denoise <= 0:
        raise ValueError("hair_cleanup_denoise must be > 0.")
    if hair_cleanup_steps < 1:
        raise ValueError("hair_cleanup_steps must be >= 1.")

    return {
        "raw": job_input,
        "instruction": instruction,
        "parser_model": parser_model,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "denoise": denoise,
        "target_width": target_width,
        "target_height": target_height,
        "mask_expand_pixels": mask_expand_pixels,
        "mask_blend_pixels": mask_blend_pixels,
        "context_expand_factor": context_expand_factor,
        "output_padding": output_padding,
        "device_mode": parse_device_mode(job_input),
        "checkpoint_name": checkpoint_name,
        "hair_cleanup": hair_cleanup,
        "hair_cleanup_denoise": hair_cleanup_denoise,
        "hair_cleanup_steps": hair_cleanup_steps,
    }


def execute_pass(
    *,
    job_id: str,
    client_id: str,
    pass_index: int,
    stage_name: str,
    current_image_bytes: bytes,
    edit_pass: EditPass,
    options: dict[str, Any],
    mask_mode: str,
    positive_prompt: str,
    negative_prompt: str,
    denoise: float,
    steps: int,
) -> tuple[bytes, dict[str, Any]]:
    input_name = f"{job_id}_pass_{pass_index:02d}_{stage_name}_input.png"
    upload_image_bytes(input_name, current_image_bytes)

    filename_prefix = (
        f"{DEFAULT_FILENAME_PREFIX}/{job_id}/pass_{pass_index:02d}_{slugify(edit_pass.category)}_{slugify(stage_name)}"
    )

    workflow = build_workflow(
        template_dir=WORKFLOW_TEMPLATE_DIR,
        input_image_name=input_name,
        checkpoint_name=options["checkpoint_name"],
        edit_pass=edit_pass,
        mask_mode=mask_mode,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        seed=options["seed"] + pass_index - 1,
        steps=steps,
        cfg=options["cfg"],
        sampler_name=options["sampler_name"],
        scheduler=options["scheduler"],
        denoise=denoise,
        target_width=options["target_width"],
        target_height=options["target_height"],
        mask_expand_pixels=options["mask_expand_pixels"],
        mask_blend_pixels=options["mask_blend_pixels"],
        context_expand_factor=options["context_expand_factor"],
        output_padding=options["output_padding"],
        device_mode=options["device_mode"],
        filename_prefix=filename_prefix,
    )

    socket = websocket.WebSocket()
    socket.connect(ws_url(client_id), timeout=10)
    socket.settimeout(COMFY_POLL_INTERVAL_S)

    try:
        queue_response = queue_workflow(workflow, client_id)
        prompt_id = queue_response["prompt_id"]
        wait_for_execution(socket, prompt_id)
    finally:
        socket.close()
    history_entry = wait_for_history_entry(prompt_id)

    output_images = extract_output_images(history_entry)
    if not output_images:
        raise RuntimeError(f"No output images were produced for pass {pass_index}.")

    first_image = output_images[0]
    image_bytes = get_image_bytes(
        filename=first_image["filename"],
        subfolder=first_image.get("subfolder", ""),
        image_type=first_image.get("type", "output"),
    )

    metadata = {
        "prompt_id": prompt_id,
        "category": edit_pass.category,
        "parser_field": edit_pass.parser_field,
        "parser_type": edit_pass.parser_type,
        "mask_scope": "hair_cleanup" if stage_name == "hair-cleanup" else MASK_SCOPE_NAME,
        "mask_mode": mask_mode,
        "stage_name": stage_name,
        "edit_text": edit_pass.edit_text,
        "filename": first_image["filename"],
        "subfolder": first_image.get("subfolder", ""),
        "image_type": first_image.get("type", "output"),
    }
    return image_bytes, metadata


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        wait_for_server()

        options = validate_input(job.get("input"))
        source_bytes = load_source_image_bytes(options["raw"])
        source_bytes, resize_metadata = resize_image_bytes_to_long_side(source_bytes, MAX_INPUT_LONG_SIDE)
        maybe_autosize_target(source_bytes, options)
        maybe_autotune_denoise(source_bytes, options)
        passes = force_lip_for_body_mask(
            parse_instruction(options["instruction"], preferred_parser=options["parser_model"])
        )
        maybe_tune_leg_reveal_denoise(passes, options)

        client_id = str(uuid.uuid4())
        current_bytes = source_bytes
        pass_results: list[dict[str, Any]] = []
        job_id = str(job.get("id", uuid.uuid4()))

        for index, edit_pass in enumerate(passes, start=1):
            current_bytes, metadata = execute_pass(
                job_id=job_id,
                client_id=client_id,
                pass_index=index,
                stage_name="silhouette",
                current_image_bytes=current_bytes,
                edit_pass=edit_pass,
                options=options,
                mask_mode="silhouette",
                positive_prompt=build_positive_prompt(edit_pass),
                negative_prompt=build_negative_prompt(edit_pass),
                denoise=options["denoise"],
                steps=options["steps"],
            )
            pass_results.append(metadata)

            if should_run_target_refinement(edit_pass):
                current_bytes, metadata = execute_pass(
                    job_id=job_id,
                    client_id=client_id,
                    pass_index=index,
                    stage_name="target-refine",
                    current_image_bytes=current_bytes,
                    edit_pass=edit_pass,
                    options=options,
                    mask_mode="target",
                    positive_prompt=build_target_refine_positive_prompt(edit_pass),
                    negative_prompt=build_negative_prompt(edit_pass),
                    denoise=refinement_denoise(options),
                    steps=refinement_steps(options),
                )
                pass_results.append(metadata)

        if options["hair_cleanup"]:
            cleanup_pass = build_hair_cleanup_pass(options["instruction"])
            current_bytes, metadata = execute_pass(
                job_id=job_id,
                client_id=client_id,
                pass_index=len(passes) + 1,
                stage_name="hair-cleanup",
                current_image_bytes=current_bytes,
                edit_pass=cleanup_pass,
                options=options,
                mask_mode="target",
                positive_prompt=build_hair_cleanup_positive_prompt(options["instruction"]),
                negative_prompt=build_hair_cleanup_negative_prompt(),
                denoise=options["hair_cleanup_denoise"],
                steps=options["hair_cleanup_steps"],
            )
            pass_results.append(metadata)

        final_filename = f"{job_id}_final.png"
        artifact = encode_output_artifact(job_id, final_filename, current_bytes)
        return {
            "images": [artifact],
            "passes": pass_results,
            "preprocess": {
                "max_input_long_side": MAX_INPUT_LONG_SIDE,
                **resize_metadata,
                "target_width": options["target_width"],
                "target_height": options["target_height"],
            },
        }
    except (ValueError, InstructionParseError, requests.RequestException, RuntimeError, TimeoutError) as exc:
        LOGGER.exception("Job failed")
        return {"error": str(exc)}
    except Exception as exc:
        LOGGER.exception("Unexpected handler error")
        return {"error": f"Unexpected error: {exc}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
