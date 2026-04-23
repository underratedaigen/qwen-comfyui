from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random

from instruction_parser import EditPass


LOAD_IMAGE_NODE_ID = "1"
CHECKPOINT_NODE_ID = "2"
PARSER_NODE_ID = "3"
CROP_NODE_ID = "4"
EMPTY_LATENT_NODE_ID = "5"
POSITIVE_NODE_ID = "6"
NEGATIVE_NODE_ID = "7"
VAE_ENCODE_NODE_ID = "8"
SET_NOISE_MASK_NODE_ID = "9"
KSAMPLER_NODE_ID = "10"
VAE_DECODE_NODE_ID = "11"
STITCH_NODE_ID = "12"
SAVE_IMAGE_NODE_ID = "13"

ATR_TEMPLATE_NAME = "clothing_edit_single_pass_atr.json"
LIP_TEMPLATE_NAME = "clothing_edit_single_pass_lip.json"

ATR_BOOLEAN_FIELDS = (
    "background",
    "hat",
    "hair",
    "sunglasses",
    "upper_clothes",
    "skirt",
    "pants",
    "dress",
    "belt",
    "left_shoe",
    "right_shoe",
    "face",
    "left_leg",
    "right_leg",
    "left_arm",
    "right_arm",
    "bag",
    "scarf",
)

LIP_BOOLEAN_FIELDS = (
    "background",
    "hat",
    "hair",
    "glove",
    "sunglasses",
    "upper_clothes",
    "dress",
    "coat",
    "socks",
    "pants",
    "jumpsuits",
    "scarf",
    "skirt",
    "face",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "left_shoe",
    "right_shoe",
)

ATR_SILHOUETTE_MINUS_HEAD_FIELDS = (
    "upper_clothes",
    "skirt",
    "pants",
    "dress",
    "belt",
    "left_shoe",
    "right_shoe",
    "left_leg",
    "right_leg",
    "left_arm",
    "right_arm",
    "bag",
    "scarf",
)

LIP_SILHOUETTE_MINUS_HEAD_FIELDS = (
    "glove",
    "upper_clothes",
    "dress",
    "coat",
    "socks",
    "pants",
    "jumpsuits",
    "scarf",
    "skirt",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "left_shoe",
    "right_shoe",
)


def coerce_seed(seed: int | str | None) -> int:
    if seed is None:
        return random.randint(0, 2**63 - 1)
    seed_value = int(seed)
    if seed_value < 0:
        return random.randint(0, 2**63 - 1)
    return seed_value


def load_template(template_path: Path) -> dict:
    return json.loads(template_path.read_text(encoding="utf-8"))


def template_name_for_parser(parser_type: str) -> str:
    if parser_type == "atr":
        return ATR_TEMPLATE_NAME
    if parser_type == "lip":
        return LIP_TEMPLATE_NAME
    raise ValueError(f"Unsupported parser_type '{parser_type}'")


def _boolean_fields_for_parser(parser_type: str) -> tuple[str, ...]:
    if parser_type == "atr":
        return ATR_BOOLEAN_FIELDS
    if parser_type == "lip":
        return LIP_BOOLEAN_FIELDS
    raise ValueError(f"Unsupported parser_type '{parser_type}'")


def _silhouette_minus_head_fields(parser_type: str) -> tuple[str, ...]:
    if parser_type == "atr":
        return ATR_SILHOUETTE_MINUS_HEAD_FIELDS
    if parser_type == "lip":
        return LIP_SILHOUETTE_MINUS_HEAD_FIELDS
    raise ValueError(f"Unsupported parser_type '{parser_type}'")


def build_workflow(
    *,
    template_dir: Path,
    input_image_name: str,
    checkpoint_name: str,
    edit_pass: EditPass,
    mask_mode: str,
    positive_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    target_width: int,
    target_height: int,
    mask_expand_pixels: int,
    mask_blend_pixels: int,
    context_expand_factor: float,
    output_padding: int,
    device_mode: str,
    filename_prefix: str,
) -> dict:
    template_path = template_dir / template_name_for_parser(edit_pass.parser_type)
    workflow = deepcopy(load_template(template_path))

    workflow[LOAD_IMAGE_NODE_ID]["inputs"]["image"] = input_image_name
    workflow[CHECKPOINT_NODE_ID]["inputs"]["ckpt_name"] = checkpoint_name

    parser_inputs = workflow[PARSER_NODE_ID]["inputs"]
    if mask_mode == "silhouette":
        selected_fields = set(_silhouette_minus_head_fields(edit_pass.parser_type))
    elif mask_mode == "target":
        selected_fields = {edit_pass.parser_field}
    else:
        raise ValueError(f"Unsupported mask_mode '{mask_mode}'")
    for field_name in _boolean_fields_for_parser(edit_pass.parser_type):
        parser_inputs[field_name] = field_name in selected_fields

    crop_inputs = workflow[CROP_NODE_ID]["inputs"]
    crop_inputs["mask_expand_pixels"] = int(mask_expand_pixels)
    crop_inputs["mask_blend_pixels"] = int(mask_blend_pixels)
    crop_inputs["context_from_mask_extend_factor"] = float(context_expand_factor)
    crop_inputs["output_target_width"] = int(target_width)
    crop_inputs["output_target_height"] = int(target_height)
    crop_inputs["output_padding"] = str(int(output_padding))
    crop_inputs["device_mode"] = "gpu (much faster)" if device_mode == "gpu" else "cpu (compatible)"

    workflow[EMPTY_LATENT_NODE_ID]["inputs"]["width"] = int(target_width)
    workflow[EMPTY_LATENT_NODE_ID]["inputs"]["height"] = int(target_height)

    workflow[POSITIVE_NODE_ID]["inputs"]["prompt"] = positive_prompt
    workflow[NEGATIVE_NODE_ID]["inputs"]["prompt"] = negative_prompt

    workflow[KSAMPLER_NODE_ID]["inputs"]["seed"] = int(seed)
    workflow[KSAMPLER_NODE_ID]["inputs"]["steps"] = int(steps)
    workflow[KSAMPLER_NODE_ID]["inputs"]["cfg"] = float(cfg)
    workflow[KSAMPLER_NODE_ID]["inputs"]["sampler_name"] = sampler_name
    workflow[KSAMPLER_NODE_ID]["inputs"]["scheduler"] = scheduler
    workflow[KSAMPLER_NODE_ID]["inputs"]["denoise"] = float(denoise)

    workflow[SAVE_IMAGE_NODE_ID]["inputs"]["filename_prefix"] = filename_prefix

    return workflow
