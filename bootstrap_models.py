from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("qwen-v19-bootstrap")

PROJECT_ROOT = Path(__file__).resolve().parent
PATCHED_QWEN_NODE = PROJECT_ROOT / "patches" / "nodes_qwen.py"
COMFY_QWEN_NODE = Path("/comfyui/comfy_extras/nodes_qwen.py")
RUNPOD_VOLUME_ROOT = Path("/runpod-volume")
RUNPOD_MODELS_ROOT = RUNPOD_VOLUME_ROOT / "models"
COMFY_MODELS_ROOT = Path("/comfyui/models")

QWEN_CHECKPOINT_NAME = os.environ.get(
    "QWEN_CHECKPOINT_NAME",
    "Qwen-Rapid-AIO-NSFW-v19.safetensors",
)
QWEN_CHECKPOINT_URL = os.environ.get(
    "QWEN_CHECKPOINT_URL",
    "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v19/Qwen-Rapid-AIO-NSFW-v19.safetensors",
)
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

SCHP_ATR_MODEL_URL = os.environ.get(
    "SCHP_ATR_MODEL_URL",
    "https://huggingface.co/panyanyany/Self-Correction-Human-Parsing/resolve/main/schp/exp-schp-201908301523-atr.pth",
)
SCHP_ATR_MODEL_FILENAME = os.environ.get(
    "SCHP_ATR_MODEL_FILENAME",
    "exp-schp-201908301523-atr.pth",
)
SCHP_LIP_MODEL_URL = os.environ.get(
    "SCHP_LIP_MODEL_URL",
    "https://huggingface.co/panyanyany/Self-Correction-Human-Parsing/resolve/main/schp/exp-schp-201908261155-lip.pth",
)
SCHP_LIP_MODEL_FILENAME = os.environ.get(
    "SCHP_LIP_MODEL_FILENAME",
    "exp-schp-201908261155-lip.pth",
)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def preferred_checkpoint_dir() -> Path:
    if RUNPOD_VOLUME_ROOT.is_dir():
        path = RUNPOD_MODELS_ROOT / "checkpoints"
    else:
        path = COMFY_MODELS_ROOT / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def preferred_schp_cache_dir() -> Path:
    if RUNPOD_VOLUME_ROOT.is_dir():
        path = RUNPOD_MODELS_ROOT / "schp"
    else:
        path = COMFY_MODELS_ROOT / "schp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_comfy_schp_link(cache_dir: Path) -> Path:
    comfy_schp_dir = COMFY_MODELS_ROOT / "schp"
    comfy_schp_dir.parent.mkdir(parents=True, exist_ok=True)

    if comfy_schp_dir.is_symlink():
        target = comfy_schp_dir.resolve(strict=False)
        if target == cache_dir:
            return cache_dir
        comfy_schp_dir.unlink()

    if comfy_schp_dir.exists():
        return comfy_schp_dir

    if cache_dir != comfy_schp_dir:
        try:
            comfy_schp_dir.symlink_to(cache_dir, target_is_directory=True)
            LOGGER.info("Created symlink %s -> %s", comfy_schp_dir, cache_dir)
        except OSError:
            comfy_schp_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.warning("Could not create schp symlink, falling back to %s", comfy_schp_dir)
            return comfy_schp_dir

    return comfy_schp_dir


def copy_qwen_patch() -> None:
    if not PATCHED_QWEN_NODE.is_file():
        raise FileNotFoundError(f"Missing patched Qwen node at {PATCHED_QWEN_NODE}")

    COMFY_QWEN_NODE.parent.mkdir(parents=True, exist_ok=True)

    desired = PATCHED_QWEN_NODE.read_text(encoding="utf-8")
    current = COMFY_QWEN_NODE.read_text(encoding="utf-8") if COMFY_QWEN_NODE.exists() else None
    if current == desired:
        LOGGER.info("Patched nodes_qwen.py already in place")
        return

    shutil.copy2(PATCHED_QWEN_NODE, COMFY_QWEN_NODE)
    LOGGER.info("Installed patched Qwen node to %s", COMFY_QWEN_NODE)


def download_http(url: str, destination: Path) -> None:
    headers = {}
    if HF_TOKEN and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        with NamedTemporaryFile(delete=False, dir=str(destination.parent)) as tmp_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp_file.write(chunk)
            temp_path = Path(tmp_file.name)

    temp_path.replace(destination)


def ensure_file(destination: Path, source_url: str) -> None:
    if destination.exists():
        LOGGER.info("Found %s", destination)
        return

    LOGGER.info("Downloading %s", destination.name)
    download_http(source_url, destination)
    LOGGER.info("Downloaded %s", destination)


def main() -> None:
    copy_qwen_patch()

    if not env_flag("QWEN_SKIP_MODEL_DOWNLOAD", default=False):
        checkpoint_path = preferred_checkpoint_dir() / QWEN_CHECKPOINT_NAME
        ensure_file(checkpoint_path, QWEN_CHECKPOINT_URL)

    if env_flag("QWEN_SKIP_PARSER_DOWNLOAD", default=False):
        LOGGER.info("Skipping parser model downloads because QWEN_SKIP_PARSER_DOWNLOAD=true")
        return

    schp_cache_dir = preferred_schp_cache_dir()
    comfy_schp_dir = ensure_comfy_schp_link(schp_cache_dir)

    atr_destination = comfy_schp_dir / SCHP_ATR_MODEL_FILENAME
    lip_destination = comfy_schp_dir / SCHP_LIP_MODEL_FILENAME

    ensure_file(atr_destination, SCHP_ATR_MODEL_URL)
    ensure_file(lip_destination, SCHP_LIP_MODEL_URL)


if __name__ == "__main__":
    main()
