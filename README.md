# Qwen v19 Clothing Edit RunPod Worker

This folder is a fresh RunPod Serverless ComfyUI worker scaffold for `Phr00t/Qwen-Image-Edit-Rapid-AIO` `v19`.

The endpoint is intentionally narrow:

- one source image in
- one natural-language clothing instruction in
- one edited image out
- automatic garment masking
- crop-and-stitch inpainting so unmasked pixels stay unchanged

## What It Uses

- `Qwen-Rapid-AIO-NSFW-v19.safetensors` as the main AIO checkpoint
- Phr00t's patched `TextEncodeQwenImageEditPlus`
- vendored `human-parser-comfyui-node` runtime files for ATR and LIP garment masks
- vendored `ComfyUI-Inpaint-CropAndStitch` runtime files for garment-only inpainting
- a custom RunPod handler so callers do not need to send raw ComfyUI workflow JSON

## API Shape

Minimal request:

```json
{
  "input": {
    "image_url": "https://example.com/source.png",
    "instruction": "change the dress to red satin and remove the socks"
  }
}
```

More controlled request:

```json
{
  "input": {
    "image_url": "https://example.com/source.png",
    "instruction": "change the dress to red satin and remove the socks",
    "seed": 42,
    "steps": 6,
    "cfg": 1.0,
    "sampler_name": "euler_ancestral",
    "scheduler": "beta",
    "parser_model": "atr",
    "target_width": 1024,
    "target_height": 1024,
    "mask_expand_pixels": 12,
    "mask_blend_pixels": 6,
    "context_expand_factor": 1.25
  }
}
```

The handler parses that instruction into one or more clothing passes, uploads the current image to ComfyUI, patches a frozen workflow template, and executes the passes sequentially.

## Startup Downloads

At worker startup, `bootstrap_models.py` will download:

- `Qwen-Rapid-AIO-NSFW-v19.safetensors`
- `exp-schp-201908301523-atr.pth`
- `exp-schp-201908261155-lip.pth`

The parser weights now default to direct Hugging Face URLs instead of Google Drive so cold-start downloads are simpler on RunPod.

## Build Shape

This repo now vendors the two required ComfyUI custom nodes directly under `custom_nodes/` instead of cloning them during `docker build`.

That makes the image build closer to the simpler WAN worker layout:

- root-level `Dockerfile`
- root-level startup script
- root-level handler
- vendored ComfyUI node code already present in the repository

Pinned vendored sources are listed in `docs/vendored_dependencies.md`.

## Main Files

- `Dockerfile`
- `custom_start.sh`
- `bootstrap_models.py`
- `instruction_parser.py`
- `workflow_builder.py`
- `handler.py`
- `custom_nodes/human-parser-comfyui-node`
- `custom_nodes/ComfyUI-Inpaint-CropAndStitch`
- `workflow_templates/clothing_edit_single_pass_atr.json`
- `workflow_templates/clothing_edit_single_pass_lip.json`
- `patches/nodes_qwen.py`
- `.env.example`
- `test_input.json`

## Current Status

This is now an implementation scaffold, not just a plan:

- Docker/runtime files exist
- parser and workflow builder exist
- the RunPod handler exists
- frozen workflow templates exist

The remaining practical validation step is to run this against a live ComfyUI container and confirm the hand-authored API workflow JSON matches the currently installed node input schema exactly.

## Deployment Notes

Recommended deployment paths:

- direct from GitHub repository
- or GitHub Actions to `ghcr.io` as a fallback

For GitHub repository deployment, use:

- Build context: `.`
- Dockerfile path: `Dockerfile`

This repo now avoids cloning custom node repos during the Docker build, which keeps the GitHub build path more deterministic.

Fallback path:

- Build the image in GitHub Actions
- Push it to `ghcr.io`
- Deploy the image to RunPod with `Import from Docker Registry`

This keeps the source of truth on GitHub but avoids RunPod's GitHub builder, which has recently been failing for some users before the Docker build stage even begins.

The workflow added in `.github/workflows/build-ghcr.yml` publishes:

- `ghcr.io/underratedaigen/qwen-comfyui:sha-<commit>`
- `ghcr.io/underratedaigen/qwen-comfyui:main`
- `ghcr.io/underratedaigen/qwen-comfyui:<git-tag>` for release tags

Recommended starting point:

- Endpoint type: `Queue`
- GPU: `A100 80GB` or comparable
- Active workers: `0`
- Max workers: `1`
- GPUs per worker: `1`
- Flash boot: enabled
- Attach a network volume if you want the checkpoint and parser weights cached across cold starts

If the package is private on first publish, either:

- change the GHCR package visibility to public in GitHub Packages
- or add registry credentials in RunPod for `ghcr.io`

## References

- [Phr00t model card](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO)
- [Fixed Qwen node](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main/fixed-textencode-node)
- [Human parser node](https://github.com/cozymantis/human-parser-comfyui-node)
- [Crop and stitch node](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch)
- [RunPod worker-comfyui](https://github.com/runpod-workers/worker-comfyui)
