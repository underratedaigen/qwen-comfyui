# RunPod Smoke Test

Use this checklist with the GitHub Actions to GHCR workflow, then deploy the published image to RunPod from the container registry.

## Why This Path

The direct RunPod GitHub builder has been intermittently failing for some users at:

- `Successfully cloned repository ...`
- `Creating cache directory.`

with no Docker build steps after that. This repo now uses GitHub Actions to build the image and `ghcr.io` to deliver it to RunPod instead.

## GitHub Build Step

1. Push to `main`, publish a tag, or run the `Build and Push GHCR Image` workflow manually.
2. Wait for the workflow to finish in GitHub Actions.
3. Copy the published image tag.

Preferred tag style:

- `ghcr.io/underratedaigen/qwen-comfyui:sha-<commit>`

Convenience tags:

- `ghcr.io/underratedaigen/qwen-comfyui:main`
- `ghcr.io/underratedaigen/qwen-comfyui:<git-tag>`

## GHCR Visibility

After the first publish, make sure the package can be pulled by RunPod:

- easiest: set the GHCR package to `Public`
- otherwise: create registry credentials for `ghcr.io` in RunPod and use those on the endpoint

## RunPod Deploy Settings

In RunPod:

1. Click `New Endpoint`
2. Choose `Import from Docker Registry`
3. Use the published image URL, for example:
   `ghcr.io/underratedaigen/qwen-comfyui:sha-<commit>`

## Recommended Endpoint Settings

- Endpoint type: `Queue`
- GPU: `A100 80GB` or similar
- Active workers: `0`
- Max workers: `1`
- GPUs per worker: `1`
- Flash boot: enabled
- Idle timeout: `300-900`
- Execution timeout: `1800-3600`

If possible, attach a network volume so the Qwen checkpoint and parser weights persist across cold starts.

## Environment Variables

Start with:

```text
QWEN_CHECKPOINT_NAME=Qwen-Rapid-AIO-NSFW-v19.safetensors
QWEN_CHECKPOINT_URL=https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v19/Qwen-Rapid-AIO-NSFW-v19.safetensors
QWEN_DEFAULT_STEPS=6
QWEN_DEFAULT_CFG=1.0
QWEN_DEFAULT_SAMPLER=euler_ancestral
QWEN_DEFAULT_SCHEDULER=beta
QWEN_DEFAULT_DENOISE=1.0
QWEN_DEFAULT_PARSER_MODEL=atr
SCHP_ATR_MODEL_URL=https://huggingface.co/panyanyany/Self-Correction-Human-Parsing/resolve/main/schp/exp-schp-201908301523-atr.pth
SCHP_LIP_MODEL_URL=https://huggingface.co/panyanyany/Self-Correction-Human-Parsing/resolve/main/schp/exp-schp-201908261155-lip.pth
COMFY_LOG_LEVEL=INFO
COMFY_HISTORY_TIMEOUT_S=1800
COMFY_POLL_INTERVAL_S=2
COMFY_STARTUP_TIMEOUT_S=300
COMFY_STARTUP_POLL_INTERVAL_S=2
```

Optional:

```text
HF_TOKEN=your_huggingface_token
BUCKET_ENDPOINT_URL=...
BUCKET_ACCESS_KEY_ID=...
BUCKET_SECRET_ACCESS_KEY=...
```

## First Smoke Test Request

Start with a single-target edit before testing multi-pass behavior.

```json
{
  "input": {
    "image_url": "https://example.com/source.png",
    "instruction": "change the dress to red satin",
    "seed": 42,
    "steps": 6,
    "cfg": 1.0,
    "sampler_name": "euler_ancestral",
    "scheduler": "beta",
    "parser_model": "atr"
  }
}
```

Then test a multi-pass instruction:

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
    "parser_model": "atr"
  }
}
```

## What Success Looks Like

- The GitHub Actions build publishes successfully to GHCR.
- RunPod pulls the GHCR image successfully.
- Worker startup downloads the checkpoint and parser weights.
- ComfyUI starts without node import errors.
- The handler returns an `images` array.
- Multi-pass requests return a `passes` array showing the resolved clothing passes.

## Most Likely Failure Points

- GHCR package visibility or registry auth
- Qwen checkpoint download URL or size-related cold start delays
- parser model downloads
- `nodes_qwen.py` replacement path
- workflow JSON input names not matching the installed node schema
- ComfyUI startup timeout being too low for first cold boot

## If The First Run Fails

Check, in this order:

1. GitHub Actions logs for Docker build failures
2. RunPod startup logs for image pull or auth failures
3. Worker startup logs for checkpoint or parser download failures
4. ComfyUI logs for custom-node import errors
5. Prompt execution logs for workflow validation errors
6. Response payload for the exact failing node id or schema mismatch
