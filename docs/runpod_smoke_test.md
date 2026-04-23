# RunPod Smoke Test

Use this checklist when you push this repo to GitHub and connect it to RunPod.

## GitHub Build Settings

- Repository: your repo containing `qwen-v19-clothing-runpod-comfyui/`
- Branch: the branch you want RunPod to build
- Build context: `qwen-v19-clothing-runpod-comfyui`
- Dockerfile path: `Dockerfile`

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

- The image builds successfully on RunPod from GitHub.
- Worker startup downloads the checkpoint and parser weights.
- ComfyUI starts without node import errors.
- The handler returns an `images` array.
- Multi-pass requests return a `passes` array showing the resolved clothing passes.

## Most Likely Failure Points

- Qwen checkpoint download URL or size-related cold start delays
- parser model downloads
- `nodes_qwen.py` replacement path
- workflow JSON input names not matching the installed node schema
- ComfyUI startup timeout being too low for first cold boot

## If The First Run Fails

Check, in this order:

1. Build logs for Docker image failures
2. Worker startup logs for checkpoint or parser download failures
3. ComfyUI logs for custom-node import errors
4. Prompt execution logs for workflow validation errors
5. Response payload for the exact failing node id or schema mismatch
