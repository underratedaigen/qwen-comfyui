# RunPod Pod Deployment

This branch adds a Pod-only API path while leaving the Serverless entrypoint unchanged.

## Build Settings

Use these settings when creating the Pod image from GitHub:

```text
Branch: codex/pod-api
Build context: .
Dockerfile path: Dockerfile.pod
```

The Pod image defaults to `/pod_start.sh`, so a start command override is optional. If RunPod asks for one, use:

```text
/pod_start.sh
```

Expose HTTP port:

```text
8000
```

Your Pod API URL will look like:

```text
https://<pod-id>-8000.proxy.runpod.net
```

## Environment

Set at least:

```env
POD_API_KEY=<random-long-secret>
POD_API_PORT=8000
QWEN_MODEL_CACHE_ROOT=/workspace/models
QWEN_DEFAULT_STEPS=6
QWEN_DEFAULT_PARSER_MODEL=lip
QWEN_DEFAULT_DENOISE=1.0
QWEN_AUTO_TARGET_LONG_SIDE=1280
QWEN_MAX_INPUT_LONG_SIDE=1920
QWEN_DEFAULT_HAIR_CLEANUP=false
```

`QWEN_MODEL_CACHE_ROOT=/workspace/models` keeps the checkpoint and parser files on the Pod volume disk, so stop/start should reuse them.

## API

All generation routes require:

```text
Authorization: Bearer <POD_API_KEY>
```

Routes:

```text
GET  /health
POST /run
GET  /status/{job_id}
POST /runsync
```

`POST /run` accepts the same body shape as Serverless:

```json
{
  "input": {
    "image_url": "https://example.com/source.png",
    "instruction": "change the dress to red satin"
  }
}
```

## Local Tester

Start the tester as usual:

```text
py local_qwen_tester.py
```

Choose `Pod API URL`, paste the Pod proxy URL, and use `POD_API_KEY` as the API key.
