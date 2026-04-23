FROM runpod/worker-comfyui:5.8.5-base

ENV PYTHONUNBUFFERED=1 \
    COMFY_LOG_LEVEL=INFO \
    QWEN_CHECKPOINT_NAME=Qwen-Rapid-AIO-NSFW-v19.safetensors \
    QWEN_CHECKPOINT_URL=https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v19/Qwen-Rapid-AIO-NSFW-v19.safetensors \
    QWEN_DEFAULT_STEPS=6 \
    QWEN_DEFAULT_CFG=1.0 \
    QWEN_DEFAULT_SAMPLER=euler_ancestral \
    QWEN_DEFAULT_SCHEDULER=beta \
    QWEN_DEFAULT_DENOISE=1.0 \
    QWEN_DEFAULT_PARSER_MODEL=atr \
    QWEN_DEFAULT_FILENAME_PREFIX=qwen-v19-clothing/output \
    QWEN_DEFAULT_TARGET_WIDTH=1024 \
    QWEN_DEFAULT_TARGET_HEIGHT=1024 \
    QWEN_DEFAULT_MASK_EXPAND_PIXELS=12 \
    QWEN_DEFAULT_MASK_BLEND_PIXELS=6 \
    QWEN_DEFAULT_CONTEXT_EXPAND_FACTOR=1.25 \
    QWEN_DEFAULT_OUTPUT_PADDING=32 \
    QWEN_DEFAULT_DEVICE_MODE=gpu \
    QWEN_SKIP_MODEL_DOWNLOAD=false \
    QWEN_SKIP_PARSER_DOWNLOAD=false \
    SCHP_ATR_MODEL_URL=https://huggingface.co/panyanyany/Self-Correction-Human-Parsing/resolve/main/schp/exp-schp-201908301523-atr.pth \
    SCHP_ATR_MODEL_FILENAME=exp-schp-201908301523-atr.pth \
    SCHP_LIP_MODEL_URL=https://huggingface.co/panyanyany/Self-Correction-Human-Parsing/resolve/main/schp/exp-schp-201908261155-lip.pth \
    SCHP_LIP_MODEL_FILENAME=exp-schp-201908261155-lip.pth \
    COMFY_HISTORY_TIMEOUT_S=1800 \
    COMFY_POLL_INTERVAL_S=2 \
    COMFY_STARTUP_TIMEOUT_S=300 \
    COMFY_STARTUP_POLL_INTERVAL_S=2

COPY requirements.txt /tmp/requirements.txt

RUN uv pip install -r /tmp/requirements.txt

COPY custom_nodes /comfyui/custom_nodes
COPY patches /patches
COPY workflow_templates /workflow_templates
COPY docs /docs
COPY README.md /README.md
COPY bootstrap_models.py /bootstrap_models.py
COPY custom_start.sh /start.sh
COPY handler.py /handler.py
COPY instruction_parser.py /instruction_parser.py
COPY workflow_builder.py /workflow_builder.py
COPY .env.example /.env.example
COPY test_input.json /test_input.json

RUN chmod +x /start.sh

CMD ["/start.sh"]
