# Next Step Prompt

Use this prompt for the next implementation pass.

## Prompt

You are continuing work in:

`C:\Users\gusta\Documents\New project\qwen-v19-clothing-runpod-comfyui`

Do not redesign the project. Do not expand scope. The goal of this pass is to validate and harden the existing worker scaffold through a real build-and-test cycle.

## Current State

This repo already contains:

- `Dockerfile`
- `custom_start.sh`
- `bootstrap_models.py`
- `instruction_parser.py`
- `workflow_builder.py`
- `handler.py`
- `workflow_templates/clothing_edit_single_pass_atr.json`
- `workflow_templates/clothing_edit_single_pass_lip.json`
- `patches/nodes_qwen.py`
- docs and test input files

The parser and workflow builder compile locally. The workflow templates load as JSON. What is not yet proven is whether the full ComfyUI runtime accepts the current templates and whether the worker behaves correctly in a real container.

## Goal Of This Pass

Run the first true end-to-end validation cycle and fix anything needed so the worker is genuinely ready for GitHub deployment to RunPod.

## Main Tasks

1. Build the Docker image for this repo.
2. Start the container locally if the environment allows it.
3. Confirm that:
   - the custom nodes install correctly
   - the patched `nodes_qwen.py` is copied into the right ComfyUI location
   - the Qwen v19 checkpoint download path is correct
   - the SCHP ATR and LIP model download flow works
   - ComfyUI starts cleanly
4. Verify that the hand-authored workflow templates match the actual installed node schemas.
5. If the templates do not match, fix either:
   - the template JSON node inputs
   - the workflow builder patching logic
   - or the runtime assumptions in the handler
6. Run at least one real handler test request against the worker using:
   - one source image
   - one simple clothing instruction
7. Confirm the response shape is correct and that the multi-pass logic still works for a two-target instruction.
8. Update docs and test files if any request fields or runtime details changed during validation.

## Constraints

- Do not add new product features.
- Do not turn this into a general full-image editing endpoint.
- Do not reintroduce the old WAN repo ideas.
- Keep the endpoint clothing-only.
- Preserve the current API shape unless runtime validation forces a small change.
- If a runtime issue appears, fix the root cause rather than papering over it.

## Specific Things To Check Carefully

### Workflow correctness

Make sure these node assumptions are real in the installed ComfyUI environment:

- `CheckpointLoaderSimple`
- `Cozy Human Parser ATR`
- `Cozy Human Parser LIP`
- `InpaintCropImproved`
- `InpaintStitchImproved`
- `TextEncodeQwenImageEditPlus`
- `SetLatentNoiseMask`
- `KSampler`
- `SaveImage`

If any node name or input key is wrong, correct the workflow template and the builder.

### Qwen patch placement

Confirm that replacing `/comfyui/comfy_extras/nodes_qwen.py` is the correct runtime path for the installed ComfyUI image version. If the path is different, fix `bootstrap_models.py`.

### Parser model downloads

The human parser repo expects:

- `models/schp/exp-schp-201908301523-atr.pth`
- `models/schp/exp-schp-201908261155-lip.pth`

Confirm that the bootstrap logic places them where the node will actually find them in the container.

### Handler execution flow

Confirm the order is safe:

1. wait for ComfyUI
2. parse instruction
3. upload image
4. open websocket
5. queue prompt
6. wait for execution complete
7. fetch history
8. fetch output image
9. feed output into next pass if needed

If there is a race condition or missing error handling, fix it.

## Suggested Validation Path

1. Run a Docker build from the repo root for Linux amd64.
2. Start the image with the minimum env vars needed.
3. Inspect logs for startup failures.
4. If startup works, call the handler locally with `test_input.json` or a small custom request.
5. If the worker cannot be fully run locally in the current environment, at least run every feasible step short of the blocked one and document the exact blocker.

## Acceptance Criteria

This pass is successful when:

1. The Docker image builds successfully, or the exact blocker is identified and fixed if possible.
2. The container startup path is validated.
3. The workflow templates are confirmed or corrected against the live node schemas.
4. The handler can execute at least one real request, or the exact remaining runtime blocker is isolated clearly.
5. The repo ends this pass closer to actual GitHub-to-RunPod deployment, not just theoretical readiness.

## Deliverables

At the end of the pass:

- keep all code fixes in the repo
- update `README.md` if needed
- mention exactly what was validated successfully
- mention any remaining blocker plainly
- if everything works, state the exact next step for GitHub push and RunPod deployment
