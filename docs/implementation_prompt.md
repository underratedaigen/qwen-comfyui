# Implementation Prompt

Use this prompt as the exact brief for the next coding pass.

## Prompt

You are building a brand-new RunPod Serverless ComfyUI worker from scratch in:

`C:\Users\gusta\Documents\New project\qwen-v19-clothing-runpod-comfyui`

Do not reuse, inspect, or copy architecture from any older WAN-related repos in the workspace. Those are not the source of truth. The only source of truth is the new repo above plus its local docs.

Your job is to implement a narrow clothing-only image editing worker around `Phr00t/Qwen-Image-Edit-Rapid-AIO` `v19`.

## Product Goal

The API accepts:

- one source image
- one natural-language clothing instruction

It must:

- automatically detect the clothing region
- inpaint only that clothing region
- preserve everything outside the mask
- support edits like recolor, replace garment type, or remove a garment
- split multi-garment instructions into sequential passes
- run on RunPod Serverless through a GitHub-deployed ComfyUI worker

This is not a general editor. Do not add any full-image edit fallback.

## Core Constraints

1. Treat `Qwen-Rapid-AIO-NSFW-v19.safetensors` as a single AIO checkpoint loaded with `Load Checkpoint`.
2. Use Phr00t's fixed `TextEncodeQwenImageEditPlus` replacement.
3. Use automatic garment segmentation first, then crop-and-stitch inpainting.
4. Preserve identity, face, hair, skin, pose, body shape, hands, framing, lighting, and background.
5. If the clothing target cannot be resolved confidently, return a validation error instead of attempting a broad edit.
6. Keep the public API simple. The caller should not need to send raw ComfyUI JSON.

## Mandatory References

Read these local files first:

- `README.md`
- `docs/architecture.md`

Use these external references only as implementation support:

- Phr00t model card: <https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO>
- fixed text encode node: <https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main/fixed-textencode-node>
- human parser node: <https://github.com/cozymantis/human-parser-comfyui-node>
- crop and stitch node: <https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch>
- RunPod worker-comfyui: <https://github.com/runpod-workers/worker-comfyui>

## Exact Workflow Shape

Build around this single-pass ComfyUI graph:

1. `LoadImage`
2. `CheckpointLoaderSimple`
3. `HumanParserATR` or `HumanParserLIP`
4. mask grow/blur refinement
5. `Inpaint Crop`
6. `EmptyLatentImage`
7. `TextEncodeQwenImageEditPlus` for positive prompt
8. `TextEncodeQwenImageEditPlus` for negative prompt
9. `VAEEncode`
10. latent mask application for inpainting
11. `KSampler`
12. `VAEDecode`
13. `Inpaint Stitch`
14. `SaveImage`

The handler should reuse that graph for each clothing pass.

Use two workflow templates if needed:

- `workflow_templates/clothing_edit_single_pass_atr.json`
- `workflow_templates/clothing_edit_single_pass_lip.json`

Prefer two templates if that makes node class selection simpler.

## Parsing and Edit Strategy

Implement `instruction_parser.py` to turn a user instruction into one or more passes.

Supported clothing targets in the first version:

- dress
- skirt
- socks
- shirt
- top
- upper clothes
- coat
- pants
- trousers
- jumpsuit

Map them to parser categories. For example:

- `dress` -> `Dress`
- `skirt` -> `Skirt`
- `socks` -> `Socks`
- `shirt`, `top`, `upper clothes` -> `Upper-clothes`
- `coat` -> `Coat`
- `pants`, `trousers` -> `Pants`
- `jumpsuit` -> `Jumpsuits` or nearest available parser category

For a prompt like:

`change the dress to a red satin skirt and remove the socks`

produce sequential passes like:

```json
[
  {
    "category": "Dress",
    "edit_text": "change the dress to a red satin skirt"
  },
  {
    "category": "Socks",
    "edit_text": "remove the socks"
  }
]
```

Do not hardcode user examples into generation logic. Use them only as parser behavior examples.

If parsing fails or no supported garment is found, return a clear validation error.

## Prompt Construction

The handler should wrap the user instruction before sending it into Qwen.

Positive wrapper:

```text
Apply only the requested clothing edit inside the masked garment region.
Preserve identity, face, hair, skin, body shape, pose, hands, framing, lighting, and background.
Keep all unmasked regions unchanged.
User request: {instruction}
```

Negative wrapper:

```text
change face, change identity, different person, different hairstyle, different body, different pose,
background change, lighting change, extra garments, extra limbs, duplicate body parts, warped hands,
mask bleed, edits outside clothing, altered skin, altered hair
```

Add category-specific negatives when useful, such as:

- socks removal: `shoes changed`
- skirt edits: `legs changed`

## Defaults

Use these defaults unless the request overrides them:

- checkpoint: `Qwen-Rapid-AIO-NSFW-v19.safetensors`
- steps: `6`
- cfg: `1.0`
- sampler: `euler_ancestral`
- scheduler: `beta`
- parser model: `atr`
- mask grow: about `12 px`
- mask blur: about `6 px`
- crop context: about `128 px`

## Files You Need to Create

Implement these files:

- `Dockerfile`
- `custom_start.sh`
- `bootstrap_models.py`
- `handler.py`
- `instruction_parser.py`
- `workflow_builder.py`
- `requirements.txt`
- `.env.example`
- `test_input.json`

Keep these docs updated if needed:

- `README.md`
- `docs/architecture.md`

## Responsibilities Per File

### `Dockerfile`

- base it on `runpod/worker-comfyui:<version>-base`
- install Python dependencies
- install custom nodes or clone them during build
- copy repo files into the image
- start through `custom_start.sh`

### `custom_start.sh`

- bootstrap model files and any node overrides
- start ComfyUI
- wait until ComfyUI is healthy
- start the RunPod handler

### `bootstrap_models.py`

- download the Qwen v19 checkpoint if missing
- install or copy the fixed Qwen text encode node override
- download any required parser weights
- prefer network volume paths when available

### `instruction_parser.py`

- validate incoming instruction
- resolve clothing targets
- split multi-part instructions into sequential passes
- expose a clean structured output for the handler

### `workflow_builder.py`

- load workflow template JSON
- patch node ids and values cleanly
- inject filenames, category labels, prompts, seed, steps, sampler values, and output prefix
- support ATR and LIP templates

### `handler.py`

- accept `image_url` or `image_base64`
- validate inputs
- parse the instruction into passes
- upload images to ComfyUI
- run passes in sequence
- feed the output image from each pass into the next pass if there are multiple passes
- collect final output and return it as base64 or bucket URL
- return useful metadata including the passes used

### `requirements.txt`

- include only the dependencies the Python handler needs beyond the base image

### `.env.example`

- document the checkpoint URL, checkpoint filename, defaults, timeout settings, and optional HF token or bucket variables

### `test_input.json`

- provide a minimal working request example for local and RunPod testing

## Acceptance Criteria

The implementation is successful when:

1. The repo builds as a Docker image for RunPod Serverless.
2. The worker accepts a simple API request with an image and clothing instruction.
3. The handler converts the request into one or more structured clothing passes.
4. Each pass runs through a fixed internal ComfyUI workflow template.
5. Only the detected garment area is regenerated and the rest of the image is stitched back unchanged.
6. The final response returns a valid image payload and lists the passes applied.
7. Unsupported instructions fail safely instead of attempting a general edit.

## Working Rules

- Do not implement a broad full-image edit mode.
- Do not introduce unrelated features.
- Do not depend on the old WAN repo layout.
- Keep business logic in Python and keep workflow JSONs as frozen graph templates.
- If you cannot export final workflow JSONs from ComfyUI in the current environment, scaffold the code so the repo is ready for those templates and clearly mark that as the only remaining blocker.

## What To Do First

1. Read `README.md` and `docs/architecture.md`.
2. Create the repo files listed above.
3. Implement the parser and workflow-builder first.
4. Implement the RunPod handler next.
5. Implement Docker/bootstrap last.
6. Add a concise README update explaining how the worker is supposed to run.

## Expected Outcome

At the end, this repo should be a clean, self-contained RunPod Serverless worker scaffold for Qwen v19 clothing-only editing, with code ready to run once the final ComfyUI workflow templates are exported into `workflow_templates/`.
