# Architecture

## Objective

Build a RunPod Serverless ComfyUI worker that performs clothing-only edits with `Qwen-Rapid-AIO-NSFW-v19.safetensors` while changing as little as possible outside the garment mask.

This worker is not a general image editing endpoint. It is a constrained clothing editor.

## Sources

- Phr00t model card: <https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO>
- Fixed Qwen text-encode node: <https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/tree/main/fixed-textencode-node>
- Cozy human parser node: <https://github.com/cozymantis/human-parser-comfyui-node>
- Inpaint Crop & Stitch: <https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch>
- RunPod worker-comfyui base: <https://github.com/runpod-workers/worker-comfyui>

## Core Design Decisions

### 1. Treat the checkpoint as a single AIO artifact

`Qwen-Rapid-AIO-NSFW-v19.safetensors` is loaded through a plain `Load Checkpoint` node.

We do not plan around separate model, CLIP, or VAE downloads for this repo.

### 2. Mask first, prompt second

The strongest guarantee that only clothing changes is not the prompt wording. It is the automatic garment mask plus crop-and-stitch inpainting.

### 3. The handler owns instruction parsing

ComfyUI should receive already-resolved target categories and refined prompts.

The Python handler will:

- parse the freeform instruction
- map it to garment categories
- decide whether to run one pass or multiple sequential passes
- build the final prompt wrapper for each pass

### 4. Multi-pass is required

Requests like "change the dress to a skirt and remove socks" are safer as separate inpaint passes:

- pass 1: edit `Dress`
- pass 2: edit `Socks`

This prevents one giant mask from causing collateral edits.

## Recommended Custom Nodes

### Required

- `human-parser-comfyui-node`
  - use `ATR` or `LIP`
  - `LIP` categories include `Upper-clothes`, `Dress`, `Coat`, `Socks`, `Pants`, `Jumpsuits`, `Skirt`, `Face`, `Hair`
  - `ATR` categories include `Upper-clothes`, `Skirt`, `Pants`, `Dress`, `Face`, `Hair`
- `ComfyUI-Inpaint-CropAndStitch`
- Phr00t fixed `TextEncodeQwenImageEditPlus` node replacement

### Optional

- a mask grow or blur utility node if the crop-and-stitch node does not already cover all needed mask refinement controls
- a logic node pack only if conditional routing inside ComfyUI becomes necessary

## Node-Level Workflow

This worker should use one exported single-pass workflow template. The handler will reuse it for each clothing target.

The workflow below is the exact graph shape we should create in ComfyUI.

### Single-Pass Graph

```text
[1] LoadImage
  input: source image uploaded by handler
  output: source_image

[2] CheckpointLoaderSimple
  input: Qwen-Rapid-AIO-NSFW-v19.safetensors
  output: model, clip, vae

[3] HumanParserATR or HumanParserLIP
  input: source_image
  params: parser selected by handler, category selected by handler
  output: raw_garment_mask

[4] Mask Blur/Grow Refinement
  input: raw_garment_mask
  params:
    grow = 8 to 18 px
    blur = 4 to 12 px
    fill_holes = true
  output: refined_mask

[5] Inpaint Crop
  input:
    image = source_image
    mask = refined_mask
  params:
    context_expand_pixels = 96 to 192
    fill_mask_holes = true
    blur_mask_pixels = 0 if already blurred upstream
    rescale_mode = resize_to_target
    target_resolution = 1024 or 1280 long side equivalent
  output:
    cropped_image
    cropped_mask
    stitch_context

[6] EmptyLatentImage
  input:
    width = crop width
    height = crop height
  output: sample_latent

[7] TextEncodeQwenImageEditPlus
  input:
    clip from [2]
    image_1 = cropped_image
    latent = sample_latent
    prompt = pass prompt built by handler
  output: positive_conditioning

[8] TextEncodeQwenImageEditPlus
  input:
    clip from [2]
    image_1 = cropped_image
    latent = sample_latent
    prompt = negative prompt built by handler
  output: negative_conditioning

[9] VAEEncode
  input:
    image = cropped_image
    vae from [2]
  output: crop_latent

[10] SetLatentNoiseMask or equivalent inpaint-conditioning path
  input:
    samples = crop_latent
    mask = cropped_mask
  output: masked_latent

[11] KSampler
  input:
    model from [2]
    positive from [7]
    negative from [8]
    latent_image = masked_latent
  params:
    seed = handler input
    steps = default 6
    cfg = 1.0
    sampler_name = euler_ancestral
    scheduler = beta
    denoise = 1.0 for replacement edits, lower only after testing
  output: sampled_latent

[12] VAEDecode
  input:
    samples = sampled_latent
    vae from [2]
  output: edited_crop

[13] Inpaint Stitch
  input:
    edited_crop
    stitch_context from [5]
  output: stitched_image

[14] SaveImage
  input:
    stitched_image
  output: final image file
```

## Why This Graph

### Human parser before crop

We want the mask to be created from the full image so garment boundaries are found in the original context.

### Crop before sampling

This sharply reduces collateral edits because Qwen only regenerates the relevant garment area with nearby context.

### Qwen conditioning uses the cropped image

The image edit prompt should describe edits relative to the garment crop, not the whole scene.

### The original image is only modified through stitch-back

That means everything outside the mask remains unchanged at pixel level.

## Workflow Template Strategy

We should export exactly one ComfyUI workflow JSON:

- `workflow_templates/clothing_edit_single_pass.json`

The handler will patch these values at runtime:

- uploaded input filename
- checkpoint name
- parser node choice if there are separate ATR and LIP templates
- garment category name
- prompt
- negative prompt
- seed
- steps
- cfg
- sampler
- scheduler
- filename prefix

If switching parser type requires different node classes, keep two templates instead:

- `clothing_edit_single_pass_atr.json`
- `clothing_edit_single_pass_lip.json`

I would start with two templates if ComfyUI export IDs stay cleaner that way.

## Prompting Strategy

The public API accepts only the user instruction. The handler wraps it.

### Positive prompt template

```text
Apply only the requested clothing edit inside the masked garment region.
Preserve identity, face, hair, skin, body shape, pose, hands, framing, lighting, and background.
Keep all unmasked regions unchanged.
User request: {instruction}
```

### Negative prompt template

```text
change face, change identity, different person, different hairstyle, different body, different pose,
background change, lighting change, extra garments, extra limbs, duplicate body parts, warped hands,
mask bleed, edits outside clothing, altered skin, altered hair
```

The handler should optionally add category-aware negatives. Example:

- for socks removal: add `shoes changed`
- for skirt edits: add `legs changed`

## Instruction Parsing Rules

Instruction parsing belongs in `instruction_parser.py`.

### Supported targets

Initial supported clothing targets:

- `dress`
- `skirt`
- `socks`
- `shirt`
- `top`
- `upper clothes`
- `coat`
- `pants`
- `trousers`
- `jumpsuit`

### Category mapping

The parser should output one or more passes:

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

### Multi-target parsing

Split on conjunctions only after preserving phrase meaning:

- `change the dress to red and remove socks`
- `replace the coat with a leather jacket and darken the skirt`

The parser should prefer explicit clothing nouns over color-only phrases.

### Ambiguity fallback

If no supported garment is detected:

- return a validation error
- do not run a full-image edit fallback

That restriction is deliberate because the endpoint is supposed to be safe and narrow.

## Repo Structure

This is the repo structure we should build.

```text
qwen-v19-clothing-runpod-comfyui/
  Dockerfile
  custom_start.sh
  bootstrap_models.py
  handler.py
  instruction_parser.py
  workflow_builder.py
  requirements.txt
  .env.example
  README.md
  docs/
    architecture.md
  workflow_templates/
    clothing_edit_single_pass_atr.json
    clothing_edit_single_pass_lip.json
  scripts/
    export_workflow_notes.md
  test_input.json
```

## File Responsibilities

### `Dockerfile`

- start from `runpod/worker-comfyui:<version>-base`
- install custom node dependencies
- copy handler and helper scripts
- copy workflow templates
- launch `custom_start.sh`

### `custom_start.sh`

- start ComfyUI
- run model bootstrap
- wait for ComfyUI readiness
- start the RunPod handler

### `bootstrap_models.py`

- download `Qwen-Rapid-AIO-NSFW-v19.safetensors` if missing
- install or copy the fixed `nodes_qwen` override into the correct ComfyUI path
- download human parser weights if we decide to bundle them during startup instead of baking them into the image

### `handler.py`

- accept `image_url` or `image_base64`
- validate input
- parse clothing instruction into one or more passes
- upload the input image to ComfyUI
- build one workflow per pass
- execute passes sequentially
- pass each output image into the next pass when needed
- collect final image and return base64 or object storage URL

### `instruction_parser.py`

- extract garment targets
- map natural language to parser categories
- split multi-edit requests into sequential passes
- reject unsupported instructions

### `workflow_builder.py`

- load JSON template
- patch node inputs by node id
- inject prompt, category, seed, and filenames
- support ATR and LIP templates

### `workflow_templates/*.json`

- frozen ComfyUI exports
- no business logic
- only stable graph definitions

### `test_input.json`

- a minimal local or RunPod request sample

## API Contract

### Input

```json
{
  "input": {
    "image_url": "https://example.com/look.png",
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

### Output

```json
{
  "output": {
    "images": [
      {
        "filename": "qwen-v19-clothing/job-123/final.png",
        "type": "base64_or_bucket_url",
        "data": "..."
      }
    ],
    "passes": [
      {
        "category": "Dress"
      },
      {
        "category": "Socks"
      }
    ]
  }
}
```

## Defaults

- checkpoint: `Qwen-Rapid-AIO-NSFW-v19.safetensors`
- steps: `6`
- cfg: `1.0`
- sampler: `euler_ancestral`
- scheduler: `beta`
- parser model: `atr`
- mask grow: `12 px`
- mask blur: `6 px`
- crop context: `128 px`

## Known Risks

### Human parser misses unusual garments

Some fashion items may be misclassified or merged. That is why `LIP` should remain available as a fallback parser, because it includes `Socks` while `ATR` does not.

### Category swaps can need two passes

Changing a dress into a skirt may expose legs that were previously covered. This might work in one pass, but we should expect more failures than pure recolors.

### Over-broad masks can still alter nearby limbs

That is why mask refinement and category-specific negative prompts matter.

## Recommended Build Order

1. Build the ComfyUI graph manually with one hardcoded garment category and validate quality.
2. Export the stable single-pass workflow JSON.
3. Implement `workflow_builder.py` around the frozen node ids.
4. Implement `instruction_parser.py`.
5. Implement the RunPod `handler.py`.
6. Add model bootstrap and Docker image build.
7. Test locally before GitHub deployment to RunPod.
