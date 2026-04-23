# Vendored Dependencies

This repo vendors the runtime files for the ComfyUI custom nodes it depends on so the Docker build does not need to clone external Git repositories.

## Human Parser

- Source: `https://github.com/cozymantis/human-parser-comfyui-node`
- Commit: `0ce414f7c939d36312f44bc2209f12f32fd663a8`
- Vendored path: `custom_nodes/human-parser-comfyui-node`
- Included: runtime Python files, `schp/`, license, and readme
- Excluded: `.git/`, `.github/`, and demo image assets

## Inpaint Crop And Stitch

- Source: `https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch`
- Commit: `8e59ab12d6709616528279c85c0648ca8441684d`
- Vendored path: `custom_nodes/ComfyUI-Inpaint-CropAndStitch`
- Included: runtime Python files, `js/`, license, and readme
- Excluded: `.git/`, `.github/`, example workflows, tests, and sample images

## Notes

The vendored files are intended to keep the image build deterministic and closer to the simple WAN worker layout already used locally.
