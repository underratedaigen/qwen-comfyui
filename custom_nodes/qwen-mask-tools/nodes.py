from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_bchw(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    original_dims = mask.dim()
    if original_dims == 2:
        return mask.unsqueeze(0).unsqueeze(0).float(), original_dims
    if original_dims == 3:
        return mask.unsqueeze(1).float(), original_dims
    if original_dims == 4:
        return mask.float(), original_dims
    raise ValueError(f"Unsupported mask dimensions: {original_dims}")


def _from_bchw(mask: torch.Tensor, original_dims: int) -> torch.Tensor:
    if original_dims == 2:
        return mask.squeeze(0).squeeze(0)
    if original_dims == 3:
        return mask.squeeze(1)
    return mask


def _grow(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    kernel_size = pixels * 2 + 1
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pixels)


class QwenBodyMaskFromBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_mask": ("MASK",),
                "head_mask": ("MASK",),
                "head_protect_grow_pixels": ("INT", {"default": 4, "min": 0, "max": 256, "step": 1}),
                "body_grow_pixels": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "build"
    CATEGORY = "Qwen/Image Edit"

    def build(
        self,
        background_mask: torch.Tensor,
        head_mask: torch.Tensor,
        head_protect_grow_pixels: int,
        body_grow_pixels: int,
        threshold: float,
    ):
        background, original_dims = _as_bchw(background_mask)
        head, _ = _as_bchw(head_mask)

        background = (background >= threshold).float()
        head = (head >= threshold).float()

        person = 1.0 - background
        person = _grow(person, int(body_grow_pixels))
        protected_head = _grow(head, int(head_protect_grow_pixels))
        body = (person - protected_head).clamp(0.0, 1.0)

        return (_from_bchw(body, original_dims),)


NODE_CLASS_MAPPINGS = {
    "Qwen Body Mask From Background": QwenBodyMaskFromBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen Body Mask From Background": "Qwen Body Mask From Background",
}
