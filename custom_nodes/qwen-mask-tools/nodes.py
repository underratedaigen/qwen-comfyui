from __future__ import annotations

import numpy as np
from scipy import ndimage
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


def _erode(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    kernel_size = pixels * 2 + 1
    return -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pixels)


def _close(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    return _erode(_grow(mask, pixels), pixels)


def _fill_holes(mask: torch.Tensor) -> torch.Tensor:
    device = mask.device
    mask_np = mask.detach().cpu().numpy() > 0.5
    filled = np.empty_like(mask_np, dtype=np.float32)
    for batch_index in range(mask_np.shape[0]):
        for channel_index in range(mask_np.shape[1]):
            filled[batch_index, channel_index] = ndimage.binary_fill_holes(
                mask_np[batch_index, channel_index]
            ).astype(np.float32)
    return torch.from_numpy(filled).to(device=device, dtype=mask.dtype)


def _fill_short_horizontal_gaps(mask: torch.Tensor, max_gap_pixels: int) -> torch.Tensor:
    if max_gap_pixels <= 0:
        return mask

    device = mask.device
    mask_np = mask.detach().cpu().numpy() > 0.5
    filled = mask_np.copy()
    for batch_index in range(mask_np.shape[0]):
        for channel_index in range(mask_np.shape[1]):
            rows = filled[batch_index, channel_index]
            for y_index in range(rows.shape[0]):
                foreground = np.flatnonzero(rows[y_index])
                if foreground.size < 2:
                    continue
                starts = [foreground[0]]
                ends = []
                for idx in range(1, foreground.size):
                    if foreground[idx] != foreground[idx - 1] + 1:
                        ends.append(foreground[idx - 1])
                        starts.append(foreground[idx])
                ends.append(foreground[-1])

                for left_end, right_start in zip(ends[:-1], starts[1:]):
                    gap = int(right_start - left_end - 1)
                    if 0 < gap <= max_gap_pixels:
                        rows[y_index, left_end : right_start + 1] = True
    return torch.from_numpy(filled.astype(np.float32)).to(device=device, dtype=mask.dtype)


class QwenBodyMaskFromBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_mask": ("MASK",),
                "head_mask": ("MASK",),
                "head_protect_grow_pixels": ("INT", {"default": 4, "min": 0, "max": 256, "step": 1}),
                "body_grow_pixels": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "body_close_pixels": ("INT", {"default": 24, "min": 0, "max": 256, "step": 1}),
                "row_fill_max_gap_pixels": ("INT", {"default": 96, "min": 0, "max": 1024, "step": 1}),
                "fill_holes": ("BOOLEAN", {"default": True}),
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
        body_close_pixels: int,
        row_fill_max_gap_pixels: int,
        fill_holes: bool,
        threshold: float,
    ):
        background, original_dims = _as_bchw(background_mask)
        head, _ = _as_bchw(head_mask)

        background = (background >= threshold).float()
        head = (head >= threshold).float()

        person = 1.0 - background
        person = _close(person, int(body_close_pixels))
        person = _fill_short_horizontal_gaps(person, int(row_fill_max_gap_pixels))
        if fill_holes:
            person = _fill_holes(person)
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
