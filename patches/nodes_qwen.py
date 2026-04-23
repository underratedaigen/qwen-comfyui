import math

import comfy.utils

try:
    import comfy.node_helpers as node_helpers
except ModuleNotFoundError:
    import node_helpers


def _encode_reference_image(image, *, scale_total_pixels: int, upscale_method: str, crop_mode: str):
    samples = image.movedim(-1, 1)
    scale_by = math.sqrt(scale_total_pixels / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    resized = comfy.utils.common_upscale(samples, width, height, upscale_method, crop_mode)
    return resized.movedim(1, -1), samples


class TextEncodeQwenImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, image=None):
        ref_latent = None
        images = []

        if image is not None:
            resized_image, _ = _encode_reference_image(
                image,
                scale_total_pixels=int(1024 * 1024),
                upscale_method="area",
                crop_mode="disabled",
            )
            images = [resized_image]
            if vae is not None:
                ref_latent = vae.encode(resized_image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": [ref_latent]},
                append=True,
            )
        return (conditioning,)


class TextEncodeQwenImageEditPlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "target_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None, image4=None, target_latent=None):
        ref_latents = []
        images_vl = []
        llama_template = (
            "<|im_start|>system\n"
            "Describe key details of the input image (including any objects, characters, poses, facial features, clothing, "
            "setting, textures and style), then explain how the user's text instruction should alter, modify or recreate the "
            "image. Generate a new image that meets the user's requirements, which can vary from a small change to a completely "
            "new image using inputs as a guide.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        image_prompt_parts = []

        for index, image in enumerate((image1, image2, image3, image4), start=1):
            if image is None:
                continue

            resized_image, original_samples = _encode_reference_image(
                image,
                scale_total_pixels=int(384 * 384),
                upscale_method="lanczos",
                crop_mode="center",
            )
            images_vl.append(resized_image)

            if vae is not None:
                latent_samples = original_samples
                if target_latent is not None:
                    twidth = target_latent["samples"].shape[-1] * 8
                    theight = target_latent["samples"].shape[-2] * 8
                    latent_samples = comfy.utils.common_upscale(original_samples, twidth, theight, "lanczos", "center")
                ref_latents.append(vae.encode(latent_samples.movedim(1, -1)[:, :, :, :3]))

            image_prompt_parts.append(f"Picture {index}: <|vision_start|><|image_pad|><|vision_end|>")

        tokens = clip.tokenize("".join(image_prompt_parts) + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latents:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": ref_latents},
                append=True,
            )
        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEdit": TextEncodeQwenImageEdit,
    "TextEncodeQwenImageEditPlus": TextEncodeQwenImageEditPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEdit": "Text Encode Qwen Image Edit",
    "TextEncodeQwenImageEditPlus": "Text Encode Qwen Image Edit Plus",
}
