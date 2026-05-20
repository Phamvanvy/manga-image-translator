"""
Patch: manga_translator/inpainting/inpainting_sd_cn.py
------------------------------------------------------
Thay đổi so với bản gốc:
  1. BASE_MODEL_ID đọc từ env var MIT_SD_CHECKPOINT (nếu set) → cho phép dùng
     checkpoint monochrome/lineart mà không cần sửa source.
  2. Prompt được tăng cường thêm 'grayscale, black and white' để model hiểu
     context manga B&W.
  3. Low-VRAM path: nếu VRAM < 6 GB → tự động bật enable_sequential_cpu_offload
     thay vì enable_model_cpu_offload (tiết kiệm thêm ~1 GB).

Checkpoint khuyến nghị (thay thế MeinaMix):
  - cagliostrolab/animagine-xl-3.1  (SDXL — không tương thích SD1.5, không dùng)
  - hakurei/waifu-diffusion          (SD1.5, anime nhưng có thể dùng)
  - Lykon/DreamShaper                (SD1.5, flexible)
  *** KHUYẾN NGHỊ THỰC TẾ ***:
  - "stablediffusionapi/manga-diffusion"    ← SD1.5, train thuần manga B&W
  - "Linaqruf/anything-v3-1"               ← SD1.5 anime, dùng với prompt mono
  - Local path: "D:/models/manga_mono.safetensors" (sau khi tải từ Civitai)

Để dùng:
  Trước khi chạy MIT, set env:
    MIT_SD_CHECKPOINT=stablediffusionapi/manga-diffusion
  Hoặc trong run_hybrid.py, truyền qua sub_env["MIT_SD_CHECKPOINT"].
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .common import OfflineInpainter
from ..config import InpainterConfig
from ..utils import resize_keep_aspect


# ---------------------------------------------------------------------------
# Helpers (giữ nguyên từ bản gốc)
# ---------------------------------------------------------------------------

def _np_to_pil(arr: np.ndarray) -> Image.Image:
    """RGB uint8 numpy → PIL."""
    return Image.fromarray(arr.astype(np.uint8))


def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def _make_inpaint_condition(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Build the inpaint ControlNet conditioning image:
    masked-out pixels → -1 (grey after rescale)  [lllyasviel spec].
    """
    image_np = np.array(image).astype(np.float32) / 255.0
    mask_np  = np.array(mask.convert('L')).astype(np.float32) / 255.0
    mask_np  = (mask_np > 0.5).astype(np.float32)
    image_np[mask_np > 0.5] = -1.0
    image_np = np.clip(image_np, -1, 1)
    image_np = (image_np + 1.0) / 2.0
    return Image.fromarray((image_np * 255).astype(np.uint8))


def _make_lineart_condition(image: Image.Image) -> Image.Image:
    """
    Canny-based lineart map cho ControlNet-lineart conditioning.
    Đảo ngược: nền đen → nền trắng (ControlNet lineart chuẩn).
    """
    gray  = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_not(edges)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def _pad_to_multiple(img: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, int, int]:
    h, w     = img.shape[:2]
    new_h    = ((h + multiple - 1) // multiple) * multiple
    new_w    = ((w + multiple - 1) // multiple) * multiple
    padded   = cv2.copyMakeBorder(img, 0, new_h - h, 0, new_w - w, cv2.BORDER_REFLECT_101)
    return padded, h, w


def _get_free_vram_gb() -> float:
    """Trả về VRAM trống (GB) của GPU đầu tiên; 99 nếu không có GPU."""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            used  = torch.cuda.memory_allocated(0)
            free  = props.total_memory - used
            return free / (1024 ** 3)
    except Exception:
        pass
    return 99.0


# ---------------------------------------------------------------------------
# Inpainter
# ---------------------------------------------------------------------------

class MangaSDControlNetInpainter(OfflineInpainter):
    """
    Stable Diffusion 1.5 + dual ControlNet inpainter.
    Registered as Inpainter.manga_sd_cn.

    Checkpoint được đọc từ env MIT_SD_CHECKPOINT nếu set, nếu không dùng
    mặc định 'Meina/MeinaMix_V11'. Để dùng checkpoint monochrome:
        export MIT_SD_CHECKPOINT=stablediffusionapi/manga-diffusion
    """

    _MODEL_MAPPING = {}

    # Checkpoint mặc định — bị override bởi env MIT_SD_CHECKPOINT
    # MeinaMix là model gốc, hoạt động. Prompts mới + strength<1 giúp giữ nền trắng.
    BASE_MODEL_ID   = os.environ.get('MIT_SD_CHECKPOINT', 'Meina/MeinaMix_V11')
    CN_INPAINT_ID   = 'lllyasviel/control_v11p_sd15_inpaint'
    CN_LINEART_ID   = 'lllyasviel/control_v11p_sd15_lineart'
    VAE_ID          = 'stabilityai/sd-vae-ft-mse'

    NUM_INFERENCE_STEPS = 20
    GUIDANCE_SCALE      = 7.5
    CN_INPAINT_SCALE    = 1.0
    CN_LINEART_SCALE    = 0.65
    MAX_INPAINT_SIZE    = 768

    # Prompt được tăng cường cho manga B&W — phòng nền đen / screentone
    POSITIVE_PROMPT = (
        'masterpiece, best quality, manga, monochrome, grayscale, '
        'black and white, lineart, plain white background, clean white, '
        'white paper, empty interior, clean speech bubble, no text'
    )
    NEGATIVE_PROMPT = (
        'color, colorful, text, speech bubble, word, letter, watermark, '
        'lowres, bad anatomy, worst quality, blurry, jpeg artifacts, '
        'extra lines, overexposed, purple tint, blue tint, '
        'dark background, black background, grey background, dark fill, '
        'screentone, halftone pattern, crosshatch, dot pattern, noise'
    )
    # strength=1.0: SD có toàn quyền xóa text; post-process sẽ đảm bảo nền trắng
    INPAINT_STRENGTH = 1.0

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Load / Unload
    # ------------------------------------------------------------------

    async def _load(self, device: str):
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetInpaintPipeline,
            AutoencoderKL,
        )

        # Đọc lại env tại thời điểm load (hỗ trợ override muộn)
        checkpoint = os.environ.get('MIT_SD_CHECKPOINT', self.BASE_MODEL_ID)

        self.device = device
        dtype = torch.float16 if device.startswith('cuda') else torch.float32

        self.logger.info(f'[SD-CN] Checkpoint: {checkpoint}')
        self.logger.info('Loading ControlNet models …')

        cn_inpaint = ControlNetModel.from_pretrained(self.CN_INPAINT_ID, torch_dtype=dtype)
        cn_lineart = ControlNetModel.from_pretrained(self.CN_LINEART_ID, torch_dtype=dtype)

        self.logger.info(f'Loading base checkpoint: {checkpoint} …')
        vae = AutoencoderKL.from_pretrained(self.VAE_ID, torch_dtype=dtype)

        # Hỗ trợ cả HuggingFace ID lẫn path local (.safetensors / thư mục)
        # Fallback về MeinaMix nếu checkpoint không tồn tại
        import os as _os
        _FALLBACK = 'Meina/MeinaMix_V11'
        if _os.path.exists(checkpoint) and checkpoint.endswith('.safetensors'):
            from diffusers import StableDiffusionControlNetInpaintPipeline as _Pipe
            self.pipe = _Pipe.from_single_file(
                checkpoint,
                controlnet=[cn_inpaint, cn_lineart],
                vae=vae,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            try:
                self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    checkpoint,
                    controlnet=[cn_inpaint, cn_lineart],
                    vae=vae,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            except Exception as _e:
                if checkpoint != _FALLBACK:
                    self.logger.warning(
                        f'[SD-CN] Checkpoint "{checkpoint}" failed ({type(_e).__name__}: {_e}). '
                        f'Falling back to {_FALLBACK}.'
                    )
                    self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                        _FALLBACK,
                        controlnet=[cn_inpaint, cn_lineart],
                        vae=vae,
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                    )
                else:
                    raise

        # xformers (tuỳ chọn, giảm VRAM ~20%)
        # Kiểm tra thực tế CUDA ops trước khi bật, tránh overhead khi GPU chưa hỗ trợ
        _xformers_ok = False
        try:
            import xformers.ops as _xops
            _tq = torch.randn(1, 1, 64, 64, device=device, dtype=dtype)
            _xops.memory_efficient_attention(_tq, _tq, _tq)
            self.pipe.enable_xformers_memory_efficient_attention()
            self.logger.info('xformers enabled')
            _xformers_ok = True
        except Exception:
            self.logger.warning('xformers không khả dụng — dùng attention_slicing')
        if not _xformers_ok:
            try:
                self.pipe.enable_attention_slicing(1)
                self.logger.info('attention_slicing enabled')
            except Exception:
                pass

        self.pipe.enable_vae_slicing()

        # Phát hiện VRAM và chọn offload mode
        free_vram = _get_free_vram_gb()
        self.logger.info(f'[SD-CN] Free VRAM: {free_vram:.1f} GB')

        try:
            self.pipe.to(device)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                if free_vram < 6.0:
                    # VRAM < 6GB: sequential offload (chậm nhưng an toàn nhất)
                    self.logger.warning('VRAM < 6GB → enable_sequential_cpu_offload')
                    self.pipe.enable_sequential_cpu_offload()
                else:
                    # VRAM 6-8GB: model offload (cân bằng tốc độ/bộ nhớ)
                    self.logger.warning('VRAM đầy → enable_model_cpu_offload')
                    self.pipe.enable_model_cpu_offload()
            else:
                raise

        self.pipe.set_progress_bar_config(disable=True)
        self.logger.info('MangaSDControlNetInpainter ready')

    async def _unload(self):
        del self.pipe

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    async def _infer(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        config: InpainterConfig,
        inpainting_size: int = 1024,
        verbose: bool = False,
    ) -> np.ndarray:

        img_original = image.copy()
        h_orig, w_orig = image.shape[:2]

        # Giới hạn kích thước để tránh OOM
        work_size = min(inpainting_size, self.MAX_INPAINT_SIZE)

        if max(h_orig, w_orig) > work_size:
            image = resize_keep_aspect(image, work_size)
            mask  = resize_keep_aspect(mask, work_size)

        image_padded, h_work, w_work = _pad_to_multiple(image, 8)
        mask_padded,  _,      _      = _pad_to_multiple(mask, 8)

        pil_image = _np_to_pil(image_padded)
        pil_mask  = Image.fromarray(mask_padded).convert('L')

        cn_inpaint_image = _make_inpaint_condition(pil_image, pil_mask)
        cn_lineart_image = _make_lineart_condition(pil_image)

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pipe(
                prompt=self.POSITIVE_PROMPT,
                negative_prompt=self.NEGATIVE_PROMPT,
                image=pil_image,
                mask_image=pil_mask,
                control_image=[cn_inpaint_image, cn_lineart_image],
                controlnet_conditioning_scale=[self.CN_INPAINT_SCALE, self.CN_LINEART_SCALE],
                num_inference_steps=self.NUM_INFERENCE_STEPS,
                guidance_scale=self.GUIDANCE_SCALE,
                strength=self.INPAINT_STRENGTH,   # <1.0 → giữ background gốc
                height=image_padded.shape[0],
                width=image_padded.shape[1],
                generator=torch.Generator(device=self.device).manual_seed(42),
            )
        )

        inpainted_np = _pil_to_np(result.images[0])
        inpainted_np = inpainted_np[:h_work, :w_work]
        inpainted_np = cv2.resize(inpainted_np, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)

        # Composite: chỉ thay thế vùng masked, giữ nguyên phần còn lại
        mask_original = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        binary_mask   = (mask_original > 127).astype(np.float32)[:, :, None]
        output = (
            img_original * (1.0 - binary_mask) +
            inpainted_np * binary_mask
        ).astype(np.uint8)

        # Post-process 1: Khôi phục pixel trắng gốc bị SD làm tối.
        # Nền bong bóng thoại vốn trắng (≥180 trong ảnh gốc) → cưỡng chế trắng lại,
        # bất kể SD tạo ra gì trong vùng masked. Pixel tối gốc (line art) giữ nguyên.
        if np.any(binary_mask > 0.5):
            orig_gray   = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY).astype(np.float32)
            was_bright  = (orig_gray >= 180).astype(np.float32)[:, :, None]
            restore     = was_bright * binary_mask            # chỉ trong vùng mask
            output_f    = output.astype(np.float32)
            # Đè trắng với cường độ 95% (giữ lại 5% để tránh hard-clip hoàn toàn)
            output_f    = output_f * (1.0 - restore * 0.95) + 255.0 * (restore * 0.95)
            output      = np.clip(output_f, 0, 255).astype(np.uint8)

        # Post-process 2: Bleach mid-tone (khử screentone SD còn sót trong vùng không phải gốc-trắng).
        # Ngưỡng tối từ 60 → 40 để hung hăng hơn.
        if np.any(binary_mask > 0.5):
            gray_out  = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY).astype(np.float32)
            bleach    = np.clip((gray_out - 40.0) / 140.0, 0.0, 1.0)
            bleach_3ch = (bleach[:, :, None] * binary_mask).astype(np.float32)
            output_f  = output.astype(np.float32)
            output_f  = output_f + bleach_3ch * (255.0 - output_f) * 0.90
            output    = np.clip(output_f, 0, 255).astype(np.uint8)

        return output
