import os
import traceback

import numpy as np
from typing import List, Optional
from .common import CommonOCR, OfflineOCR
from .model_32px import Model32pxOCR
from .model_48px import Model48pxOCR
from .model_48px_ctc import Model48pxCTCOCR
from .model_manga_ocr import ModelMangaOCR
from ..config import Ocr, OcrConfig
from ..utils import Quadrilateral, get_logger

OCRS = {
    Ocr.ocr32px: Model32pxOCR,
    Ocr.ocr48px: Model48pxOCR,
    Ocr.ocr48px_ctc: Model48pxCTCOCR,
    Ocr.mocr: ModelMangaOCR,
}
ocr_cache = {}

logger = get_logger('ocr')

# ── OCR đọc vớt (fallback) ───────────────────────────────────────────────────
# 48px đọc chuẩn nhưng LOẠI HẲN vùng có prob dưới ngưỡng → bong bóng khó bị bỏ
# sót, không bao giờ được dịch. 48px_ctc bắt được hết nhưng hay đọc sai. Khi env
# MIT_OCR_FALLBACK đặt tên một model khác (backend đặt "48px_ctc" khi model
# chính là 48px), dispatch() chạy 2 lượt: model chính trước, rồi CHỈ những vùng
# còn text rỗng mới đưa cho model fallback đọc vớt. Vùng model chính đọc được
# giữ nguyên (giữ độ chính xác), vùng sót có fallback đỡ (không mất chữ).
# MIT_OCR_FALLBACK_PROB (tuỳ chọn) siết ngưỡng prob riêng cho lượt vớt.
# Không set env → hành vi gốc, không đổi.

def _fallback_key(primary: Ocr) -> Optional[Ocr]:
    """Đọc MIT_OCR_FALLBACK; trả về key fallback hợp lệ khác model chính, hoặc None."""
    name = os.environ.get('MIT_OCR_FALLBACK', '').strip()
    if not name:
        return None
    try:
        key = Ocr(name)
    except ValueError:
        logger.warning(f'[OCR-FALLBACK] MIT_OCR_FALLBACK="{name}" không hợp lệ, bỏ qua')
        return None
    return key if key != primary else None

def get_ocr(key: Ocr, *args, **kwargs) -> CommonOCR:
    if key not in OCRS:
        raise ValueError(f'Could not find OCR for: "{key}". Choose from the following: %s' % ','.join(OCRS))
    if not ocr_cache.get(key):
        ocr = OCRS[key]
        ocr_cache[key] = ocr(*args, **kwargs)
    return ocr_cache[key]

async def prepare(ocr_key: Ocr, device: str = 'cpu'):
    ocr = get_ocr(ocr_key)
    if isinstance(ocr, OfflineOCR):
        await ocr.download()
        await ocr.load(device)
    # Chuẩn bị sẵn model đọc vớt (nếu bật) — download/load chỉ ở đây, dispatch không tải
    fb_key = _fallback_key(ocr_key)
    if fb_key:
        fb_ocr = get_ocr(fb_key)
        if isinstance(fb_ocr, OfflineOCR):
            await fb_ocr.download()
            await fb_ocr.load(device)

async def dispatch(ocr_key: Ocr, image: np.ndarray, regions: List[Quadrilateral], config:Optional[OcrConfig] = None, device: str = 'cpu', verbose: bool = False) -> List[Quadrilateral]:
    ocr = get_ocr(ocr_key)
    if isinstance(ocr, OfflineOCR):
        await ocr.load(device)
    config = config or OcrConfig()
    result = await ocr.recognize(image, regions, config, verbose)

    # Lượt 2: đọc vớt vùng bị model chính bỏ sót. Chỉ áp cho nhánh Quadrilateral
    # (detector luôn tạo text='' và _infer set .text lên chính object khi vượt
    # ngưỡng → vùng sót = vùng text còn rỗng); nhánh TextBlock trả textlines
    # nguyên vẹn nên không xác định được vùng sót.
    fb_key = _fallback_key(ocr_key)
    if fb_key and regions and isinstance(regions[0], Quadrilateral):
        try:
            missed = [r for r in regions if not (r.text or '').strip()]
            if missed:
                fb_ocr = get_ocr(fb_key)
                if isinstance(fb_ocr, OfflineOCR):
                    await fb_ocr.load(device)  # đã download ở prepare(); load có guard _loaded
                fb_config = config
                fb_prob = os.environ.get('MIT_OCR_FALLBACK_PROB', '').strip()
                if fb_prob:
                    try:
                        fb_config = config.model_copy(update={'prob': float(fb_prob)})
                    except AttributeError:  # pydantic v1
                        fb_config = config.copy(update={'prob': float(fb_prob)})
                await fb_ocr.recognize(image, missed, fb_config, verbose)
                rescued = sum(1 for r in missed if (r.text or '').strip())
                logger.info(f'[OCR-FALLBACK] {fb_key.value} vớt được {rescued}/{len(missed)} vùng bị {ocr_key.value} bỏ sót')
                # Khôi phục ĐÚNG THỨ TỰ vùng gốc: lọc lại từ regions, giữ vùng đã có chữ
                result = [r for r in regions if (r.text or '').strip()]
        except Exception:
            logger.warning(f'[OCR-FALLBACK] lỗi lượt đọc vớt, giữ kết quả {ocr_key.value}:\n{traceback.format_exc()}')
    return result

async def unload(ocr_key: Ocr):
    ocr_cache.pop(ocr_key, None)
