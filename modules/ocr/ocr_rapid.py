import logging
from typing import List
import numpy as np

from .base import register_OCR, OCRBase, TextBlock

# 로거 설정
logger = logging.getLogger(__name__)

@register_OCR("rapid_ocr")
class RapidOCREngine(OCRBase):
    """
    RapidOCR 라이브러리를 사용하는 OCR 엔진입니다.
    가볍고 빠른 ONNX 런타임을 기반으로 동작하며, 다양한 언어를 지원합니다.
    """

    # 사용자가 설정할 수 있는 파라미터를 정의합니다.
    # ocr_llm_api.py의 형식을 따릅니다.
    params = {
        "language": {
            "type": "selector",
            "options": ["English", "Japanese", "Korean", "Chinese", "French", "Spanish", "German", "Russian"],
            "value": "Japanese",
            "description": "OCR을 수행할 언어를 선택합니다.",
        },
        "device": {
            "type": "selector",
            "options": ["CPU", "GPU"],
            "value": "CPU",
            "description": "OCR 연산을 수행할 장치를 선택합니다. GPU 선택 시 CUDA가 필요합니다.",
        },
        "description": "RapidOCR 엔진을 사용하여 이미지에서 텍스트를 추출합니다."
    }

    # 내부적으로 사용할 언어 코드 매핑
    lang_map = {
        'English': 'en',
        'French': 'fr',
        'Spanish': 'es',
        'German': 'de',
        'Japanese': 'ja',
        'Chinese': 'ch',
        'Russian': 'ru',
        'Korean': 'ko',
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.engine = None
        # BaseModule이 모델을 관리할 수 있도록 모델이 저장될 속성 이름을 지정합니다.
        self._load_model_keys = {'engine'}

    # --- 파라미터 접근을 위한 속성들 ---
    @property
    def language(self) -> str:
        lang_name = self.get_param_value("language")
        return self.lang_map.get(lang_name, 'en') # 기본값으로 영어 설정

    @property
    def device(self) -> str:
        return self.get_param_value("device")

    def _load_model(self):
        """
        BaseModule의 load_model()에 의해 호출되어 실제 RapidOCR 엔진을 로드하고 초기화합니다.
        """
        if self.engine is not None:
            self.logger.info("RapidOCR engine is already loaded.")
            return

        self.logger.info(f"Loading RapidOCR engine for language: '{self.language}' on device: '{self.device}'")

        try:
            # 필요한 라이브러리를 이 시점에서 임포트하여 초기 로딩 속도를 높입니다.
            from rapidocr import RapidOCR
        except ImportError:
            self.logger.error("RapidOCR is not installed. Please install it using: pip install rapidocr-onnxruntime")
            # GPU를 사용한다면: pip install rapidocr-onnxruntime-gpu
            raise ImportError("RapidOCR library not found.")

        use_gpu = self.device == 'GPU'
        
        # RapidOCR 엔진을 생성합니다.
        # lang 인자는 ['ch', 'en', 'korean', 'japan', ...] 형태의 리스트를 받습니다.
        # 여기서는 단일 언어만 사용하도록 간소화합니다.
        try:
            self.engine = RapidOCR(lang_list=[self.language], use_gpu=use_gpu)
            self.logger.info("RapidOCR engine loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize RapidOCR engine: {e}")
            self.engine = None


    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> List[TextBlock]:
        """
        각 TextBlock 영역을 개별적으로 잘라 OCR을 수행하고, 결과를 TextBlock에 직접 할당합니다.
        프로젝트의 기존 OCR 처리 방식과 일관성을 유지합니다.
        """
        # 모델이 로드되지 않았다면 로드를 시도합니다.
        if not self.all_model_loaded():
            self.load_model()

        if self.engine is None:
            self.logger.warning("RapidOCR engine is not initialized. Skipping OCR process.")
            # 엔진이 없으면 모든 블록의 텍스트를 비웁니다.
            for blk in blk_list:
                blk.text = ""
            return blk_list

        im_h, im_w = img.shape[:2]

        # 각 텍스트 블록을 순회하며 OCR을 수행합니다.
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy

            # 블록 좌표가 이미지 범위 내에 있는지 확인합니다.
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                # 해당 블록의 이미지만 잘라냅니다.
                cropped_img = img[y1:y2, x1:x2]
                
                try:
                    # 잘라낸 이미지에 대해 OCR을 실행합니다.
                    result, _ = self.engine(cropped_img)

                    if result:
                        # OCR 결과에서 텍스트만 추출하여 리스트로 만듭니다.
                        # result는 [(box, text, score), ...] 형태의 리스트입니다.
                        texts = [text for _, text, _ in result]
                        # 추출된 텍스트들을 공백으로 이어 붙여 한 줄로 만듭니다.
                        final_text = " ".join(texts)
                    else:
                        # 텍스트가 인식되지 않은 경우
                        final_text = ""
                    
                    blk.text = final_text

                    if self.debug_mode:
                        self.logger.debug(f"Processed block ({x1},{y1},{x2},{y2}), Text: '{final_text}'")

                except Exception as e:
                    if self.debug_mode:
                        self.logger.error(f"Error recognizing block at ({x1},{y1},{x2},{y2}): {e}")
                    blk.text = ""
            else:
                if self.debug_mode:
                    self.logger.warning(f"Invalid text block coordinates {blk.xyxy} for image size {im_w}x{im_h}")
                blk.text = ""
        
        return blk_list

    def updateParam(self, param_key: str, param_content):
        """
        파라미터가 변경될 때 호출됩니다.
        중요한 파라미터(언어, 장치)가 바뀌면 기존 모델을 언로드하여 다음 호출 시 새로 로드되도록 합니다.
        """
        super().updateParam(param_key, param_content)
        if param_key in ["language", "device"]:
            self.logger.info(f"Parameter '{param_key}' changed. Unloading model to re-initialize on next use.")
            self.unload_model(empty_cache=True)