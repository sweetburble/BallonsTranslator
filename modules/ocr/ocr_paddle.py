import numpy as np
from typing import List
import os
import logging

LOGGER = logging.getLogger("BallonTranslator")

try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    LOGGER.debug(
        "PaddleOCR is not installed, so the module will not be initialized. \nCheck this issue https://github.com/dmMaze/BallonsTranslator/issues/835#issuecomment-2772940806"
    )

import re
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

# Specify the path for storing PaddleOCR models
PADDLE_OCR_PATH = os.path.join("data", "models", "paddle-ocr")
# Set an environment variable to store PaddleOCR models
os.environ["PPOCR_HOME"] = PADDLE_OCR_PATH
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if PADDLE_OCR_AVAILABLE:
    @register_OCR("paddle_ocr")
    class PaddleOCRModule(OCRBase):
        # Mapping language names to PaddleOCR codes
        lang_map = {
            'English': 'en',
            'French': 'fr',
            'Spanish': 'es',
            'German': 'de',
            'Japanese': 'japan',
            'Chinese': 'ch',
            'Russian': 'ru',
            'Korean': 'korean',
        }

        params = {
            "language": {
                "type": "selector",
                "options": list(lang_map.keys()),
                "value": "Japanese",
                "description": "Select the language for OCR",
            },
            "text_rec_score_thresh": {
                "value": 0.3,
                "description": "Confidence threshold for text recognition",
            },
            "text_case": {
                "type": "selector",
                "options": ["Uppercase", "Capitalize Sentences", "Lowercase"],
                "value": "Capitalize Sentences",
                "description": "Text case transformation",
            },
            "output_format": {
                "type": "selector",
                "options": ["Single Line", "As Recognized"],
                "value": "As Recognized",
                "description": "Text output format",
            },
        }

        device = DEFAULT_DEVICE

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.language = self.params["language"]["value"]
            self.text_rec_score_thresh = self.params["text_rec_score_thresh"]["value"]
            self.text_case = self.params["text_case"]["value"]
            self.output_format = self.params["output_format"]["value"]
            self._setup_logging()
            self._load_model()

        def _setup_logging(self):
            # 로그 레벨 조정
            log_level = logging.DEBUG if self.debug_mode else logging.WARNING
            logging.getLogger("ppocr").setLevel(log_level)
            logging.getLogger("paddleocr").setLevel(log_level)

        def _load_model(self):
            lang_code = self.lang_map[self.language]
            
            if self.debug_mode:
                self.logger.info(
                    f"Loading PaddleOCR ({self.ocr_version}) for language: {self.language} ({lang_code})"
                )

            # PaddleOCR 초기화
            self.model = PaddleOCR(
                lang=lang_code,
                text_detection_model_name='PP-OCRv5_mobile_det',
                text_recognition_model_name='PP-OCRv5_mobile_rec',
                use_doc_orientation_classify=False,  
                use_doc_unwarping=False,  
                use_textline_orientation=False  
            )

        def ocr_img(self, img: np.ndarray) -> str:
            self.logger.debug(f'ocr_img: {img.shape}')

            result_list = self.model.predict(img)

            if result_list:
                result = result_list[0]
                # json 속성에서 rec_texts 필드 가져오기
                rec_texts = result.json.get('rec_texts', [])
                
                # 출력 형식에 따라 텍스트 조합
                if self.output_format == "Single Line":
                    # 모든 텍스트를 공백으로 연결
                    combined_text = ' '.join(rec_texts)
                else:  # "As Recognized"
                    # 줄바꿈으로 연결
                    combined_text = '\n'.join(rec_texts)
                
                # 텍스트 케이스 변환 적용
                final_text = self._apply_text_case(combined_text)
                # 문장부호 및 공백 정리
                final_text = self._apply_punctuation_and_spacing(final_text)
                
                return final_text
            
            return ''

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
            """블록 리스트에 대해 OCR 수행"""
            im_h, im_w = img.shape[:2]
            
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                if y2 < im_h and x2 < im_w and x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2:
                    try:
                        crop = img[y1:y2, x1:x2]
                        result_list = self.model.predict(crop)
                        
                        if result_list:
                            result = result_list[0]
                            rec_texts = result.json.get('rec_texts', [])
                            
                            # 출력 형식에 따라 텍스트 조합
                            if self.output_format == "Single Line":
                                combined_text = ' '.join(rec_texts)
                            else:  # "As Recognized"
                                combined_text = '\n'.join(rec_texts)
                            
                            # 텍스트 처리 적용
                            final_text = self._apply_text_case(combined_text)
                            final_text = self._apply_punctuation_and_spacing(final_text)
                            
                            blk.text = [final_text]
                        else:
                            blk.text = ['']
                            
                    except Exception as e:
                        self.logger.exception('Paddle-OCRv5 블록 수준 인식 실패')
                        blk.text = ['']
                else:
                    self.logger.warning('invalid textbbox to target img')
                    blk.text = ['']

        def _apply_text_case(self, text: str) -> str:
            if self.text_case == "Uppercase":
                return text.upper()
            elif self.text_case == "Capitalize Sentences":
                return self._capitalize_sentences(text)
            elif self.text_case == "Lowercase":
                return text.lower()
            return text

        def _capitalize_sentences(self, text: str) -> str:
            def process_sentence(sentence):
                words = sentence.split()
                if not words: return ""
                if len(words) == 1: return words[0].capitalize()
                return " ".join([words[0].capitalize()] + [word.lower() for word in words[1:]])

            sentences = re.split(r"(?<=[.!?…])\s+", text)
            return " ".join(process_sentence(s) for s in sentences)

        def _apply_punctuation_and_spacing(self, text: str) -> str:
            text = re.sub(r"\s+([,.!?…])", r"\1", text)
            text = re.sub(r"([,.!?…])(?!\s)(?![,.!?…])", r"\1 ", text)
            text = re.sub(r"([,.!?…])\s+([,.!?…])", r"\1\2", text)
            return text.strip()

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            # 파라미터 업데이트 시 모델을 다시 로드해야 하는 항목들
            reload_needed_keys = [
                "language",
            ]
            
            if param_key in reload_needed_keys:
                self.language = self.params["language"]["value"]
                self._load_model()
            
            elif param_key == "text_rec_score_thresh":
                self.text_rec_score_thresh = self.params["text_rec_score_thresh"]["value"]
            elif param_key == "text_case":
                self.text_case = self.params["text_case"]["value"]
            elif param_key == "output_format":
                self.output_format = self.params["output_format"]["value"]

else:
    logging.info("PaddleOCR module will not be loaded as the library is not installed.")