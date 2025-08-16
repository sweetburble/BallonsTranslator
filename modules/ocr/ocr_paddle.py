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

import cv2
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
            "Chinese & English": "ch",
            "English": "en",
            "French": "fr",
            "German": "german",
            "Japanese": "japan",
            "Korean": "korean",
            "Chinese Traditional": "chinese_cht",
            "Italian": "it",
            "Spanish": "es",
            "Portuguese": "pt",
            "Russian": "ru",
            "Ukrainian": "uk",
            "Belarusian": "be",
            "Telugu": "te",
            "Saudi Arabia": "sa",
            "Tamil": "ta",
            "Afrikaans": "af",
            "Azerbaijani": "az",
            "Bosnian": "bs",
            "Czech": "cs",
            "Welsh": "cy",
            "Danish": "da",
            "Dutch": "nl",
            "Norwegian": "no",
            "Polish": "pl",
            "Romanian": "ro",
            "Slovak": "sk",
            "Slovenian": "sl",
            "Albanian": "sq",
            "Swedish": "sv",
            "Swahili": "sw",
            "Tagalog": "tl",
            "Turkish": "tr",
            "Uzbek": "uz",
            "Vietnamese": "vi",
            "Mongolian": "mn",
            "Arabic": "ar",
            "Hindi": "hi",
            "Uyghur": "ug",
            "Persian": "fa",
            "Urdu": "ur",
            "Serbian (Latin)": "rs_latin",
            "Occitan": "oc",
            "Marathi": "mr",
            "Nepali": "ne",
            "Serbian (Cyrillic)": "rs_cyrillic",
            "Bulgarian": "bg",
            "Estonian": "et",
            "Irish": "ga",
            "Croatian": "hr",
            "Hungarian": "hu",
            "Indonesian": "id",
            "Icelandic": "is",
            "Kurdish": "ku",
            "Lithuanian": "lt",
            "Latvian": "lv",
            "Maori": "mi",
            "Malay": "ms",
            "Maltese": "mt",
            "Adyghe": "ady",
            "Kabardian": "kbd",
            "Avar": "ava",
            "Dargwa": "dar",
            "Ingush": "inh",
            "Lak": "lbe",
            "Lezghian": "lez",
            "Tabassaran": "tab",
            "Bihari": "bh",
            "Maithili": "mai",
            "Angika": "ang",
            "Bhojpuri": "bho",
            "Magahi": "mah",
            "Nagpur": "sck",
            "Newari": "new",
            "Goan Konkani": "gom",
        }

        params = {
            "language": {
                "type": "selector",
                "options": list(lang_map.keys()),
                "value": "English",  # Default language
                "description": "Select the language for OCR",
            },
            "device": DEVICE_SELECTOR(),
            "use_textline_orientation": {
                "type": "checkbox",
                "value": False,
                "description": "Enable angle classification for rotated text",
            },
            "ocr_version": {
                "type": "selector",
                "options": ["PP-OCRv5"],
                "value": "PP-OCRv5",
                "description": "Select the OCR model version",
            },
            "text_det_limit_side_len": {
                "value": 240,
                "description": "Maximum side length for text detection",
            },
            "text_recognition_batch_size": {
                "value": 6,
                "description": "Batch size for text recognition",
            },
            "text_rec_score_thresh": {
                "value": 0.5,
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
            self.device = self.params["device"]["value"]
            self.use_textline_orientation = self.params["use_textline_orientation"]["value"]
            # self.ocr_version = self.params["ocr_version"]["value"]
            self.ocr_version = "PP-OCRv5"
            self.text_det_limit_side_len = self.params["text_det_limit_side_len"]["value"]
            self.text_recognition_batch_size = self.params["text_recognition_batch_size"]["value"]
            self.text_rec_score_thresh = self.params["text_rec_score_thresh"]["value"]
            self.text_case = self.params["text_case"]["value"]
            self.output_format = self.params["output_format"]["value"]
            self.model = None
            self._setup_logging()
            self._load_model()

        def _setup_logging(self):
            if self.debug_mode:
                logging.getLogger("ppocr").setLevel(logging.DEBUG)
                logging.getLogger("paddleocr").setLevel(logging.DEBUG)
                logging.getLogger("predict_system").setLevel(logging.DEBUG)
            else:
                logging.getLogger("ppocr").setLevel(logging.WARNING)
                logging.getLogger("paddleocr").setLevel(logging.WARNING)
                logging.getLogger("predict_system").setLevel(logging.WARNING)

        def _load_model(self):
            lang_code = self.lang_map[self.language]
            use_gpu = True if self.device == "cuda" else False
            if self.debug_mode:
                self.logger.info(
                    f"Loading PaddleOCR model for language: {self.language} ({lang_code}), GPU: {use_gpu}"
                )
            self.model = PaddleOCR(
                use_textline_orientation=self.use_textline_orientation,
                lang=lang_code,
                # use_gpu=use_gpu,
                # enable_mkldnn=self.enable_mkldnn,
                text_rec_score_thresh=self.text_rec_score_thresh,
                ocr_version=self.ocr_version,
                text_det_limit_side_len=self.text_det_limit_side_len,
                text_recognition_batch_size=self.text_recognition_batch_size,
                text_detection_model_dir=os.path.join(
                    PADDLE_OCR_PATH, lang_code, self.ocr_version, "det"
                ),
                text_recognition_model_dir=os.path.join(
                    PADDLE_OCR_PATH, lang_code, self.ocr_version, "rec"
                ),
                textline_orientation_model_dir=(
                    os.path.join(PADDLE_OCR_PATH, lang_code, self.ocr_version, "cls")
                    if self.use_textline_orientation
                    else None
                ),
            )

        def ocr_img(self, img: np.ndarray) -> str:
            if self.debug_mode:
                self.logger.debug(f"Starting OCR for image size: {img.shape}")
            result = self.model.predict(img, )
            if self.debug_mode:
                self.logger.debug(f"OCR recognition result: {result}")
            text = self._process_result(result)
            return text

        def _ocr_blk_list(
            self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs
        ):
            im_h, im_w = img.shape[:2]
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                    cropped_img = img[y1:y2, x1:x2]
                    try:
                        result = self.model.predict(
                            cropped_img
                        )

                        # Extract raw text from OCR result
                        raw_texts = []
                        if (
                            isinstance(result, list)
                            and len(result) > 0
                            and isinstance(result[0], list)
                        ):
                            for line in result[0]:
                                if (
                                    isinstance(line, list)
                                    and len(line) > 1
                                    and isinstance(line[1], (list, tuple))
                                    and len(line[1]) > 0
                                ):
                                    raw_texts.append(line[1][0])
                        raw_text = " ".join(raw_texts)

                        if self.debug_mode:
                            self.logger.debug(
                                f"Raw OCR text from block ({x1}, {y1}, {x2}, {y2}): {raw_text}"
                            )

                        # Process the OCR result
                        text = self._process_result(result)

                        if self.debug_mode:
                            self.logger.debug(
                                f"Processed text from block ({x1}, {y1}, {x2}, {y2}): {text}"
                            )

                        blk.text = text if text else ""

                    except Exception as e:
                        if self.debug_mode:
                            self.logger.error(f"Error recognizing block: {str(e)}")
                        blk.text = ""
                else:
                    if self.debug_mode:
                        self.logger.warning(
                            "Invalid text block coordinates for target image"
                        )
                    blk.text = ""

        def _process_result(self, result):
            try:
                if not result or result[0] is None:
                    return ""

                if (
                    isinstance(result, list)
                    and len(result) > 0
                    and isinstance(result[0], list)
                ):
                    result = result[0]

                raw_texts = []
                for line in result:
                    if (
                        isinstance(line, list)
                        and len(line) > 1
                        and isinstance(line[1], (list, tuple))
                        and len(line[1]) > 0
                    ):
                        text = line[1][0]
                        raw_texts.append(text)

                # Depending on the output_format, we concatenate the lines
                if self.output_format == "Single Line":
                    joined_text = " ".join(raw_texts)
                    # Text cleaning
                    joined_text = re.sub(r"-(?!\w)", "", joined_text)
                    joined_text = re.sub(r"\s+", " ", joined_text)
                elif self.output_format == "As Recognized":
                    joined_text = " ".join(
                        raw_texts
                    )  # Combine with spaces to create a single text
                    # Clean up text, preserve line breaks
                    joined_text = re.sub(r"-(?!\w)", "", joined_text)
                    joined_text = re.sub(r"\s+", " ", joined_text)
                else:
                    joined_text = " ".join(raw_texts)
                    joined_text = re.sub(r"-(?!\w)", "", joined_text)
                    joined_text = re.sub(r"\s+", " ", joined_text)

                # Apply case conversion to all text
                processed_text = self._apply_text_case(joined_text)
                processed_text = self._apply_punctuation_and_spacing(processed_text)

                if self.debug_mode:
                    self.logger.debug(f"Final processed text: {processed_text}")

                return processed_text
            except Exception as e:
                if self.debug_mode:
                    self.logger.error(f"Error processing OCR result: {str(e)}")
                return ""

        def _apply_text_case(self, text: str) -> str:
            if self.text_case == "Uppercase":
                return text.upper()
            elif self.text_case == "Capitalize Sentences":
                return self._capitalize_sentences(text)
            elif self.text_case == "Lowercase":
                return text.lower()
            else:
                return text  # No change if the mode is not recognized

        def _capitalize_sentences(self, text: str) -> str:
            def process_sentence(sentence):
                words = sentence.split()
                if not words:
                    return ""
                if len(words) == 1:
                    return words[0].capitalize()
                else:
                    return " ".join(
                        [words[0].capitalize()] + [word.lower() for word in words[1:]]
                    )

            # We divide into sentences only by punctuation marks
            sentences = re.split(r"(?<=[.!?…])\s+", text)
            return " ".join(process_sentence(sentence) for sentence in sentences)

        def _apply_punctuation_and_spacing(self, text: str) -> str:
            text = re.sub(r"\s+([,.!?…])", r"\1", text)
            text = re.sub(r"([,.!?…])(?!\s)(?![,.!?…])", r"\1 ", text)
            text = re.sub(r"([,.!?…])\s+([,.!?…])", r"\1\2", text)
            return text.strip()

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in [
                "language",
                "device",
                "use_textline_orientation",
                "ocr_version",
                "text_det_limit_side_len",
                "text_recognition_batch_size",
                "text_rec_score_thresh",
            ]:
                self.language = self.params["language"]["value"]
                self.device = self.params["device"]["value"]
                self.use_textline_orientation = self.params["use_textline_orientation"]["value"]
                self.ocr_version = self.params["ocr_version"]["value"]
                self.text_det_limit_side_len = self.params["text_det_limit_side_len"]["value"]
                self.text_recognition_batch_size = self.params["text_recognition_batch_size"]["value"]
                self.text_rec_score_thresh = self.params["text_rec_score_thresh"]["value"]
                self._load_model()
            elif param_key == "text_case":
                self.text_case = self.params["text_case"]["value"]
            elif param_key == "output_format":
                self.output_format = self.params["output_format"]["value"]

else:
    # If PaddleOCR is not installed, you can define a stub or alternative module
    logging.info("PaddleOCR module will not be loaded as the library is not installed.")
