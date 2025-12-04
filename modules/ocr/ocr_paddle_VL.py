import numpy as np
import json
import cv2
import requests
import base64
from typing import List, Any

from .base import register_OCR, OCRBase, TextBlock
from utils.message import create_error_dialog, create_info_dialog


@register_OCR('paddle_vl')
class OCRPaddleVL(OCRBase):
    params = {
        'server_url': 'http://127.0.0.1:8080/layout-parsing',
        'prettifyMarkdown': {'type': 'checkbox', 'value': False},
        'visualize': {'type': 'checkbox', 'value': False},
        'description': '로컬에 배포된 Paddle OCR-VL 서비스 (POST /layout-parsing)'
    }

    @property
    def server_url(self):
        val = self.params.get('server_url')
        # UI가 파라미터를 {'value': 'http://...', 'data_type': <class 'str'>}와 같은 딕셔너리로 감쌀 수 있음
        if isinstance(val, dict):
            return val.get('value') or val.get('text') or ''
        return val or ''

    @property
    def prettifyMarkdown(self):
        v = self.params.get('prettifyMarkdown')
        if isinstance(v, dict):
            return bool(v.get('value', False))
        return bool(v)

    @property
    def visualize(self):
        v = self.params.get('visualize')
        if isinstance(v, dict):
            return bool(v.get('value', False))
        return bool(v)

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.debug = False

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        """
        각 텍스트 블록을 개별적으로 크롭하여 로컬 Paddle-VL 서비스를 호출하여 인식합니다.
        이렇게 하면 기존 블록 수준 워크플로우와 호환됩니다 (TextBlock API 유지).
        """
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if y2 < im_h and x2 < im_w and x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2:
                try:
                    crop = img[y1:y2, x1:x2]
                    blk.text = self.ocr(crop)
                except Exception as e:
                    self.logger.exception('Paddle-VL 블록 수준 인식 실패')
                    blk.text = ['']
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ['']

    def ocr_img(self, img: np.ndarray) -> str:
        self.logger.debug(f'ocr_img: {img.shape}')
        return self.ocr(img)

    def _extract_texts_from_pruned(self, pruned: Any) -> List[str]:
        texts: List[str] = []

        def walk(node: Any):
            if node is None:
                return
            if isinstance(node, dict):
                # 일반적인 키에는 'texts' 또는 'text'가 포함될 수 있음
                if 'texts' in node and isinstance(node['texts'], (list, str)):
                    if isinstance(node['texts'], list):
                        texts.append(''.join(node['texts']).strip())
                    else:
                        texts.append(str(node['texts']).strip())
                if 'text' in node and isinstance(node['text'], str):
                    texts.append(node['text'].strip())
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for it in node:
                    walk(it)
            elif isinstance(node, str):
                texts.append(node.strip())

        walk(pruned)
        # 빈 문자열 필터링 및 인접 중복 제거
        return [t for t in texts if t]

    def _markdown_to_text(self, md: str) -> str:
        """
        Markdown을 간단히 일반 텍스트로 변환:
        - 이미지 구문 제거 ![...](...)
        - 링크 [text](url) -> text로 변환
        - 제목 앞의 # 제거
        - 강조 기호 제거 (*, _, **)
        - 인라인 코드 및 HTML 태그 제거
        - 연속된 빈 줄 병합 및 앞뒤 공백 제거
        """
        if not md:
            return ''
        try:
            import re

            # 이미지 마크다운 제거
            md = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', md)
            # 링크 [text](url) -> text로 변환
            md = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md)
            # 줄 시작의 제목 마커 제거
            md = re.sub(r'(?m)^\s{0,3}#{1,6}\s*', '', md)
            # 굵게/기울임 마커 제거 (*, _, **, __)
            md = re.sub(r'(\*\*|__)(.*?)\1', r'\2', md)
            md = re.sub(r'(\*|_)(.*?)\1', r'\2', md)
            # 인라인 코드 백틱 제거
            md = re.sub(r'`([^`]*)`', r'\1', md)
            # 남아있는 HTML 태그 제거
            md = re.sub(r'<[^>]+>', '', md)
            # 공백 정규화 및 여러 빈 줄 제거
            md = re.sub(r"\r\n|\r", "\n", md)
            md = re.sub(r"\n{2,}", "\n", md)
            md = md.strip()
            return md
        except Exception:
            return md

    def ocr(self, img: np.ndarray) -> str:
        """
        이미지(단일 또는 블록)를 Base64로 로컬 Paddle-VL 서비스의 `/layout-parsing`에 전송합니다.
        반환된 Markdown 텍스트를 우선 사용하며, 없으면 prunedResult에서 텍스트를 추출하려고 시도합니다.
        문자열을 반환합니다 (전체 블록 인식 결과).
        """
        try:
            image_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        except Exception as e:
            self.logger.exception('이미지 인코딩 실패')
            raise

        image_b64 = base64.b64encode(image_bytes).decode('ascii')

        payload = {
            'file': image_b64,
            'fileType': 1,
            'prettifyMarkdown': self.prettifyMarkdown,
            'visualize': self.visualize,
        }

        try:
            resp = requests.post(self.server_url, json=payload, timeout=60)
        except Exception as e:
            self.logger.exception('로컬 Paddle-VL 서비스 요청 실패')
            raise

        if resp.status_code != 200:
            self.logger.error(f'Paddle-VL 요청 실패, 상태 코드: {resp.status_code}')
            raise ValueError(f'Paddle-VL 요청 실패, 상태 코드: {resp.status_code}')

        try:
            data = resp.json()
        except Exception:
            self.logger.exception('Paddle-VL 응답 JSON 파싱 실패')
            raise

        # Paddle 서비스 표준 응답: { logId, errorCode, errorMsg, result }
        if 'errorCode' in data and data.get('errorCode', -1) != 0:
            msg = data.get('errorMsg', '')
            self.logger.error(f'Paddle-VL 오류 반환: {msg}')
            raise ValueError(f'Paddle-VL 오류 반환: {msg}')

        result = data.get('result', data)
        lprs = result.get('layoutParsingResults') or []
        if not lprs:
            # layoutParsingResults가 없으면 result에서 직접 파싱을 시도합니다
            # 마지막으로 전체 응답을 문자열로 반환합니다 (디버그용)
            self.logger.debug('layoutParsingResults를 찾을 수 없음, 전체 응답 텍스트 반환')
            return json.dumps(result, ensure_ascii=False)

        first = lprs[0]
        # Markdown을 우선 사용하되, Markdown을 일반 텍스트로 정리합니다
        md_raw = first.get('markdown', {}).get('text') if isinstance(first.get('markdown'), dict) else None
        if md_raw:
            md_txt = self._markdown_to_text(md_raw)
            if md_txt:
                return md_txt

        # 그렇지 않으면 prunedResult에서 texts 필드를 추출하려고 시도합니다
        pruned = first.get('prunedResult')
        if pruned is not None:
            texts = self._extract_texts_from_pruned(pruned)
            if texts:
                # 결과를 결합하고 정리하여 가능한 마크다운 아티팩트를 제거합니다
                joined = '\n'.join(texts)
                return self._markdown_to_text(joined)

        # 마지막으로 outputImages 또는 pruned의 JSON 문자열로 되돌립니다
        return json.dumps(first, ensure_ascii=False)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        # server_url 등의 파라미터가 변경될 때 사용자에게 알립니다
        if param_key == 'server_url':
            create_info_dialog('Paddle-VL 서비스 주소가 업데이트되었습니다')
