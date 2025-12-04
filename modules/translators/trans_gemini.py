import re
import time
import json
from typing import List, Dict, Union
import yaml
import traceback
from pydantic import BaseModel

from google import genai
from google.genai import types

from .base import BaseTranslator, register_translator
from .sys_prompt import get_system_prompt

# Structured Output을 위한 스키마 정의
class TranslationResponse(BaseModel):
    translations: List[str]

class InvalidNumTranslations(Exception):
    pass

@register_translator('Gemini')
class GeminiTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'api key': '',
        'model': {
            'type': 'selector',
            'options': [
                'gemini-flash-lite-latest',
                'gemini-flash-latest',
                'gemini-3-pro-preview'
            ],
            'value': 'gemini-flash-latest'
        },
        'override model': '',   
        'prompt template': {
            'type': 'editor',
            'value': 'Translate the following list of texts from {src_lang} to {tgt_lang}. Output strictly in JSON format.',
            'description': 'Instruction for the model.'
        },
        'chat sample': {
            'type': 'editor',
            'value':
'''日本語-한국어:
    source:
        - 二人のちゅーを 目撃した ぼっちちゃん
        - ふたりさん
        - 大好きなお友達には あいさつ代わりに ちゅーするんだって
    target:
        - 둘의 키스를 목격한 봇치쨩
        - 두 분
        - 좋아하는 친구에게는 인사 대신에 뽀뽀를 한다고 해.'''
        },
        'invalid repeat count': 2,
        'max requests per minute': 20,
        'delay': 0.3,
        'max tokens': 8192, # JSON 모드는 토큰을 좀 더 쓸 수 있으므로 여유있게
        'temperature': 0.1, # 포맷 준수를 위해 낮은 temperature 권장
        'top p': 0.95,
        'retry attempts': 5,
        'retry timeout': 15,
        'frequency penalty': 0.0,
        'presence penalty': 0.0,
        'low vram mode': {
            'value': False,
            'description': 'check it if you\'re running it locally on a single device and encountered a crash due to vram OOM',
            'type': 'checkbox',
        }
    }

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'Simplified Chinese'
        self.lang_map['繁體中文'] = 'Traditional Chinese'
        self.lang_map['日本語'] = 'Japanese'
        self.lang_map['English'] = 'English'
        self.lang_map['한국어'] = 'Korean'
        self.lang_map['Tiếng Việt'] = 'Vietnamese'
        self.lang_map['Français'] = 'French'
        self.lang_map['Deutsch'] = 'German'
        self.lang_map['Italiano'] = 'Italian'
        self.lang_map['Português'] = 'Portuguese'
        self.lang_map['русский язык'] = 'Russian'
        self.lang_map['Español'] = 'Spanish'

        self.token_count = 0
        self.token_count_last = 0
    
    @property
    def model(self) -> str:
        return self.params['model']['value']

    @property
    def temperature(self) -> float:
        return self.params['temperature']
    
    @property
    def max_tokens(self) -> int:
        return self.params['max tokens']
    
    @property
    def top_p(self) -> float:
        return self.params['top p']
    
    @property
    def retry_attempts(self) -> int:
        return self.params['retry attempts']
    
    @property
    def retry_timeout(self) -> int:
        return self.params['retry timeout']
    
    @property
    def chat_sample(self):
        """
        YAML 샘플을 파싱하여 (source_list, target_list) 튜플로 반환합니다.
        Structured Output 훈련을 위해 리스트 형태 그대로 유지합니다.
        """
        samples_raw = self.params['chat sample']['value']
        try: 
            samples = yaml.load(samples_raw, Loader=yaml.FullLoader)
        except:
            self.logger.error(f'failed to load parse sample: {samples_raw}')
            samples = {}
            
        src_tgt = self.lang_source + '-' + self.lang_target
        if src_tgt in samples:
            # 리스트 그대로 반환 (JSON 직렬화는 요청 시 수행)
            return samples[src_tgt]['source'], samples[src_tgt]['target']
        else:
            return None

    def _assemble_prompts(self, queries: List[str], from_lang: str = None, to_lang: str = None, max_tokens = None) -> List[str]:
        """
        쿼리 리스트를 JSON 문자열 덩어리로 묶어서 반환합니다.
        """
        if from_lang is None:
            from_lang = self.lang_map[self.lang_source]
        if to_lang is None:
            to_lang = self.lang_map[self.lang_target]
            
        if max_tokens is None:
            max_tokens = self.max_tokens

        # 프롬프트 템플릿 (JSON 포맷 강조는 시스템 프롬프트/Schema 설정에서 처리되지만, 명시적으로 추가)
        base_instruction = self.params['prompt template']['value'].format(src_lang=from_lang, tgt_lang=to_lang)
        
        current_batch = []
        current_batch_len = 0
        
        # 예상되는 JSON 오버헤드 (대략적인 값)
        overhead = len(base_instruction) + 100 

        for query in queries:
            # 쿼리 자체 길이 + JSON 리스트로 변환했을 때의 대략적 추가 길이(따옴표, 콤마 등)
            query_len = len(query) + 10 
            
            # 현재 배치가 너무 커지면 yield
            if current_batch and (current_batch_len + query_len + overhead) > max_tokens:
                # 리스트를 JSON 문자열로 변환하여 프롬프트로 사용
                input_json = json.dumps(current_batch, ensure_ascii=False)
                full_prompt = f"{base_instruction}\n\nInput List:\n{input_json}"
                yield full_prompt, len(current_batch)
                
                current_batch = []
                current_batch_len = 0
            
            current_batch.append(query)
            current_batch_len += query_len

        # 남은 배치 처리
        if current_batch:
            input_json = json.dumps(current_batch, ensure_ascii=False)
            full_prompt = f"{base_instruction}\n\nInput List:\n{input_json}"
            yield full_prompt, len(current_batch)

    def _translate(self, src_list: List[str]) -> List[str]:
        translations = []
        from_lang = self.lang_map[self.lang_source]
        to_lang = self.lang_map[self.lang_target]
        
        chat_sample = self.chat_sample # (src_list, tgt_list) or None
        
        for prompt, num_src in self._assemble_prompts(src_list, from_lang, to_lang):
            retry_attempt = 0
            while True:
                try:
                    # JSON 응답 요청
                    response_obj = self._request_translation(prompt, chat_sample)
                    
                    # Pydantic 모델이나 dict로 반환됨. translations 리스트 추출
                    if isinstance(response_obj, BaseModel):
                        new_translations = response_obj.translations
                    elif isinstance(response_obj, dict):
                        new_translations = response_obj.get('translations', [])
                    else:
                        # Fallback: 혹시 텍스트로 왔을 경우 JSON 파싱 시도
                        try:
                            parsed = json.loads(response_obj)
                            new_translations = parsed.get('translations', [])
                        except:
                            raise ValueError(f"Unexpected response format: {response_obj}")

                    # 개수 검증
                    if len(new_translations) != num_src:
                        raise InvalidNumTranslations(f"Expected {num_src} items, got {len(new_translations)}")
                    
                    translations.extend([str(t).strip() for t in new_translations])
                    break

                except InvalidNumTranslations as e:
                    retry_attempt += 1
                    self.logger.warning(f"Translation count mismatch: {e}. Attempt: {retry_attempt}")
                    if retry_attempt >= self.retry_attempts:
                        self.logger.error("Max retries reached for count mismatch.")
                        translations.extend([''] * num_src) # 실패 시 빈 문자열로 채움
                        break
                        
                except Exception as e:
                    retry_attempt += 1
                    self.logger.error(f'Translation failed: {e}')
                    self.logger.error(traceback.format_exc())
                    
                    if retry_attempt >= self.retry_attempts:
                        translations.extend([''] * num_src)
                        break
                    time.sleep(self.retry_timeout)

        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    def _request_translation(self, prompt, chat_sample):
        # self.logger.debug(f'gemini prompt: \n {prompt}' )

        client = genai.Client(api_key=self.params['api key'].strip())
        
        override_model = self.params['override model'].strip()
        model_name = override_model if override_model else self.model

        source_lang = self.lang_map[self.lang_source]
        target_lang = self.lang_map[self.lang_target]

        # Structured Output 설정
        config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
            system_instruction=get_system_prompt(source_lang, target_lang),
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
            # JSON 응답 강제 설정
            response_mime_type="application/json",
            response_schema=TranslationResponse, 
        )

        contents = []
        
        # Few-shot 예제 처리 (JSON 구조에 맞게 변환)
        if chat_sample is not None:
            sample_src, sample_tgt = chat_sample
            # 예제 입력: JSON 문자열 형태의 리스트
            user_sample_content = "Translate the following list:\n" + json.dumps(sample_src, ensure_ascii=False)
            # 예제 출력: 스키마에 맞는 JSON 구조
            model_sample_content = json.dumps({"translations": sample_tgt}, ensure_ascii=False)
            
            contents.append(types.Content(role='user', parts=[types.Part(text=user_sample_content)]))
            contents.append(types.Content(role='model', parts=[types.Part(text=model_sample_content)]))

        # 실제 사용자 프롬프트 추가
        contents.append(types.Content(role='user', parts=[types.Part(text=prompt)]))

        try:
            response = client.models.generate_content(
                model=model_name, 
                contents=contents, 
                config=config
            )
            
            # 토큰 사용량 기록
            if hasattr(response, 'usage_metadata'):
                self.token_count_last = response.usage_metadata.total_token_count
                self.token_count += self.token_count_last

            # Structured Output은 response.parsed를 통해 객체로 바로 접근 가능할 수 있음 (SDK 버전에 따라 다름)
            # text로 받아서 파싱하는 것이 가장 안전
            text_resp = response.candidates[0].content.parts[0].text
            return json.loads(text_resp)
            
        except Exception as e:
            self.logger.error(f"Gemini API request error: {e}")
            raise e