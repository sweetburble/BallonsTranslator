import re
import time
import json
import os
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
    
    # params의 초기값은 UI 렌더링을 위한 뼈대만 남겨둡니다.
    # 실제 값은 _setup_translator에서 동적으로 로드됩니다.
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
            'value': '', # 초기값은 비워두거나 기본 템플릿을 둡니다.
            'description': 'Few-shot examples for the current language pair. (YAML format)',
        },
        'invalid repeat count': 2,
        'max requests per minute': 20,
        'delay': 0.3,
        'max tokens': 8192,
        'temperature': 0.1,
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
        # 1. 언어 매핑 설정
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

        # 2. chat_sample.json 파일 경로 설정
        self.sample_file_path = os.path.join(os.path.dirname(__file__), 'chat_sample.json')
        
        # 3. 현재 언어 쌍에 맞는 샘플 로드 및 UI 업데이트
        self._load_sample_to_ui()

    def set_source(self, lang: str):
        # 1. 부모 클래스의 로직을 먼저 수행하여 self.lang_source 업데이트
        super().set_source(lang)
        # 2. 변경된 언어에 맞춰 샘플 UI 갱신
        self._load_sample_to_ui()

    def set_target(self, lang: str):
        # 1. 부모 클래스의 로직을 먼저 수행하여 self.lang_target 업데이트
        super().set_target(lang)
        # 2. 변경된 언어에 맞춰 샘플 UI 갱신
        self._load_sample_to_ui()

    def _get_lang_key(self):
        """JSON 파일에서 사용할 키 생성 (예: 日本語-한국어)"""
        return f"{self.lang_source}-{self.lang_target}"

    def _load_sample_to_ui(self):
        """JSON 파일에서 현재 언어 쌍의 예제를 불러와 UI(params)에 설정"""
        key = self._get_lang_key()
        samples = {}

        # 파일이 존재하면 로드
        if os.path.exists(self.sample_file_path):
            try:
                with open(self.sample_file_path, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load chat_sample.json: {e}")

        # 해당 언어 쌍의 샘플이 있으면 YAML로 변환하여 UI 값으로 설정
        if key in samples:
            data = samples[key]
            # UI 가독성을 위해 YAML 형식으로 변환하여 표시
            yaml_str = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
            self.params['chat sample']['value'] = yaml_str.strip()
        else:
            # 샘플이 없으면 사용자가 채워넣을 수 있도록 기본 템플릿 제공
            default_template = {
                'source': ['Origin Text 1', 'Origin Text 2'],
                'target': ['Translated Text 1', 'Translated Text 2']
            }
            self.params['chat sample']['value'] = yaml.dump(default_template, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def _save_sample_from_ui(self, source_list: List[str], target_list: List[str]):
        """현재 UI에서 파싱된 유효한 샘플을 chat_sample.json에 저장/업데이트"""
        key = self._get_lang_key()
        new_data = {
            'source': source_list,
            'target': target_list
        }

        # 기존 파일 로드
        all_samples = {}
        if os.path.exists(self.sample_file_path):
            try:
                with open(self.sample_file_path, 'r', encoding='utf-8') as f:
                    all_samples = json.load(f)
            except:
                pass # 파일이 깨져있거나 없으면 새로 작성

        # 데이터 업데이트 (변경사항이 있을 때만 저장하면 좋지만, 단순화를 위해 덮어쓰기)
        # 내용이 실질적으로 바뀌었는지 체크할 수도 있음
        current_entry = all_samples.get(key, {})
        if current_entry != new_data:
            all_samples[key] = new_data
            try:
                with open(self.sample_file_path, 'w', encoding='utf-8') as f:
                    json.dump(all_samples, f, ensure_ascii=False, indent=4)
                # self.logger.info(f"Updated chat sample for {key}")
            except Exception as e:
                self.logger.error(f"Failed to save chat_sample.json: {e}")

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
        UI(YAML 문자열)에서 샘플을 파싱하여 반환하고, 동시에 파일에 업데이트합니다.
        """
        samples_raw = self.params['chat sample']['value']
        try: 
            # YAML 파싱
            parsed = yaml.load(samples_raw, Loader=yaml.FullLoader)
            
            # 구조 검증 (source, target 리스트 확인)
            if isinstance(parsed, dict) and 'source' in parsed and 'target' in parsed:
                src_list = parsed['source']
                tgt_list = parsed['target']
                
                # 유효한 샘플이라면 파일에 자동 저장 (User가 UI에서 수정했을 수 있으므로)
                self._save_sample_from_ui(src_list, tgt_list)
                
                return src_list, tgt_list
            else:
                return None
        except Exception as e:
            self.logger.error(f'failed to parse chat sample from UI: {e}')
            return None

    def _assemble_prompts(self, queries: List[str], from_lang: str = None, to_lang: str = None, max_tokens = None) -> List[str]:
        # (기존 코드와 동일)
        if from_lang is None:
            from_lang = self.lang_map[self.lang_source]
        if to_lang is None:
            to_lang = self.lang_map[self.lang_target]
            
        if max_tokens is None:
            max_tokens = self.max_tokens

        base_instruction = self.params['prompt template']['value'].format(src_lang=from_lang, tgt_lang=to_lang)
        
        current_batch = []
        current_batch_len = 0
        overhead = len(base_instruction) + 100 

        for query in queries:
            query_len = len(query) + 10 
            if current_batch and (current_batch_len + query_len + overhead) > max_tokens:
                input_json = json.dumps(current_batch, ensure_ascii=False)
                full_prompt = f"{base_instruction}\n\nInput List:\n{input_json}"
                yield full_prompt, len(current_batch)
                
                current_batch = []
                current_batch_len = 0
            
            current_batch.append(query)
            current_batch_len += query_len

        if current_batch:
            input_json = json.dumps(current_batch, ensure_ascii=False)
            full_prompt = f"{base_instruction}\n\nInput List:\n{input_json}"
            yield full_prompt, len(current_batch)

    def _translate(self, src_list: List[str]) -> List[str]:
        translations = []
        from_lang = self.lang_map[self.lang_source]
        to_lang = self.lang_map[self.lang_target]
        
        # 여기서 프로퍼티를 호출하여 최신 UI 값을 가져오고 파일 저장도 수행됨
        chat_sample = self.chat_sample 
        
        for prompt, num_src in self._assemble_prompts(src_list, from_lang, to_lang):
            retry_attempt = 0
            while True:
                try:
                    response_obj = self._request_translation(prompt, chat_sample)
                    
                    if isinstance(response_obj, BaseModel):
                        new_translations = response_obj.translations
                    elif isinstance(response_obj, dict):
                        new_translations = response_obj.get('translations', [])
                    else:
                        try:
                            parsed = json.loads(response_obj)
                            new_translations = parsed.get('translations', [])
                        except:
                            raise ValueError(f"Unexpected response format: {response_obj}")

                    if len(new_translations) != num_src:
                        raise InvalidNumTranslations(f"Expected {num_src} items, got {len(new_translations)}")
                    
                    translations.extend([str(t).strip() for t in new_translations])
                    break

                except InvalidNumTranslations as e:
                    retry_attempt += 1
                    self.logger.warning(f"Translation count mismatch: {e}. Attempt: {retry_attempt}")
                    if retry_attempt >= self.retry_attempts:
                        self.logger.error("Max retries reached for count mismatch.")
                        translations.extend([''] * num_src)
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
        # (기존 코드와 동일)
        client = genai.Client(api_key=self.params['api key'].strip())
        
        override_model = self.params['override model'].strip()
        model_name = override_model if override_model else self.model

        source_lang = self.lang_map[self.lang_source]
        target_lang = self.lang_map[self.lang_target]

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
            response_mime_type="application/json",
            response_schema=TranslationResponse, 
        )

        contents = []
        
        if chat_sample is not None:
            sample_src, sample_tgt = chat_sample
            user_sample_content = "Translate the following list:\n" + json.dumps(sample_src, ensure_ascii=False)
            model_sample_content = json.dumps({"translations": sample_tgt}, ensure_ascii=False)
            
            contents.append(types.Content(role='user', parts=[types.Part(text=user_sample_content)]))
            contents.append(types.Content(role='model', parts=[types.Part(text=model_sample_content)]))

        contents.append(types.Content(role='user', parts=[types.Part(text=prompt)]))

        try:
            response = client.models.generate_content(
                model=model_name, 
                contents=contents, 
                config=config
            )
            
            if hasattr(response, 'usage_metadata'):
                self.token_count_last = response.usage_metadata.total_token_count
                self.token_count += self.token_count_last

            text_resp = response.candidates[0].content.parts[0].text
            return json.loads(text_resp)
            
        except Exception as e:
            self.logger.error(f"Gemini API request error: {e}")
            raise e