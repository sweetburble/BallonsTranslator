import re
import time
import requests
from typing import List, Dict, Union
import yaml
import traceback
from .base import BaseTranslator, register_translator
from .sys_prompt import get_system_prompt

class InvalidNumTranslations(Exception):
    pass

@register_translator('Grok')
class GrokTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'api key': '',
        'model': {
            'type': 'selector',
            'options': [
                'grok-4-fast-non-reasoning',
            ],
            'value': 'grok-4-fast-non-reasoning'
        },
        'override model': '',
        'prompt template': {
            'type': 'editor',
            'value': 'Translate This:\n',
        },
        'chat sample': {
            'type': 'editor',
            'value':
'''日本語-한국어:
    source:
        - 二人のちゅーを 目撃した ぼっちちゃん
        - ふたりさん
        - 大好きなお友達には あいさつ代わりに ちゅーするんだって
        - アイス あげた
        - 喜多ちゃんとは どどど どういった ご関係なのでしようか...
        - テレビで見た！
    target:
        - 둘의 키스를 목격한 봇치쨩
        - 두 분
        - 좋아하는 친구에게는 인사 대신에 뽀뽀를 한다고 해.
        - 아이스크림을 줬어.
        - 키타 쨩과는 어떤 관계일까요...
        - 텔레비전에서 봤어!'''
        },
        'invalid repeat count': 2,
        'max requests per minute': 20,
        'delay': 0.3,
        'max tokens': 4096,
        'temperature': 0.5,
        'top p': 1,
        'retry attempts': 5,
        'retry timeout': 15,
        'frequency penalty': 0.0,
        'presence penalty': 0.0,
        'base_url': 'https://api.x.ai/v1',
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
        self.lang_map['čeština'] = 'Czech'
        self.lang_map['Français'] = 'French'
        self.lang_map['Deutsch'] = 'German'
        self.lang_map['magyar nyelv'] = 'Hungarian'
        self.lang_map['Italiano'] = 'Italian'
        self.lang_map['Polski'] = 'Polish'
        self.lang_map['Português'] = 'Portuguese'
        self.lang_map['limba română'] = 'Romanian'
        self.lang_map['русский язык'] = 'Russian'
        self.lang_map['Español'] = 'Spanish'
        self.lang_map['Türk dili'] = 'Turkish'
        self.lang_map['украї́нська мо́ва'] = 'Ukrainian'
        self.lang_map['Thai'] = 'Thai'
        self.lang_map['Arabic'] = 'Arabic'
        self.lang_map['Malayalam'] = 'Malayalam'
        self.lang_map['Tamil'] = 'Tamil'
        self.lang_map['Hindi'] = 'Hindi'

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
    def top_p(self) -> int:
        return self.params['top p']

    @property
    def retry_attempts(self) -> int:
        return self.params['retry attempts']

    @property
    def retry_timeout(self) -> int:
        return self.params['retry timeout']

    @property
    def chat_sample(self):
        samples = self.params['chat sample']['value']
        try:
            samples = yaml.load(self.params['chat sample']['value'], Loader=yaml.FullLoader)
        except:
            self.logger.error(f'failed to load parse sample: {samples}')
            samples = {}
        
        src_tgt = self.lang_source + '-' + self.lang_target
        if src_tgt in samples:
            src_list = samples[src_tgt]['source']
            tgt_list = samples[src_tgt]['target']
            src_queries = ''
            tgt_queries = ''
            for i, (src, tgt) in enumerate(zip(src_list, tgt_list)):
                src_queries += f'\n<|{i+1}|>{src}'
                tgt_queries += f'\n<|{i+1}|>{tgt}'
            src_queries = src_queries.lstrip()
            tgt_queries = tgt_queries.lstrip()
            return [src_queries, tgt_queries]
        else:
            return None

    def _assemble_prompts(self, queries: List[str], from_lang: str = None, to_lang: str = None, max_tokens = None) -> List[str]:
        if from_lang is None:
            from_lang = self.lang_map[self.lang_source]
        if to_lang is None:
            to_lang = self.lang_map[self.lang_target]
        
        prompt = ''
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        prompt_template = self.params['prompt template']['value'].format(to_lang=to_lang).rstrip()
        prompt += prompt_template
        
        i_offset = 0
        num_src = 0
        for i, query in enumerate(queries):
            prompt += f'\n<|{i+1-i_offset}|>{query}'
            num_src += 1
            
            # 프롬프트가 너무 커지고 아직 많은 텍스트가 남아있는 경우
            # 남은 쿼리를 새 프롬프트로 분할합니다
            if len(prompt) > max_tokens * 2 and len(''.join(queries[i+1:])) > max_tokens:
                yield prompt.lstrip(), num_src
                prompt = prompt_template
                # 카운팅을 1부터 다시 시작
                i_offset = i + 1
                num_src = 0
        
        yield prompt.lstrip(), num_src

    def _translate(self, src_list: List[str]) -> List[str]:
        translations = []
        
        from_lang = self.lang_map[self.lang_source]
        to_lang = self.lang_map[self.lang_target]
        
        queries = src_list
        chat_sample = self.chat_sample
        
        for prompt, num_src in self._assemble_prompts(queries, from_lang, to_lang):
            retry_attempt = 0
            while True:
                try:
                    response = self._request_translation(prompt, chat_sample)
                    new_translations = re.split(r'<\|\d+\|>', response)[-num_src:]
                    
                    if len(new_translations) != num_src:
                        _tr2 = re.sub(r'<\|\d+\|>', '', response)
                        _tr2 = _tr2.split('\n')
                        if len(_tr2) == num_src:
                            new_translations = _tr2
                        else:
                            raise InvalidNumTranslations
                    break
                except InvalidNumTranslations:
                    retry_attempt += 1
                    message = f'number of translations does not match to source:\nprompt:\n {prompt}\ntranslations:\n {new_translations}\ngrok response:\n {response}'
                    if retry_attempt >= self.retry_attempts:
                        self.logger.error(message)
                        new_translations = [''] * num_src
                        break
                    self.logger.warning(message + '\n' + f'Restarting request. Attempt: {retry_attempt}')
                except Exception as e:
                    retry_attempt += 1
                    if retry_attempt >= self.retry_attempts:
                        new_translations = [''] * num_src
                        break
                    self.logger.warning(f'Translation failed due to {e}. Attempt: {retry_attempt}, sleep for {self.retry_timeout} secs...')
                    self.logger.error(f'Request traceback: %s', traceback.format_exc())
                    time.sleep(self.retry_timeout)
            
            translations.extend([t.strip() for t in new_translations])
        
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        
        return translations

    def _request_translation(self, prompt, chat_sample: List):
        self.logger.debug(f'grok prompt: \n {prompt}')
        
        api_key = self.params['api key'].strip()
        base_url = self.params['base_url'].strip()
        
        override_model = self.params['override model'].strip()
        if override_model != '':
            model_name: str = override_model
        else:
            model_name: str = self.model
        
        # 시스템 프롬프트 로딩
        source_lang = self.lang_map[self.lang_source]
        target_lang = self.lang_map[self.lang_target]
        
        # 메시지 구성
        messages = []
        
        # 시스템 메시지 추가
        system_prompt = get_system_prompt(source_lang, target_lang)
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 샘플 메시지 추가
        if chat_sample is not None:
            messages.append({
                "role": "user",
                "content": chat_sample[0]
            })
            messages.append({
                "role": "assistant",
                "content": chat_sample[1]
            })
        
        # 유저 메시지 추가
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # API 요청 구성
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens // 2
        }
        
        try:
            # API 요청
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # 응답 검증
            if response.status_code != 200:
                error_message = f"Grok API returned error: {response.status_code}, {response.text}"
                self.logger.error(error_message)
                raise Exception(error_message)
            
            # 응답 처리
            response_data = response.json()
            
            # 토큰 사용량 기록
            if 'usage' in response_data:
                self.token_count_last = response_data['usage'].get('total_tokens', 0)
                self.token_count += self.token_count_last
            
            # 응답 텍스트 추출
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                raise Exception("Unexpected API response format")
            
        except Exception as e:
            self.logger.error(f"Grok API request error: {e}")
            self.logger.error(f'Request traceback: %s', traceback.format_exc())
            raise e
