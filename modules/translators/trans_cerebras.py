import re
import time
import requests
import yaml
import traceback
from typing import List, Dict

from .base import BaseTranslator, register_translator
from .sys_prompt import get_system_prompt, get_prefill

class InvalidNumTranslations(Exception):
    pass

@register_translator('Cerebras')
class CerebrasTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'Cerebras API Key': '', # API 키 이름 변경
        'model': {
            'type': 'selector',
            'options': [ # Cerebras에서 지원하는 모델 목록으로 변경
                'qwen-3-235b-a22b-instruct-2507',
                'qwen-3-235b-a22b-thinking-2507',
                'qwen-3-coder-480b',
                'gpt-oss-120b'
            ],
            'value': 'gpt-oss-120b'
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
        'max tokens': 4096, # max_completion_tokens에 해당
        'temperature': 0.6,
        'top p': 0.95,
        'retry attempts': 5,
        'retry timeout': 15,
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
        override_model = self.params['override model'].strip()
        return override_model if override_model else self.params['model']['value']

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

            if max_tokens * 2 and len(''.join(queries[i+1:])) > max_tokens:
                yield prompt.lstrip(), num_src
                prompt = prompt_template
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
                    message = f'number of translations does not match to source:\nprompt:\n    {prompt}\ntranslations:\n  {new_translations}\ncerebras response:\n  {response}'
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
                    self.logger.error(f'Request traceback: {traceback.format_exc()}')
                    time.sleep(self.retry_timeout)
            
            translations.extend([t.strip() for t in new_translations])

        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    def _request_translation(self, prompt: str, chat_sample: List) -> str:
        # === Cerebras API 호출을 위해 완전히 재작성된 부분 ===
        self.logger.debug(f'Cerebras prompt: \n {prompt}')

        api_key = self.params['Cerebras API Key'].strip()
        if not api_key:
            raise ValueError("Cerebras API Key is not set.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        source_lang = self.lang_map[self.lang_source]
        target_lang = self.lang_map[self.lang_target]
        
        # Cerebras API의 messages 형식에 맞게 데이터 구성
        messages = []
        # 1. 시스템 프롬프트 추가
        system_prompt = get_system_prompt(source_lang, target_lang)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. Few-shot 샘플 추가 (user -> assistant)
        if chat_sample:
            messages.append({"role": "user", "content": chat_sample[0]})
            messages.append({"role": "assistant", "content": chat_sample[1]})
        
        # 3. 실제 번역 요청 프롬프트 추가
        messages.append({"role": "user", "content": prompt})

        # 4. Prefill 추가 (모델의 응답 시작을 유도)
        # prefill_text = get_prefill()
        # if prefill_text:
        #     messages.append({"role": "assistant", "content": prefill_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": False
        }

        api_url = 'https://api.cerebras.ai/v1/chat/completions'
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=self.retry_timeout * 2)
            response.raise_for_status() # 200 OK가 아니면 예외 발생

            result = response.json()
            
            # 토큰 사용량 기록
            usage = result.get('usage', {})
            self.token_count_last = usage.get('total_tokens', 0)
            self.token_count += self.token_count_last

            # 번역 결과 텍스트 반환
            return result['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Cerebras API request error: {e}")
            if e.response:
                self.logger.error(f"Response Body: {e.response.text}")
            raise e