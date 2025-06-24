import re
import time
from typing import List, Dict, Union
import yaml
import traceback

from openai import OpenAI # 변경: google.generativeai -> openai
from .base import BaseTranslator, register_translator

from .sys_prompt import get_system_prompt


class InvalidNumTranslations(Exception):
    pass

@register_translator('Deepseek')
class DeepseekTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'api key': '',
        'prompt template': {
            'type': 'editor',
            'value': 'Please help me to translate the following text from a manga to {to_lang} (if it\'s already in {to_lang} or looks like gibberish you have to output it as it is instead):\n',
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
        - 둘의 키스를 목격한 혼자 있는 아이
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
        # 'return prompt': False,
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
        api_key = self.params['api key'].strip()

        # 개행 문자와 공백 모두 제거
        api_key = api_key.replace('\n', '').replace('\r', '').strip()

        # API 키가 비어있는지 확인
        if not api_key:
            raise ValueError("DeepSeek API 키가 비어 있거나 올바르게 구성되지 않았습니다")

        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1") # Deepseek API 클라이언트 초기화

    @property
    def model(self) -> str:
        return "deepseek-chat" # 모델은 deepseek-chat 으로 고정

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
    def chat_system_template(self) -> str:
        from_lang = self.lang_map[self.lang_source]
        to_lang = self.lang_map[self.lang_target]
        return self.params['chat system template']['value'].format(from_lang=from_lang, to_lang=to_lang)

    @property
    def chat_sample(self): # chat_sample property 는 Gemini 코드 참고하여 유지. 필요에 따라 제거 가능
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
        # return_prompt = self.params['return prompt']
        prompt_template = self.params['prompt template']['value'].format(to_lang=to_lang).rstrip()
        prompt += prompt_template

        i_offset = 0
        num_src = 0
        for i, query in enumerate(queries):
            prompt += f'\n<|{i+1-i_offset}|>{query}'
            num_src += 1
            # If prompt is growing too large and theres still a lot of text left
            # split off the rest of the queries into new prompts.
            # 1 token = ~4 characters according to https://platform.openai.com/tokenizer
            # TODO: potentially add summarizations from special requests as context information
            if max_tokens * 2 and len(''.join(queries[i+1:])) > max_tokens:
                # if return_prompt:
                #     prompt += '\n<|1|>'
                yield prompt.lstrip(), num_src
                prompt = prompt_template
                # Restart counting at 1
                i_offset = i + 1
                num_src = 0

        # if return_prompt:
        #     prompt += '\n<|1|>'
        yield prompt.lstrip(), num_src

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        chat_sample = self.chat_sample # chat_sample 은 Deepseek API 에서는 불필요할 수 있지만, 일단 Gemini 코드를 참고하여 유지. 필요에 따라 제거 가능
        if chat_sample is not None:
            return '\n'.join([
                'System:',
                self.chat_system_template,
                'User:',
                chat_sample[0],
                'Assistant:',
                chat_sample[1],
                'User:',
                prompt,
            ])
        else:
            return '\n'.join([
                'System:',
                self.chat_system_template,
                'User:',
                prompt,
            ])

    def _translate(self, src_list: List[str]) -> List[str]:
        translations = []
        # self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')
        from_lang = self.lang_map[self.lang_source]
        to_lang = self.lang_map[self.lang_target]
        queries = src_list
        # return_prompt = self.params['return prompt']
        chat_sample = self.chat_sample # chat_sample 은 Deepseek API 에서는 불필요할 수 있지만, 일단 Gemini 코드를 참고하여 유지. 필요에 따라 제거 가능
        for prompt, num_src in self._assemble_prompts(queries, from_lang, to_lang):
            retry_attempt = 0
            while True:
                try:
                    response = self.get_deepseek_translation(prompt)
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
                    message = f'number of translations does not match to source:\nprompt:\n    {prompt}\ntranslations:\n  {new_translations}\ndeepseek response:\n  {response}'
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
                    self.logger.error(f'Request traceback: %s', traceback.format_exc()) # 변경: TypeError 해결을 위해 format string 변경
                    time.sleep(self.retry_timeout)
                    # time.sleep(self.retry_timeout)
            # if return_prompt:
            #     new_translations = new_translations[:-1]

            # if chat_sample is not None:
            #     new_translations = new_translations[1:]
            translations.extend([t.strip() for t in new_translations])

        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    def get_deepseek_translation(self, user_prompt: str): # Deepseek API 호출 함수
        # 시스템 프롬프트 로딩을 위해
        source_lang = self.lang_map[self.lang_source]
        target_lang = self.lang_map[self.lang_target]
        
        system_prompt = get_system_prompt(source_lang, target_lang)

        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model, # 모델 이름은 속성에서 가져옴
                messages=message,
                temperature=self.temperature, # temperature 속성 사용
                max_tokens=self.max_tokens, # max_tokens 속성 사용
                top_p=self.top_p # top_p 속성 사용
            )
            translated = response.choices[0].message.content
            return translated
        except Exception as e:
            self.logger.error(f"Deepseek API request error: {e}") # 에러 메시지 변경
            self.logger.error(f'Request traceback: %s', traceback.format_exc()) # 변경: TypeError 해결을 위해 format string 변경
            raise e