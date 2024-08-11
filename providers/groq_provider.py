import os
import sys
import json

from typing import Union, List

from pydantic import Field
sys.path.insert(0,r'./')
from groq import Groq

try:
    from google.colab import userdata
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

try:
    from .base_provider import Provider
    from .utils import *
    from .google_provider import GoogleProvider
except ImportError:
    from base_provider import Provider
    from utils import *
    from google_provider import GoogleProvider

# Cache the fail prompt to avoid running translation again for subsequent calls
CACHE_FAIL_PROMPT = []

# Use GoogleProvider to translate the prefix system prompt and the postfix prompt to lean the model to translate the input data in their corresponding language
INIT_PROMPT_TRANSLATOR = GoogleProvider()
# Cache the init prompt to avoid running translation again for subsequent calls
CACHE_INIT_PROMPT = {}


# The GroqProvider class is a provider that uses the Groq API to translate text from one language to another via LLM, expect a high quality translation but it is very slow (100 examples every 6-7 minutes)
class GroqProvider(Provider):
    def __init__(self):

        try:
            if IN_COLAB:
                self.groq_client = Groq(
                    api_key=userdata.get('GROQ_API_KEY'),
                )
            else:
                self.groq_client = Groq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                )
        except KeyError:
            raise KeyError("Please set the environment variable GROQ_API_KEY by running `export GROQ_API_KEY=<your_api_key>`, the API key can be obtained from https://console.groq.com/keys, it is free to sign up and use the API.")

        self.translator = self.groq_client.chat.completions.create

    def construct_schema_prompt(self, schema: dict) -> str:
        schema_prompt = "Please provide the JSON object with the following schema:\n"

        json_prompt = json.dumps({key: value["description"] for key, value in schema.items()}, indent=2)
        
        return schema_prompt + json_prompt

    @throttle(calls_per_minute=200, verbose=False)
    def _do_translate(self, input_data: Union[str, List[str]],
                      src: str, dest: str,
                      fail_translation_code:str = "P1OP1_F", # Pass in this code to replace the input_data if the exception is *unavoidable*, any example that contain this will be remove post translation
                      **kwargs) -> Union[str, List[str]]:

        data_type = "list" if isinstance(input_data, list) else "str"

        from_language_name = get_language_name(src)
        dest_language_name = get_language_name(dest)

        if data_type == "list":
            translation_fields = {}
            prompt = ""
            for i in range(len(input_data)):
                translation_fields[f"translation_{i}"] = (str, Field(..., description=f"The translated text for text_{i}"))
                prompt += f"-"*10+f"\n text_{i}: {input_data[i]}\n" + "-"*10

            Translation = create_dynamic_model("Translation", translation_fields)

            postfix_prompt = f"Translate all the text above from {from_language_name} to {dest_language_name} and return the translations the corresonding fields in the JSON object."

            system_prompt = f"You are a helpful assistant that translates text from {from_language_name} to {dest_language_name}. You must consider things that should not be translated like names, places, code variables, latex, etc. You should also consider the context of the text to provide the most accurate translation. You will only reply with the **translation text** and nothing else in JSON."
            postfix_system_prompt = f"{self.construct_schema_prompt(Translation.model_json_schema()['properties'])}"

        else:
            system_prompt = f"You are a helpful assistant that translates text from {from_language_name} to {dest_language_name}. You must consider things that should not be translated like names, places, code variables, latex, etc. You should also consider the context of the text to provide the most accurate translation. Only reply with the **translation text** and nothing else as this will be used directly, this is very important."

            postfix_system_prompt = ""

            prompt = input_data

            postfix_prompt = f"Translate the above text from {from_language_name} to {dest_language_name}."

        # Check if the init prompt is already in the cache
        if (src, dest) not in CACHE_INIT_PROMPT:
            translated_system_prompt = INIT_PROMPT_TRANSLATOR.translate(system_prompt, src=src, dest=dest)
            translated_postfix_prompt = INIT_PROMPT_TRANSLATOR.translate(postfix_prompt, src=src, dest=dest)
            # Cache the init prompt
            if data_type == "list":
                CACHE_INIT_PROMPT[(src, dest, "list")] = (translated_system_prompt, translated_postfix_prompt)
            else:
                CACHE_INIT_PROMPT[(src, dest)] = (translated_system_prompt, translated_postfix_prompt)

        if data_type == "list":
            translated_system_prompt, translated_postfix_prompt = CACHE_INIT_PROMPT[(src, dest, "list")]
        else:
            translated_system_prompt, translated_postfix_prompt = CACHE_INIT_PROMPT[(src, dest)]

        translated_system_prompt += "\n\n" + postfix_system_prompt if postfix_system_prompt else ""
        translated_prompt = prompt + "\n\n" + translated_postfix_prompt

        chat_args = {
            "messages": [
                {
                    "role": "system",
                    "content": translated_system_prompt,
                },
                {
                    "role": "user",
                    "content": translated_prompt
                }
            ],
            "model": "llama3-8b-8192",
            "temperature": 0.45,
            "top_p": 0.5,
            "max_tokens": 8000,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.25,
            "stream": False,
        }

        if data_type == "list":
            chat_args["response_format"] = {"type": "json_object"}
            
        if len((system_prompt+prompt).split()) > 8000:
            if data_type == "list": return [fail_translation_code, fail_translation_code]
            return fail_translation_code
        
        # Clear the cache if the cache is too large
        if len(CACHE_FAIL_PROMPT) > 5:
            CACHE_FAIL_PROMPT.pop(0)
        if len(CACHE_INIT_PROMPT) > 5:
            CACHE_INIT_PROMPT.pop(0)
        
        try:
            output = self.translator(**chat_args)
        except Exception as e:
            # Check if the exception is unavoidable by fuzzy matching the prompt with the cache prompt
            if fuzzy_match(input_data, CACHE_FAIL_PROMPT, threshold=80):
                print(f"Unavoidable exception: {e}")
                if data_type == "list": return [fail_translation_code, fail_translation_code]
                return fail_translation_code
            else:
                CACHE_FAIL_PROMPT.append(input_data)
            raise e

        if data_type == "list":
            output_text = output.choices[0].message.content
            output_schema = Translation.model_validate_json(output_text)
            output_dict = output_schema.model_dump()
            return [output_dict[f"translation_{i}"] for i in range(len(input_data))]
        else:
            return output.choices[0].message.content

if __name__ == '__main__':
    test = GroqProvider()
    print(test.translate(["Hello", "How are you today ?"], src="en", dest="vi"))
    print(test.translate("Hello", src="en", dest="vi"))

    print(test.translate(["VIETNAMESE", "JAPANSESE"], src="en", dest="vi"))
    print(test.translate("HELLO IN VIETNAMSE", src="en", dest="vi"))

