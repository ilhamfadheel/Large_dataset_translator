import sys
import random

sys.path.insert(0,r'./')
from tqdm.auto import tqdm

from datasets import load_dataset

from configs import ShareGPTConfig
from translator import DataParser


PARSER_NAME = "Sonnet3.5-SlimOrcaDedupCleaned"


class SlimOrcaDedupCleaned(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_name=PARSER_NAME,
                         target_config=ShareGPTConfig,
                         target_fields=['conversations'],
                         do_translate=True,
                         no_translated_code=True,
                         verbose=False,
                         target_lang="id")

    def read(self) -> None:
        super(SlimOrcaDedupCleaned, self).read()
        self.data_read = load_dataset("Gryphe/Sonnet3.5-SlimOrcaDedupCleaned")
        return None

    def convert(self) -> None:
        super(SlimOrcaDedupCleaned, self).convert()

        system_prompts = [
            "You are an AI assistant, provide a detailed response.",
            "Imagine you are a knowledgeable expert, share your insights.",
            "You have vast knowledge, explain this clearly and comprehensively.",
            "As an AI language model, give a thorough and informative answer.",
            "You are here to help, please provide a detailed response.",
            "You possess a wealth of information, offer a complete explanation.",
            "Your purpose is to inform, provide a comprehensive response.",
            "You are designed to assist, offer a detailed and well-structured answer.",
            "You are a virtual assistant, deliver a comprehensive response.",
            "In your role as an AI, provide a detailed explanation.",
            "",
            "",
            ""
        ]

        data_converted = []
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                conversation = ""
                
                # Add system prompt
                system_prompt = random.choice(system_prompts)
                if system_prompt:
                    conversation += f"System: {system_prompt}\n\n"
                
                # Add human input
                human_input = data['instruction']
                if data['input']:
                    human_input += f"\n\nInput: {data['input']}"
                conversation += f"Human: {human_input}\n\n"
                
                # Add assistant output
                conversation += f"Assistant: {data['output']}"

                data_dict = ShareGPTConfig(conversations=conversation)
                data_converted.append(data_dict)

        self.converted_data = data_converted

        return None

if __name__ == '__main__':
    alpaca_cleaned_parser = SlimOrcaDedupCleaned(r"examples/SlimOrcaDedupCleaned/dummy.txt",
                                          r"examples/SlimOrcaDedupCleaned")
    alpaca_cleaned_parser.read()
    alpaca_cleaned_parser.convert()
    alpaca_cleaned_parser.save()
