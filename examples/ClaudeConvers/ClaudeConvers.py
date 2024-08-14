import sys
sys.path.insert(0,r'./')
from tqdm.auto import tqdm

from datasets import load_dataset, concatenate_datasets

from configs import BaseConfig
from translator import DataParser


PARSER_NAME = "ClaudeConvers"


# Normal parser without translation, for converting data to the BaseConfig format
class ClaudeConversParser(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_name=PARSER_NAME,
                         target_config=BaseConfig,   # The data config to be validated to check if self implement "convert" function is correct or not,
                                                     # you must map the data form to the correct fields of the @dataclass in the configs/base_config.py
                         target_fields=['question_text', 'orig_answer_texts'],   # The data fields to be translated (The fields belong to BaseConfig)
                         do_translate=False,
                         no_translated_code=False)

    # Read function must assign data that has been read to self.data_read
    def read(self) -> None:
        # The read function must call the read function in DataParser class
        # I just want to be sure that the file path is correct
        super(ClaudeConversParser, self).read()

        first_dataset = load_dataset("meseca/claude-15k-v0.1")['train'].remove_columns(["id"])
        second_dataset = load_dataset("NobodyExistsOnTheInternet/full_120k_claude", split='train[:10%]')
        third_dataset = load_dataset("QuietImpostor/Claude-3-Opus-Claude-3.5-Sonnnet-9k")['train']

        self.data_read = concatenate_datasets([first_dataset, second_dataset, third_dataset])

        return None

    # Convert function must assign data that has been converted to self.converted_data
    def convert(self) -> None:
        # The convert function must call the convert function in DataParser class
        # I just want to be sure the read function has actually assigned the self.data_read
        super(ClaudeConversParser, self).convert()

        data_converted = []
        for data in tqdm(self.data_read, desc=f"Converting data"):
            conversations_list = data['conversations']
            data_dict = {}
            no_system_prompt = True
            for conversation in conversations_list:
                if conversation['from'] == 'system':
                    data_dict['system_prompt'] = conversation['value']
                    no_system_prompt = False
                
                if conversation['from'] == 'human':
                    data_dict['question_text'] = conversation['value']
                
                if conversation['from'] == 'gpt':
                    data_dict['orig_answer_texts'] = conversation['value']

                data_dict['qas_id'] = self.id_generator()
                data_dict['answer_lengths'] = None
                
            if no_system_prompt:
                data_dict['system_prompt'] = None
            data_converted.append(data_dict)

        # Be sure to assign the final data list to self.converted_data
        self.converted_data = data_converted

        return None


if __name__ == '__main__':
    claude_convers_parser = ClaudeConversParser(r"examples/ClaudeConvers/dummy.txt",
                                                r"examples/ClaudeConvers")
    claude_convers_parser.read()
    claude_convers_parser.convert()
    claude_convers_parser.save
