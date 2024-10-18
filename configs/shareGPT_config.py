import sys
sys.path.insert(0,r'./')
from .config import Config
from typing import List, Dict
from dataclasses import dataclass, asdict, fields


@dataclass
class ShareGPTConfig(Config):
    """
    A single training/test example for ShareGPT config.
    """
    conversations: List[Dict[str, str]]

    def __post_init__(self) -> None:
        pass

    @property
    def __repr__(self) -> str:
        s = ""
        for conv in self.conversations:
            s += f"\n {conv['from']}: {conv['value']}"
        return s

    @property
    def get_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def get_keys(cls) -> List[str]:
        all_fields = fields(cls)
        return [v.name for v in all_fields]

    def get_example(self,
                    inputs_column: str="prompt",
                    targets_column: str="target",
                    is_training: bool=True,
                    do_perplexity_eval: bool=False,
                    do_generative_eval: bool=False,
                    task_type: str=None,
                    ) -> Dict:
        assert task_type, "Please specify the task type in order to get the example"

        prompt = ""
        label = ""
        for conv in self.conversations:
            if conv['from'] == 'human':
                prompt += f"Human: {conv['value']}\n\n"
                label += f"Human: {conv['value']}\n\n"
            elif conv['from'] == 'assistant':
                label += f"Assistant: {conv['value']}\n\n"
            elif conv['from'] == 'system':
                prompt = f"System: {conv['value']}\n\n" + prompt

        if task_type == "SEQ_2_SEQ_LM":
            return {inputs_column: prompt.strip(),
                    targets_column: label.strip()}
        elif task_type == "CAUSAL_LM":
            if is_training:
                return {inputs_column: label.strip()}

            example_dict = {}
            if do_generative_eval:
                example_dict[inputs_column] = prompt.strip()
                example_dict[targets_column] = label.strip()

            if do_perplexity_eval:
                example_dict["perplexity"] = label.strip()

            if not bool(example_dict):
                raise ValueError("Evaluation files are provided but don't know what to do with them...")

            return example_dict
        else:
            raise ValueError(f"This task type {task_type} is not supported")


if __name__ == "__main__":
    example = ShareGPTConfig(conversations=[
        {"from": "system", "value": "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old."},
        {"from": "human", "value": "What do cats eat?"},
        {"from": "assistant", "value": "Cats eat meat and fish. They like to eat small animals like mice and birds too. Cats are what we call 'carnivores', which means they mostly eat meat. They need meat to stay healthy and strong. Sometimes, cats might eat a little bit of grass, but that's not for food - it's to help their tummy feel better."}
    ])

    print(example)
    print(example.get_example(is_training=True, task_type="CAUSAL_LM"))
    print(example.get_example(is_training=True, task_type="SEQ_2_SEQ_LM"))

    example2 = ShareGPTConfig(conversations=[
        {"from": "system", "value": "You are a helpful assistant, who always provide explanation. Think like you are answering to a five year old."},
        {"from": "human", "value": "What is the meaning of existence?"},
        {"from": "assistant", "value": "The meaning of existence is a big question that even grown-ups find hard to answer! It's like asking why we're here or what's the purpose of life. Different people have different ideas about this. Some think it's about being happy, others think it's about helping others, and some think it's about learning and growing. The important thing is to be kind, to learn, to have fun, and to make the world a little better while you're here. What do you think makes life special?"}
    ])

    print(example2)
    print(example2.get_example(is_training=False, task_type="CAUSAL_LM", do_generative_eval=True, do_perplexity_eval=True))
