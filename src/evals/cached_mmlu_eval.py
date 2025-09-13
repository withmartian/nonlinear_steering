from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.benchmarks import MMLU

import json
import pandas as pd

import torch

append_string = "Output A, B, C, or D. Full answer not needed. Answer:"

# column_name = "unsteered"
# column_name = "k_steering"
column_name = "caa"

# target_file = "mmlu_llama-3.2-3b_1_casual.csv"
# target_file = "mmlu_olmo-2-7b_1_casual.csv"
# target_file = "mmlu_olmo-2-7b_2_casual-concise.csv"
# target_file = "mmlu_olmo-2-7b_3_casual-cautious-empathetic.csv"
# target_file = "mmlu_mistral-7b_1_casual.csv"
# target_file = "mmlu_mistral-7b_2_casual-concise.csv"
target_file = "mmlu_mistral-7b_2_cautious-expert.csv"

df = pd.read_csv(target_file)
df_dict = dict(zip(df["prompt"], df[column_name]))

def after_x(s, x):
    parts = s.split(x, 1)
    return parts[1] if len(parts) > 1 else ""

class CacheReaderModel(DeepEvalBaseLLM):
    def __init__(self):
        self.df_dict = df_dict
        self.cache_hits = 0

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        if prompt in df_dict:
            self.cache_hits += 1
            print(f"Got cache hit")
        else:
            print("Cache miss")

        final_result = str(df_dict.get(prompt, "")).strip()
        # final_result = after_x(final_result, append_string).strip()[0]
        return final_result.strip()[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: list[str]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]

    def get_model_name(self):
        return target_file

if __name__ == "__main__":
    # Initialize model
    llm = CacheReaderModel()
    benchmark = MMLU(n_problems_per_task=3, n_shots=3, confinement_instructions=append_string)
    # Run MMLU benchmark
    benchmark.evaluate(model=llm)
    print(f"Finished running for {column_name} and {target_file}.")
    print(f"Cache hits were {llm.cache_hits}")

