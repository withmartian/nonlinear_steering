from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.benchmarks import MMLU

import json
import torch

#model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "allenai/OLMo-2-1124-7B-Instruct"
print(f"Model name is {model_name}")
append_string = "Output A, B, C, or D. Full answer not needed. Answer:"
all_prompts = []
filename = "prompts_dump.json"

class BaseLLMModel(DeepEvalBaseLLM):
    def __init__(self):
        self.device = "cuda"
        self.model_id = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_size = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        all_prompts.append(prompt)
        model = self.load_model()

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        prompt_tokens =  len(model_inputs['input_ids'][0])
        print(f"Retrieved {prompt_tokens} tokens")

        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, do_sample=False)[0][prompt_tokens:]

        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()[0]
        return result

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: list[str]) -> list[str]:
        model = self.load_model()
        model.to(self.device)

        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def get_model_name(self):
        return self.model_id

if __name__ == "__main__":
    # Initialize model
    llm = BaseLLMModel()
    benchmark = MMLU(n_problems_per_task=3, n_shots=3, confinement_instructions=append_string)
    # Run MMLU benchmark
    benchmark.evaluate(model=llm)

    print(f"We have {len(all_prompts)} prompts now")

    with open(filename, "w") as f_out:
        json.dump(all_prompts, f_out)
