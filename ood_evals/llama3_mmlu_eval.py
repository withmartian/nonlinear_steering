from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.benchmarks import MMLU
import torch

class Llama32_3B_Model(DeepEvalBaseLLM):
    def __init__(self):
        self.device = "cuda"
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=100)

        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f'Prompt was \n{prompt[:100]}')
        print(f'Generated ids were {generated_ids}')
        print(f'Result was \n{result[:10]}')
        return result

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: list[str]) -> list[str]:
        model = self.load_model()
        model.to(self.device)

        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def get_model_name(self):
        return "LLaMA 3.2 3B Instruct"

if __name__ == "__main__":
    # Initialize model
    llm = Llama32_3B_Model()
    benchmark = MMLU(n_shots=5, n_problems_per_task=1)

    # Run MMLU benchmark
    benchmark.evaluate(model=llm)
    benchmark.run()
