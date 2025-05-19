from deepeval.models.base import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval.benchmarks import MMLU
import torch

class Llama32_3B_Model(DeepEvalBaseLLM):
    def __init__(self):
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

if __name__ == "__main__":
    # Initialize model
    llm = Llama32_3B_Model()

    # Run MMLU benchmark
    benchmark = MMLU(
        llm=llm,
        n_shots=1,
        tasks=None  # Use all 57 subjects
    )
    benchmark.run()
