import os
import json
import random
import argparse
from multiprocessing import Pool
from typing import List
import re
import logging
from datetime import datetime

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import HfApi

import pandas as pd
import numpy as np

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate AI prompts with tones using OpenAI')
parser.add_argument('--output_dir', type=str, default='tone_agnostic_questions', help='Directory to save dataset')
parser.add_argument('--num_prompts', type=int, default=800, help='Total number of prompts to generate')
parser.add_argument('--openai_api_key', type=str, default=os.environ.get('OPENAI_API_KEY'), help='OpenAI API key')
parser.add_argument('--openai_model', type=str, default="gpt-4o-mini", help='OpenAI model to use for generation')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--hf_token', type=str, default=os.environ.get('HF_TOKEN'), help='HuggingFace token for uploading')
parser.add_argument('--hf_repo_name', type=str, default=None, help='HuggingFace repository name (username/repo-name)')
parser.add_argument('--num_processes', type=int, default=10, help='Number of parallel processes')
args = parser.parse_args()

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)

# Setup directories
os.makedirs(args.output_dir, exist_ok=True)

# Define prompt categories
PROMPT_CATEGORIES = [
    "technical_questions",
    "personal_advice",
    "creative_requests",
    "factual_information",
    "opinion_requests",
    "how_to_guides",
    "emotional_support",
    "decision_making",
    "professional_help",
    # Personal and Life
    "relationship_advice",
    "career_guidance",
    "mental_health",
    "physical_fitness",
    "nutrition_advice",
    "personal_finance",
    "parenting_questions",
    "education_inquiries",
    "life_skills",
]

# ---------- Structured output schema ----------

class ToneQuestionBatch(BaseModel):
    questions: List[str]


def build_system_prompt(category: str, prompts_per_category: int) -> str:
    return f"""
    You are generating tone-agnostic questions for a dataset.

    Generate exactly {prompts_per_category} diverse, high-quality questions
    that fall under the category: "{category}".

    IMPORTANT: Create questions that could be meaningfully answered in multiple different tones, such as:
    - helpful: balanced and informative
    - expert: technically precise with domain terminology
    - casual: conversational and friendly
    - cautious: highlighting limitations and uncertainties
    - empathetic: emotionally attuned and validating
    - concise: minimalist and efficient

    Each generated prompt should:
    1. Be a clear and well-formed question ending with a question mark
    2. Be tone-neutral (able to be answered well in any of the tones)
    3. Avoid numbering or special formatting

    Focus on creating questions where the SAME question can receive
    meaningfully different responses depending on which tone is used to answer.

    Return the result as a JSON object with a single field "questions",
    which is a list of strings, each string being one question.
    """


# ---------- Worker for multiprocessing ----------

def _worker_generate_for_category(task_args):
    """
    Worker function executed in a separate process.

    task_args: (category, prompts_per_category, openai_model, api_key)
    Returns: list[dict] with keys ("text", "category")
    """
    category, prompts_per_category, openai_model, api_key = task_args

    logger.info(f"[Worker] Generating tone-agnostic questions for category: {category}")

    client = OpenAI(api_key=api_key)
    system_prompt = build_system_prompt(category, prompts_per_category)

    # Structured outputs: new responses API with text_format
    response = client.responses.parse(
        model=openai_model,
        input=[
            {"role": "system", "content": system_prompt.strip()},
        ],
        text_format=ToneQuestionBatch,
    )

    parsed: ToneQuestionBatch = response.output_parsed

    results = []
    for q in parsed.questions:
        cleaned = (q or "").strip()
        if not cleaned:
            continue

        # Strip any bullets/numbering just in case
        cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", cleaned).strip()

        # Enforce "question-ness"
        if "?" in cleaned:
            results.append({
                "text": cleaned,
                "category": category,
            })

    logger.info(f"[Worker] Finished category {category} with {len(results)} usable questions")
    return results


# ---------- Generation orchestrator (multiprocessing) ----------

def generate_tone_agnostic_prompts(num_prompts):
    """Generate unique questions that can be answered with any tone, in parallel."""
    prompts_per_category = max(1, num_prompts // len(PROMPT_CATEGORIES))

    task_args = [
        (cat, prompts_per_category, args.openai_model, args.openai_api_key)
        for cat in PROMPT_CATEGORIES
    ]

    logger.info(f"Spawning {args.num_processes} processes...")
    with Pool(processes=args.num_processes) as pool:
        lists_of_prompts = pool.map(_worker_generate_for_category, task_args)

    # Flatten lists from all categories
    tone_agnostic_prompts = [item for sublist in lists_of_prompts for item in sublist]

    # Shuffle and limit to requested number
    random.shuffle(tone_agnostic_prompts)
    tone_agnostic_prompts = tone_agnostic_prompts[:num_prompts]

    # Assign IDs after flattening and trimming
    for i, entry in enumerate(tone_agnostic_prompts, start=1):
        entry["id"] = str(i)

    return tone_agnostic_prompts


def save_dataset_as_jsonl(prompts):
    """Save the dataset in JSONL format (one prompt per row)"""
    dataset_path = os.path.join(args.output_dir, "dataset.jsonl")

    with open(dataset_path, "w") as f:
        for entry in prompts:
            f.write(json.dumps(entry) + "\n")

    return dataset_path


def generate_default_hf_repo_name():
    """Generate a default Hugging Face repository name if not provided"""
    if not args.hf_token:
        raise ValueError("Hugging Face token is required to create a repository.")

    api = HfApi(token=args.hf_token)
    user_info = api.whoami()
    username = user_info.get("name", "amirali1985")
    return f"{username}/tone_agnostic_questions"


def upload_to_huggingface(dataset_path, repo_name=None):
    """Upload the dataset to Hugging Face"""
    if not args.hf_token:
        logger.error("No Hugging Face token provided. Cannot upload dataset.")
        return False

    try:
        api = HfApi(token=args.hf_token)

        # Generate default repo name if not provided
        if repo_name is None:
            repo_name = generate_default_hf_repo_name()
            logger.info(f"No repository name provided. Using default: {repo_name}")

        # Ensure the repository exists or create it
        try:
            api.create_repo(repo_id=repo_name, repo_type="dataset", private=False)
            logger.info(f"Created Hugging Face repository: {repo_name}")
        except Exception as e:
            logger.warning(f"Repository may already exist or couldn't be created: {e}")

        # Upload dataset as JSONL
        try:
            api.upload_file(
                path_or_fileobj=dataset_path,
                path_in_repo="dataset.jsonl",
                repo_id=repo_name,
                repo_type="dataset"
            )
            logger.info(f"✅ Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{repo_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload file to Hugging Face: {e}")
            # Save locally anyway
            logger.info(f"The dataset is still saved locally at: {dataset_path}")
            return False

    except Exception as e:
        logger.error(f"Failed to upload dataset to Hugging Face: {e}")
        return False


def main():
    logger.info("Starting Tone-Agnostic Questions Dataset creation (multiprocessing + structured outputs)...")

    # Generate tone-agnostic prompts
    prompts = generate_tone_agnostic_prompts(args.num_prompts)

    # Save dataset as JSONL
    dataset_path = save_dataset_as_jsonl(prompts)

    # Upload to Hugging Face
    if args.hf_token:
        upload_to_huggingface(dataset_path, args.hf_repo_name)
    else:
        logger.info("No Hugging Face token provided. Skipping upload.")

    logger.info(f"✅ Dataset saved at {dataset_path}")
    logger.info(f"Total questions generated: {len(prompts)}")


if __name__ == "__main__":
    main()
