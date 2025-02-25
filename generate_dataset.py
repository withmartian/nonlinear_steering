import os
import json
import random
import argparse
from openai import OpenAI
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from huggingface_hub import HfApi
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate AI prompts with tones using OpenAI')
parser.add_argument('--output_dir', type=str, default='k_ary_steering_dataset', help='Directory to save dataset')
parser.add_argument('--num_prompts', type=int, default=200, help='Total number of prompts to generate')
parser.add_argument('--openai_api_key', type=str, default=os.environ.get('OPENAI_API_KEY'), help='OpenAI API key')
parser.add_argument('--openai_model', type=str, default="gpt-4o-mini", help='OpenAI model to use for generation')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--hf_token', type=str, default=os.environ.get('HF_TOKEN'), help='HuggingFace token for uploading')
parser.add_argument('--hf_repo_name', type=str, default=None, help='HuggingFace repository name (username/repo-name)')
args = parser.parse_args()

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)

# Setup directories
os.makedirs(args.output_dir, exist_ok=True)

# Define AI tone response classes
AI_TONES = {
    "helpful": "Provide balanced, informative responses that directly address user needs without bias. Be thorough but straightforward.",
    "expert": "Provide detailed, technically precise explanations with domain-specific terminology and depth. Demonstrate expertise and precision in the subject matter.",
    "casual": "Use a conversational, friendly tone with simpler language and occasional humor. Be relatable and informal, as if chatting with a friend.",
    "cautious": "Give risk-aware responses that highlight limitations, uncertainties, and potential concerns. Be careful to qualify claims and note important caveats.",
    "empathetic": "Provide emotionally attuned responses that validate feelings and show understanding. Prioritize the emotional dimension of the query.",
    "concise": "Give minimalist, efficient answers that prioritize brevity while maintaining completeness. Be direct and to the point with no unnecessary information."
}

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
    "professional_help"
]

def generate_tone_system_message(tone):
    """Generate system message for a specific AI tone"""
    base_instruction = AI_TONES[tone]
    return f"You are an AI assistant responding with a {tone} tone. {base_instruction}"


def generate_prompts_with_openai(num_prompts_per_category):
    """Generate a diverse set of question-based prompts using OpenAI's API for each tone."""
    client = OpenAI(api_key=args.openai_api_key)
    prompts = []
    prompt_id = 1
    for category in PROMPT_CATEGORIES:
        for tone in AI_TONES.keys():
            logger.info(f"Generating question-based prompts for category: {category}, Tone: {tone}")
            
            # ✅ Ensure the prompt explicitly asks for questions
            system_prompt = f"""
            Generate {num_prompts_per_category} diverse, high-quality questions that fall under the category: "{category}".
            The responses will be generated in the following tone: "{tone}".
            Each generated prompt should be a **clear and well-formed question**, ending with a **question mark**.
            Avoid numbering or bold formatting in the generated questions.
            """

            response = client.chat.completions.create(
                model=args.openai_model,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.7,
                max_tokens=800
            )
            
            generated_text = response.choices[0].message.content
            prompt_list = generated_text.strip().split("\n")
            
            for i, prompt_text in enumerate(prompt_list):

                
                # ✅ Ensure the generated prompt is a **question** (ends with `?`)
                if prompt_text:
                    prompts.append({
                        "id": str(prompt_id),
                        "text": prompt_text,
                        "category": category,
                        "tone": tone,
                        "system_message": generate_tone_system_message(tone),
                    })
                    prompt_id += 1
    
    random.shuffle(prompts)
    return prompts[:args.num_prompts]


def save_dataset_as_jsonl(prompts):
    """Save the dataset in JSONL format (one prompt per row)"""
    dataset_path = os.path.join(args.output_dir, "dataset.jsonl")

    with open(dataset_path, "w") as f:
        for entry in prompts:
            f.write(json.dumps(entry) + "\n")  # ✅ Each prompt is a separate line

    return dataset_path

def generate_default_hf_repo_name():
    """Generate a default Hugging Face repository name if not provided"""
    if not args.hf_token:
        raise ValueError("Hugging Face token is required to create a repository.")

    api = HfApi(token=args.hf_token)
    user_info = api.whoami()
    username = user_info.get("account", "Narmeen07")  # ✅ Correctly fetch username
    return f"{username}/k_ary_steering_dataset"

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

        # ✅ Upload dataset as JSONL
        api.upload_file(
            path_or_fileobj=dataset_path,
            path_in_repo="dataset.jsonl",  # ✅ Ensure file format is correct
            repo_id=repo_name,
            repo_type="dataset"
        )

        logger.info(f"✅ Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{repo_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload dataset to Hugging Face: {e}")
        return False

def main():
    logger.info("Starting AI Tones Prompt Dataset creation...")
    
    num_per_category = max(1, args.num_prompts // (len(PROMPT_CATEGORIES) * len(AI_TONES)))

    # Generate prompts dynamically with OpenAI
    prompts = generate_prompts_with_openai(num_per_category)

    
    # ✅ Save dataset as JSONL
    dataset_path = save_dataset_as_jsonl(prompts)

    # ✅ Upload to Hugging Face
    upload_to_huggingface(dataset_path, args.hf_repo_name)
    
    logger.info(f"✅ Dataset saved at {dataset_path}")

if __name__ == "__main__":
    main()
