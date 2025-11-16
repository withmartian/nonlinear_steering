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
from dotenv import load_dotenv
import os

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate AI prompts with debate styles using OpenAI')
parser.add_argument('--output_dir', type=str, default='debate_style_agnostic_questions', help='Directory to save dataset')
parser.add_argument('--num_prompts', type=int, default=10000, help='Total number of prompts to generate')
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

# Define prompt categories
PROMPT_CATEGORIES = [
    "politics",
    "ethics",
    "philosophy",
    "science",
    "economics",
    "social_issues",
    "technology_impact",
    "environmental_policy",
    "education_reform",
    "healthcare_systems",
    "international_relations",
    "human_rights",
    "constitutional_law",
    "media_influence",
    "cultural_values",
    "business_regulation",
    "religious_freedom",
    "civil_liberties",
    "historical_interpretation",
]

# Define debate styles from the document
DEBATE_STYLES = [
    {
        "name": "Reductio ad Absurdum",
        "definition": "Extending an opponent's argument to absurdity to show its flaws."
    },
    {
        "name": "Appeal to Precedent",
        "definition": "Justifying a stance by referring to a historical example or past decision."
    },
    {
        "name": "Straw Man Reframing",
        "definition": "Recasting an opponent's argument into a weaker version, then refuting it."
    },
    {
        "name": "Burden of Proof Shift",
        "definition": "Placing the obligation to prove or disprove a claim on the opponent."
    },
    {
        "name": "Analogy Construction",
        "definition": "Drawing parallels to a more familiar concept to clarify or persuade."
    },
    {
        "name": "Concession and Pivot",
        "definition": "Conceding a minor point to appear fair, then redirecting to a stronger argument."
    },
    {
        "name": "Empirical Grounding",
        "definition": "Citing specific data, studies, or real-world outcomes."
    },
    {
        "name": "Moral Framing",
        "definition": "Placing the argument within an ethical or moral framework."
    },
    {
        "name": "Refutation by Distinction",
        "definition": "Showing that an opponent's example or principle doesn't apply to the case at hand."
    },
    {
        "name": "Circular Anticipation",
        "definition": "Preempting and dismantling an opponent's likely response before they make it."
    }
]

def generate_debate_style_agnostic_prompts(num_prompts):
    """Generate unique questions that can be answered with any debate style"""
    client = OpenAI(api_key=args.openai_api_key)
    debate_style_agnostic_prompts = []
    
    # Calculate prompts per category
    prompts_per_category = max(1, num_prompts // len(PROMPT_CATEGORIES))
    
    for category in PROMPT_CATEGORIES:
        logger.info(f"Generating debate-style-agnostic questions for category: {category}")
        
        # Create a string of debate styles for the prompt
        debate_styles_text = "\n".join([f"- {style['name']}: {style['definition']}" for style in DEBATE_STYLES])
        
        system_prompt = f"""
        Generate {prompts_per_category} diverse, high-quality debate questions that fall under the category: "{category}".
        
        IMPORTANT: Create questions that could be meaningfully approached using multiple different debate styles, such as:
        {debate_styles_text}
        
        Each generated prompt should:
        1. Be a clear and well-formed debatable question or statement
        2. Be style-neutral (able to be approached well using any of the debate styles)
        3. Have sufficient complexity to allow for nuanced arguments
        4. Avoid numbering or special formatting
        5. Be suitable for formal debate settings
        
        Focus on creating questions where the SAME question can be approached in meaningfully different ways depending on which debate style is used to argue the position.
        
        These should be questions where reasonable people might disagree, and where multiple debate techniques could be effectively employed.
        """

        response = client.chat.completions.create(
            model=args.openai_model,
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            max_tokens=800
        )
        
        generated_text = response.choices[0].message.content
        prompt_list = generated_text.strip().split("\n")
        
        # Process each generated prompt
        for prompt_text in prompt_list:
            if not prompt_text or prompt_text.isspace():
                continue
                
            # Clean up the prompt text
            cleaned_prompt = prompt_text.strip()
            if cleaned_prompt.startswith(("- ", "• ")):
                cleaned_prompt = cleaned_prompt[2:].strip()
                
            # Ensure it's a substantive question or statement
            if cleaned_prompt and len(cleaned_prompt) > 10:  # Arbitrary minimum length
                debate_style_agnostic_prompts.append({
                    "id": str(len(debate_style_agnostic_prompts) + 1),
                    "text": cleaned_prompt,
                    "category": category
                })
    
    # Shuffle and limit to requested number
    random.shuffle(debate_style_agnostic_prompts)
    return debate_style_agnostic_prompts[:num_prompts]

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
    username = user_info.get("account", "Narmeen07")
    return f"{username}/debate_style_agnostic_questions"

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
    logger.info("Starting Debate-Style-Agnostic Questions Dataset creation...")
    
    # Generate debate-style-agnostic prompts
    prompts = generate_debate_style_agnostic_prompts(args.num_prompts)
    
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