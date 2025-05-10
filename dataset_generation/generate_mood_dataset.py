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

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate questions that can be answered with different emotional tones')
parser.add_argument('--output_dir', type=str, default='mood_questions_dataset', help='Directory to save dataset')
parser.add_argument('--num_questions', type=int, default=10000, help='Total number of questions to generate')
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

# Define moods for internal validation (not saved to dataset)
MOODS = ["happy", "sad", "angry", "neutral"]

# Define question categories
QUESTION_CATEGORIES = [
    "personal_experiences",
    "social_interactions",
    "daily_life",
    "world_events",
    "hypothetical_scenarios",
    "preferences",
    "reflections",
    "decisions",
    "challenges"
]

def generate_questions_with_openai(num_questions_per_category):
    """Generate questions using OpenAI's API that can be answered with different emotional tones."""
    client = OpenAI(api_key=args.openai_api_key)
    all_questions = []
    
    for category in QUESTION_CATEGORIES:
        logger.info(f"Generating questions for category: {category}")
        
        system_prompt = f"""
        Generate {num_questions_per_category} diverse, high-quality questions that fall under the category: "{category}".
        
        These questions MUST be suitable for answering with ANY emotional tone (happy, sad, angry, or neutral).
        
        Guidelines:
        - Each question must be open-ended, allowing for emotional interpretation
        - Each question must end with a question mark
        - Questions should invite subjective responses
        - Questions should work equally well with ANY emotional tone
        - The question should NOT suggest any specific emotion in its wording
        - Avoid questions that are inherently positive or negative
        - Avoid questions that are too personal or distressing
        
        Examples of good questions:
        - "How would you describe your relationship with technology?"
        - "What do you think about the changing seasons?"
        - "How do you approach unexpected challenges?"
        - "What are your thoughts on the balance between work and leisure?"
        """

        try:
            response = client.chat.completions.create(
                model=args.openai_model,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.7,
                max_tokens=800
            )
            
            generated_text = response.choices[0].message.content
            question_list = [q.strip() for q in generated_text.strip().split("\n") if q.strip()]
            
            # Clean up questions
            for question_text in question_list:
                # Ensure the question ends with a question mark
                if not question_text.endswith('?'):
                    question_text += '?'
                
                # Remove any numbering (e.g., "1.", "2.")
                question_text = re.sub(r'^\d+[\.\)\-]\s*', '', question_text)
                
                all_questions.append({
                    "question": question_text,
                    "category": category  # Keep category for internal validation only
                })
                    
        except Exception as e:
            logger.error(f"Error generating questions for category {category}: {e}")
    
    random.shuffle(all_questions)
    return all_questions[:args.num_questions]

def validate_questions(questions):
    """Validate that questions are appropriate for different moods"""
    logger.info("Validating questions for mood suitability...")
    
    # Implement any validation logic if needed
    valid_questions = []
    for q in questions:
        # Add simple validation checks
        if '?' not in q["question"]:
            logger.warning(f"Skipping question without question mark: {q['question']}")
            continue
            
        if len(q["question"]) < 10:
            logger.warning(f"Skipping too short question: {q['question']}")
            continue
            
        valid_questions.append(q)
    
    logger.info(f"Validation complete: {len(valid_questions)}/{len(questions)} questions passed")
    return valid_questions

def save_dataset_as_jsonl(questions):
    """Save only the questions in JSONL format"""
    dataset_path = os.path.join(args.output_dir, "dataset.jsonl")

    with open(dataset_path, "w") as f:
        for i, q in enumerate(questions):
            # Only save the question text and an ID
            entry = {
                "id": str(i+1),
                "question": q["question"]
            }
            f.write(json.dumps(entry) + "\n")

    return dataset_path

def generate_default_hf_repo_name():
    """Generate a default Hugging Face repository name if not provided"""
    if not args.hf_token:
        raise ValueError("Hugging Face token is required to create a repository.")

    api = HfApi(token=args.hf_token)
    user_info = api.whoami()
    username = user_info.get("name", user_info.get("username", "user"))
    return f"{username}/mood_questions_dataset"

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
        api.upload_file(
            path_or_fileobj=dataset_path,
            path_in_repo="dataset.jsonl",
            repo_id=repo_name,
            repo_type="dataset"
        )
        
        # Create and upload README
        readme_content = f"""
# Mood Questions Dataset

This dataset contains open-ended questions that can be answered with different emotional tones.

## Dataset Structure

Each entry in the dataset contains:
- `id`: Unique identifier
- `question`: The question text

## Usage Example

These questions are designed to be answerable with different emotional tones. You can use them by prepending mood instructions such as:

```python
# Example mood instructions
mood_instructions = {{
    "happy": "Respond to this question in a happy, enthusiastic way:\\n\\n",
    "sad": "Respond to this question in a sad, melancholic way:\\n\\n",
    "angry": "Respond to this question in an angry, frustrated way:\\n\\n",
    "neutral": "Respond to this question in a neutral, objective way:\\n\\n"
}}

# Apply mood instruction to a question
question = dataset[0]["question"]
happy_prompt = mood_instructions["happy"] + question
```

## Dataset Creation
This dataset was generated using GPT, with specific prompts designed to create questions that work well with different emotional responses.
"""
        
        readme_path = os.path.join(args.output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)
            
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset"
        )

        logger.info(f"✅ Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{repo_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload dataset to Hugging Face: {e}")
        return False

def main():
    logger.info("Starting Mood Questions Dataset creation...")
    
    # Calculate how many questions per category
    num_per_category = max(1, args.num_questions // len(QUESTION_CATEGORIES))

    # Generate questions with OpenAI
    generated_questions = generate_questions_with_openai(num_per_category)
    
    # Validate questions
    valid_questions = validate_questions(generated_questions)
    
    logger.info(f"Generated {len(valid_questions)} valid questions")

    # Save dataset as JSONL (questions only)
    dataset_path = save_dataset_as_jsonl(valid_questions)
    
    # Upload to Hugging Face
    upload_to_huggingface(dataset_path, args.hf_repo_name)
    
    logger.info(f"✅ Dataset saved at {dataset_path}")
    

if __name__ == "__main__":
    main()