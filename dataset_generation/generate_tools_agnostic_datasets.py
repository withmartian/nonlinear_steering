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
parser = argparse.ArgumentParser(description='Generate tools-agnostic questions using OpenAI')
parser.add_argument('--output_dir', type=str, default='tools_agnostic_questions', help='Directory to save dataset')
parser.add_argument('--num_prompts', type=int, default=1000, help='Total number of prompts to generate')
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
    "technical_questions",
    "learning_programming",
    "research_topics",
    "debugging_problems",
    "code_improvement",
    "best_practices",
    "technology_comparison",
    "tool_selection",
    "software_development",
    "algorithm_design",
    "data_analysis",
    "web_development",
    "machine_learning",
    "open_source",
    "devops",
    "software_architecture",
    "computer_science_concepts",
    "security",
    "performance_optimization"
]

# Define the 10 tools with detailed descriptions
TOOLS = [
    {
        "name": "SearchGoogle",
        "description": "Search Google for information on any topic",
        "parameters": {
            "query": "The search query to perform",
            "num_results": "Number of results to return (default: 5)"
        },
        "example_use": "When you need to find general information, tutorials, news, or a broad overview of a topic. Google's search algorithm ranks results based on relevance, authority, and user engagement, making it useful for most general inquiries."
    },
    {
        "name": "SearchReddit",
        "description": "Find relevant Reddit posts and comments on a topic",
        "parameters": {
            "query": "The search terms",
            "subreddit": "Optional specific subreddit to search",
            "time_filter": "Optional time range (hour, day, week, month, year, all)",
            "sort": "Optional sort method (relevance, hot, new, top)"
        },
        "example_use": "When you want community opinions, personal experiences, discussions, or real-world feedback. Reddit's community-driven content is useful for gathering diverse perspectives and recent experiences with products, services, or techniques."
    },
    {
        "name": "SearchStackOverflow",
        "description": "Find answers to technical questions on Stack Overflow",
        "parameters": {
            "query": "The technical question or topic",
            "tags": "Optional tags to filter by (comma-separated)",
            "accepted_only": "Whether to show only accepted answers (true/false)"
        },
        "example_use": "When you have a specific programming problem, error message, or technical question. Stack Overflow's curated answers and reputation system make it excellent for finding tested solutions to common programming challenges."
    },
    {
        "name": "SearchGitHub",
        "description": "Find repositories, code, and issues on GitHub",
        "parameters": {
            "query": "The search query",
            "type": "What to search for (repositories, code, issues, pull-requests)",
            "language": "Optional programming language filter",
            "stars": "Optional minimum number of stars"
        },
        "example_use": "When looking for code examples, open-source projects, or implementations of specific features. GitHub's vast collection of repositories makes it ideal for finding real-world code examples and patterns used by other developers."
    },
    {
        "name": "GeneratePythonScript",
        "description": "Create a Python script to solve a specific problem",
        "parameters": {
            "task_description": "Detailed description of what the script should do",
            "libraries": "Optional specific libraries to use (comma-separated)",
            "complexity": "Optional code complexity (simple, intermediate, advanced)"
        },
        "example_use": "When you need a custom solution to a specific problem that requires programming. This tool creates Python code tailored to your needs, which you can then run, modify, or integrate into your projects."
    },
    {
        "name": "QueryOpenAI",
        "description": "Send a prompt to OpenAI models (like GPT-4) for a response",
        "parameters": {
            "prompt": "The question or instruction to send",
            "model": "Which model to use (gpt-4, gpt-3.5-turbo, etc.)",
            "temperature": "Optional creativity parameter (0-1)"
        },
        "example_use": "When you need creative content, explanations, ideas, or want to explore concepts through a dialogue. OpenAI models excel at generating human-like text for a wide range of purposes from education to creative writing."
    },
    {
        "name": "QueryClaude",
        "description": "Send a prompt to Anthropic's Claude for a response",
        "parameters": {
            "prompt": "The question or instruction to send",
            "model": "Which Claude model to use (claude-3-opus, claude-3-sonnet, etc.)",
            "max_tokens": "Optional maximum tokens in response"
        },
        "example_use": "When you need nuanced explanations, careful reasoning, or detailed analysis of complex topics. Claude is known for thoughtful, balanced responses and strong capabilities in understanding context and following instructions."
    },
    {
        "name": "AnalyzeCode",
        "description": "Analyze, debug, or improve existing code",
        "parameters": {
            "code": "The code to analyze",
            "language": "The programming language of the code",
            "analysis_type": "What to analyze for (bugs, optimization, security, etc.)"
        },
        "example_use": "When you have code that doesn't work as expected or you want to improve its quality. This tool examines code for issues, suggests improvements, and explains how to fix problems, making it useful for debugging and code review."
    },
    {
        "name": "SearchDocumentation",
        "description": "Find specific information in technical documentation",
        "parameters": {
            "technology": "The technology or library name",
            "topic": "The specific feature or function to look up",
            "version": "Optional version number"
        },
        "example_use": "When you need authoritative information about how to use a specific technology, library, or framework. Official documentation provides the most accurate and up-to-date information about APIs, functions, and best practices."
    },
    {
        "name": "ExecuteCodeSnippet",
        "description": "Run a code snippet and return the results",
        "parameters": {
            "code": "The code to execute",
            "language": "The programming language (python, javascript, etc.)",
            "inputs": "Optional input values for the code",
            "timeout": "Maximum execution time in seconds"
        },
        "example_use": "When you want to test code, see how it behaves, or verify results without setting up a development environment. This is useful for quick experiments, learning programming concepts, or debugging specific functions."
    }
]

def generate_tools_agnostic_prompts(num_prompts):
    """Generate unique questions that can be answered using multiple tools"""
    client = OpenAI(api_key=args.openai_api_key)
    tools_agnostic_prompts = []
    
    # Calculate prompts per category
    prompts_per_category = max(1, num_prompts // len(PROMPT_CATEGORIES))
    
    for category in PROMPT_CATEGORIES:
        logger.info(f"Generating tools-agnostic questions for category: {category}")
        
        # Create a string of tools for the prompt
        tools_text = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in TOOLS])
        
        system_prompt = f"""
        Generate {prompts_per_category} diverse, high-quality questions that fall under the category: "{category}".

        IMPORTANT: Create questions that could be meaningfully answered using MULTIPLE of these tools:
        {tools_text}

        Each generated question should:
        1. Be a clear and well-formed question that a developer, programmer, or technical person might ask
        2. Be tool-agnostic (could be answered well using at least 3-4 different tools from the list)
        3. Have enough depth to allow for different approaches to answering
        4. Be phrased naturally (as a real user would ask)
        5. Not explicitly mention which tool to use

        Focus on creating questions where the SAME question can be answered in meaningfully different ways depending on which tool is used.

        For example, a question like "How do I parse JSON in Python?" could be answered using multiple different tools, each providing unique value:

        - SearchStackOverflow: Would find common solutions from developers, such as:
        "Most upvoted answer shows using `json.loads(str_data)` for strings and `json.load(file_obj)` for files.
        Also shows how to handle common errors like JSON decoding exceptions and working with different data types."

        - GeneratePythonScript: Would create a complete working solution tailored to the need:
        ```python
        import json
        
        def parse_json_file(file_path):
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    return data
            except json.JSONDecodeError as e:
                print("Error parsing JSON")
                return None
            except FileNotFoundError:
                print("File not found")
                return None
        
        def parse_json_string(json_str):
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError as e:
                print("Error parsing JSON string")
                return None
        ```

        - SearchDocumentation: Would provide the official Python documentation on the json module:
        "The Python documentation explains that the `json` module provides methods like `json.loads()` for
        parsing strings and `json.load()` for file-like objects. It details all available parameters
        such as `object_hook` for custom object deserialization and `parse_float` for custom number parsing.
        It also documents error handling through the JSONDecodeError exception."

        - ExecuteCodeSnippet: Would show the parsing in action with real results:
        ```
        >>> import json
        >>> sample = '{{"name": "John", "age": 30, "city": "New York"}}'
        >>> parsed_data = json.loads(sample)
        >>> print(type(parsed_data))
        <class 'dict'>
        >>> print(parsed_data["name"])
        John
        >>> print(parsed_data["age"])
        30
        ```

        - SearchGitHub: Would find real-world implementations in actual projects:
        "Several repositories show advanced JSON parsing techniques such as custom deserializers,
        streaming parsers for large files, and schema validation implementations."

        - QueryOpenAI: Would explain concepts and provide examples based on best practices:
        "JSON parsing in Python is primarily done using the built-in json module. I'll explain how to
        parse both JSON strings and files, handle common errors, and work with nested data. Here are
        step-by-step examples covering simple and complex scenarios..."

        - SearchGoogle: Would provide tutorials, articles, and blog posts on the topic:
        "Returns comprehensive tutorials from Real Python, GeeksforGeeks, and official documentation covering
        basic to advanced JSON parsing topics, with examples for different use cases."

        - AnalyzeCode: Would examine existing JSON parsing code and suggest improvements:
        "Your current implementation doesn't handle nested JSON objects efficiently. Consider using
        the `object_pairs_hook` parameter to process duplicate keys or preserve order. Also, add error
        handling for UTF-8 encoding issues in the parsed data."

        Return just the list of questions, one per line.
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
            if cleaned_prompt.startswith(("- ", "• ", "1. ", "2. ")):
                cleaned_prompt = re.sub(r"^[•\-\d]+\.?\s+", "", cleaned_prompt).strip()
                
            # Ensure it's a question (has a question mark or starts with a question word)
            question_words = ["how", "what", "why", "where", "when", "which", "can", "should", "is", "are", "do", "does"]
            if ("?" in cleaned_prompt or any(cleaned_prompt.lower().startswith(word) for word in question_words)) and len(cleaned_prompt) > 15:
                tools_agnostic_prompts.append({
                    "id": str(len(tools_agnostic_prompts) + 1),
                    "text": cleaned_prompt,
                    "category": category
                })
    
    # Shuffle and limit to requested number
    random.shuffle(tools_agnostic_prompts)
    return tools_agnostic_prompts[:num_prompts]

def generate_tool_responses(prompt, tool_name):
    """Generate an example response for how a specific tool would answer the prompt"""
    client = OpenAI(api_key=args.openai_api_key)
    
    # Find the tool details
    tool = next((t for t in TOOLS if t["name"] == tool_name), None)
    if not tool:
        return f"Error: Tool {tool_name} not found"
    
    system_prompt = f"""
    Imagine you're using the {tool_name} tool to answer this question:
    "{prompt}"
    
    The {tool_name} tool has this description: {tool['description']}
    
    Generate a realistic, helpful response that would come from using this specific tool.
    Make it clear how this tool's unique capabilities address the question.
    Keep the response concise (3-5 sentences) but informative.
    """

    response = client.chat.completions.create(
        model=args.openai_model,
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.5,
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()

def save_dataset(prompts):
    """Save the dataset in various formats"""
    # Create base directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save as JSONL (one prompt per row)
    jsonl_path = os.path.join(args.output_dir, "dataset.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in prompts:
            f.write(json.dumps(entry) + "\n")
    
    # Create metadata file with tools information
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "dataset": "Tools Agnostic Questions",
            "version": "1.0",
            "description": "Questions that can be answered using multiple different tools",
            "date_created": datetime.now().isoformat(),
            "num_examples": len(prompts),
            "tools": TOOLS
        }, f, indent=2)
    
    # Save as a single JSON file without tool details in each entry
    json_path = os.path.join(args.output_dir, "tools_agnostic_dataset.json")
    with open(json_path, "w") as f:
        json.dump({
            "dataset": "Tools Agnostic Questions",
            "version": "1.0",
            "description": "Questions that can be answered using multiple different tools",
            "date_created": datetime.now().isoformat(),
            "num_examples": len(prompts),
            "examples": prompts
        }, f, indent=2)
      
    # Create README.md for dataset documentation
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"""# Tools Agnostic Questions Dataset

A collection of {len(prompts)} questions that can be meaningfully answered using multiple different tools.

## Dataset Description

This dataset contains questions that:
- Are phrased naturally as a real user would ask
- Can be answered using multiple different tools in meaningfully different ways
- Have enough depth to allow for different approaches to answering
- Cover a wide range of categories including: {", ".join(PROMPT_CATEGORIES)}

## Dataset Structure

Each entry contains:
- `id`: Unique identifier
- `text`: The question text
- `category`: The category this question belongs to

See `metadata.json` for details about the tools that can be used with this dataset.
""")
    
    return jsonl_path

def generate_default_hf_repo_name():
    """Generate a default Hugging Face repository name if not provided"""
    if not args.hf_token:
        raise ValueError("Hugging Face token is required to create a repository.")

    api = HfApi(token=args.hf_token)
    user_info = api.whoami()
    username = user_info.get("account", "Narmeen07")
    return f"{username}/tools_agnostic_questions"

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

        # Upload all files
        files_to_upload = [
            ("dataset.jsonl", os.path.join(args.output_dir, "dataset.jsonl")),
            ("metadata.json", os.path.join(args.output_dir, "metadata.json")),
            ("README.md", os.path.join(args.output_dir, "README.md"))
        ]
        
        for repo_path, local_path in files_to_upload:
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_name,
                    repo_type="dataset"
                )
                logger.info(f"Uploaded {repo_path} to Hugging Face repository")
            except Exception as e:
                logger.error(f"Failed to upload {repo_path} to Hugging Face: {e}")
        
        logger.info(f"✅ Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{repo_name}")
        return True
            
    except Exception as e:
        logger.error(f"Failed to upload dataset to Hugging Face: {e}")
        return False

def main():
    logger.info("Starting Tools-Agnostic Questions Dataset creation...")
    
    # Generate tools-agnostic prompts
    prompts = generate_tools_agnostic_prompts(args.num_prompts)
    
    # All entries should be tool-agnostic
    logger.info("Generating tool responses for the first 10 entries...")
    
    
    # Save dataset
    dataset_path = save_dataset(prompts)

    # Upload to Hugging Face
    if args.hf_token:
        upload_to_huggingface(dataset_path, args.hf_repo_name)
    else:
        logger.info("No Hugging Face token provided. Skipping upload.")
    
    logger.info(f"✅ Dataset saved at {dataset_path}")
    logger.info(f"Total questions generated: {len(prompts)}")

if __name__ == "__main__":
    main()