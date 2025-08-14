import json
import os
import pdb
import concurrent.futures
import multiprocessing
import random
import time
import traceback
import argparse
import uuid
import threading
from sqlitedict import SqliteDict
from math_verify import parse, verify
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import dotenv
dotenv.load_dotenv()

os.environ["DEEPSEEK_AZURE_KEY_CREDENTIAL"] = ""
endpoint = "https://ai-generaladapthub866259938521.services.ai.azure.com/models"
model_name = "DeepSeek-R1"

# endpoint = "https://ai-olmohub1163464654570.openai.azure.com/"

deepseek_client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(os.getenv("DEEPSEEK_AZURE_KEY_CREDENTIAL")),
)

# Add boxed instruction to prompts
BOXED_INSTRUCTION = "\n\nLet's output the final answer within \\boxed{}."

def generate_deepseek_response(question, temperature=1.0):
    question_with_instruction = question + BOXED_INSTRUCTION
    for attempt in range(3):
        try:
            response = deepseek_client.complete(
                messages=[
                    SystemMessage(content="You are a helpful assistant."),
                    UserMessage(content=question_with_instruction),
                ],
                max_tokens=18000,
                temperature=temperature,
                model=model_name,
                timeout=1800
            )
            return response.choices[0].message.content, None, response.usage.prompt_tokens, response.usage.completion_tokens
        except Exception as e:
            if attempt == 2:  # On the third attempt (index 2)
                print('Error in generate_deepseek_response')
                traceback.print_exc()
            # Continue to next attempt if not the last one
    return None, None, 0, 0


#
# ##############################
#
# from google import genai
# google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
# def generate_gemini_response(question):
#     for attempt in range(3):
#         try:
#             response = google_client.models.generate_content(
#                 model="gemini-2.5-pro-exp-03-25",
#                 contents=question,
#             config=genai.types.GenerateContentConfig(
#                 max_output_tokens=16384,
#                 temperature=0.0,
#                 thinking_config=genai.types.ThinkingConfig(
#                     thinking_budget=15800
#                 ),
#                 tools=[],
#                 ),
#             )
#             return response.text, None, response.usage_metadata.prompt_token_count, response.usage_metadata.total_token_count - response.usage_metadata.prompt_token_count
#         except Exception as e:
#             if attempt == 2:  # On the third attempt (index 2)
#                 print('Error in generate_gemini_response')
#                 traceback.print_exc()
#     return None, None, 0, 0

##############################
os.environ["ANTHROPIC_API_KEY"] = ""

from anthropic import Anthropic
import anthropic
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
def generate_anthropic_response(question):
    question_with_instruction = question + BOXED_INSTRUCTION
    for attempt in range(3):
        try:
            response = anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",  #claude-sonnet-4-20250514
                messages=[
                {"role": "user", "content": question_with_instruction},
                ],
                tools=[],
                max_tokens=32000,
                timeout=60 * 20,
                thinking={
                "type": "enabled",
                "budget_tokens": 30000
            },
            # tool_choice="none", # wrong syntax, but none by default according to docs
            # temperature=0.0, # parameter not enabled when thinking is enabled
        )
            # response.content[0] is the ThinkingBlock (with field "thinking")
            response_thinking = None
            response_text = None
            for content in response.content:
                if content.type == 'thinking':
                    response_thinking = content.thinking
                elif content.type == 'text':
                    response_text = content.text
            return response_text, response_thinking, response.usage.input_tokens, response.usage.output_tokens
        except anthropic.RateLimitError as e:
            time.sleep(60)
        except Exception as e:
            if attempt == 2:  # On the third attempt (index 2)
                print('Error in generate_anthropic_response')
                traceback.print_exc()
    return None, None, 0, 0


##############################

from openai import AzureOpenAI

# Azure OpenAI configuration
openai_client_1 = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://ai-olmohub1163464654570.openai.azure.com/",
    api_key="",
)

openai_client_2 = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://ai-generaladapthub866259938521.openai.azure.com/",
    api_key="",
)

def generate_o4mini_response(question):
    return generate_openai_response(question, openai_client_1, "o4-mini-standard")

def generate_o3mini_response(question):
    return generate_openai_response(question, openai_client_2, "o3-mini-standard")

def generate_openai_response(question, client, deployment_name):
    question_with_instruction = question + BOXED_INSTRUCTION
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": question_with_instruction,
                    }
                ],
                max_completion_tokens=32000,
                temperature=1.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                model=deployment_name
            )
            
            response_text = response.choices[0].message.content
            response_summary = None  # Azure OpenAI doesn't provide reasoning summaries like o1 models
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            return response_text, response_summary, input_tokens, output_tokens
            
        except Exception as e:
            if attempt == 2:  # On the third attempt (index 2)
                print('Error in generate_openai_response')
                traceback.print_exc()
            # Continue to next attempt if not the last one
    
    return None, None, 0, 0

# Model mapping
model_to_func = {
    "deepseek": generate_deepseek_response,
    "o4mini": generate_o4mini_response,
    "o3mini": generate_o3mini_response,
    "anthropic": generate_anthropic_response,
}

def create_cache_key(model_name, problem_family, difficulty, question_idx, run_number):
    """Create a unique cache key for a specific evaluation run"""
    return f"{model_name}_{problem_family}_{difficulty}_{question_idx}_{run_number}"

def evaluate_answer(intended_answer, response_text):
    if response_text is None:
        return False
    return verify(parse(str(intended_answer)), parse(response_text), float_rounding=2)

def load_problem_samples(prob_family, difficulty, num_samples=10):
    """Load samples from a problem family and difficulty level"""
    jsonl_file_path = f"problems/isolated_difficulty/{prob_family}_level_{difficulty}.jsonl"
    
    if not os.path.exists(jsonl_file_path):
        print(f"Warning: File {jsonl_file_path} not found")
        return []
    
    samples = []
    with open(jsonl_file_path, 'r') as f:
        for idx, line in enumerate(f):
            if len(samples) >= num_samples:
                break
            data = json.loads(line.strip())
            samples.append({
                'idx': idx,
                'question': data['question'],
                'intended_answer': data['answer'],
                'difficulty': difficulty
            })
    
    return samples

def evaluate_single_run(args):
    """Evaluate a single run for one question with one model"""
    (model_name, model_func, question_data, run_number, 
     cache_db_path, prob_family) = args
    
    question = question_data['question']
    intended_answer = question_data['intended_answer']
    difficulty = question_data['difficulty']
    question_idx = question_data['idx']
    
    cache_key = create_cache_key(model_name, prob_family, difficulty, question_idx, run_number)
    
    # Check cache first
    with SqliteDict(cache_db_path, autocommit=True) as cache:
        if cache_key in cache:
            cached_result = cache[cache_key]
            print(f"Using cached result for {model_name} - Q{question_idx} - Run{run_number}")
            return cached_result
    
    print(f"Running {model_name} - Difficulty{difficulty} - Q{question_idx} - Run{run_number}")
    
    # Generate response
    response_text, response_summary, input_tokens, output_tokens = model_func(question)
    
    if response_text is None:
        print(f"Failed to get response for {model_name} - Q{question_idx} - Run{run_number}")
        return None
    
    # Evaluate correctness
    correct = evaluate_answer(intended_answer, response_text)

    # Create result object
    result = {
        'run_id': cache_key,
        'model_name': model_name,
        'problem_family': prob_family,
        'difficulty': difficulty,
        'question_idx': question_idx,
        'run_number': run_number,
        'question': question,
        'intended_answer': intended_answer,
        'response_text': response_text,
        'response_summary': response_summary,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'correct': correct,
        'timestamp': time.time()
    }
    
    # Cache the result
    with SqliteDict(cache_db_path, autocommit=True) as cache:
        cache[cache_key] = result
    
    return result

def evaluate_models_on_problem_family(prob_family, cache_db_path="evaluation_cache.db", 
                                    num_samples=10, num_runs=64, max_workers=40, model_name=None):
    """
    Evaluate all models on a problem family across all difficulty levels
    
    Args:
        prob_family: Problem family name (e.g., "logic_puzzles_grid_knight")
        cache_db_path: Path to SQLite cache database
        num_samples: Number of samples per difficulty level (default: 10)
        num_runs: Number of runs per sample (default: 64)
        max_workers: Maximum number of multiprocessing workers
        model_name: Specific model to run (if None, run all models)
    """
    
    # Select models to run
    if model_name:
        if model_name not in model_to_func:
            print(f"Error: Model '{model_name}' not found. Available models: {list(model_to_func.keys())}")
            return
        models_to_run = {model_name: model_to_func[model_name]}
        print(f"Running evaluation for model: {model_name}")
    else:
        models_to_run = model_to_func
        print(f"Running evaluation for all models: {list(model_to_func.keys())}")
    
    # Collect all tasks to run
    tasks = []
    all_samples = {}
    
    # Load samples for all difficulty levels
    for difficulty in range(6, 0, -1):
        samples = load_problem_samples(prob_family, difficulty, num_samples)
        all_samples[difficulty] = samples
        
        for run_number in range(num_runs):
            for model_name_iter, model_func in models_to_run.items():
                for sample in samples:
                    tasks.append((model_name_iter, model_func, sample, run_number, cache_db_path, prob_family))
    
    print(f"Total tasks to run: {len(tasks)}")
    
    # Run evaluation with multiprocessing
    # Note: We prioritize looping over questions by structuring tasks appropriately
    with multiprocessing.Pool(max_workers) as pool:
        results = []
        
        # Submit all tasks
        for task in tasks:
            result = pool.apply_async(evaluate_single_run, (task,))
            results.append(result)
        
        # Collect results
        completed_results = []
        for i, result in enumerate(results):
            try:
                res = result.get(timeout=1800)  # 30 minute timeout per task
                if res is not None:
                    completed_results.append(res)
                    print(f"Completed {i+1}/{len(results)} tasks")
            except Exception as e:
                print(f"Task {i+1} failed: {e}")
                traceback.print_exc()
    
    print(f"Completed {len(completed_results)} out of {len(tasks)} total tasks")
    
    # Generate summary statistics
    generate_summary_stats(cache_db_path, prob_family)

def generate_summary_stats(cache_db_path, prob_family):
    """Generate and print summary statistics from cached results"""
    print(f"\n=== Summary Statistics for {prob_family} ===")
    
    # Load all results from cache
    all_results = []
    with SqliteDict(cache_db_path, flag='r') as cache:
        for key, result in cache.items():
            if result['problem_family'] == prob_family:
                all_results.append(result)
    
    if not all_results:
        print("No results found in cache for this problem family.")
        return
    
    # Group results by model and difficulty
    model_difficulty_stats = {}
    model_totals = {}
    
    for result in all_results:
        model_name = result['model_name']
        difficulty = result['difficulty']
        correct = result['correct']
        input_tokens = result['input_tokens']
        output_tokens = result['output_tokens']
        
        # Initialize nested dictionaries
        if model_name not in model_difficulty_stats:
            model_difficulty_stats[model_name] = {}
            model_totals[model_name] = {'total': 0, 'correct': 0, 'input_tokens': 0, 'output_tokens': 0}
        
        if difficulty not in model_difficulty_stats[model_name]:
            model_difficulty_stats[model_name][difficulty] = {'total': 0, 'correct': 0}
        
        # Update stats
        model_difficulty_stats[model_name][difficulty]['total'] += 1
        if correct:
            model_difficulty_stats[model_name][difficulty]['correct'] += 1
        
        model_totals[model_name]['total'] += 1
        if correct:
            model_totals[model_name]['correct'] += 1
        model_totals[model_name]['input_tokens'] += input_tokens
        model_totals[model_name]['output_tokens'] += output_tokens
    
    # Print accuracy by model and difficulty
    print("\nAccuracy by Model and Difficulty:")
    print("Model\t\tDiff\tRuns\tCorrect\tAccuracy%")
    print("-" * 50)
    
    for model_name in sorted(model_difficulty_stats.keys()):
        for difficulty in sorted(model_difficulty_stats[model_name].keys()):
            stats = model_difficulty_stats[model_name][difficulty]
            accuracy_pct = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{model_name:12}\t{difficulty}\t{stats['total']}\t{stats['correct']}\t{accuracy_pct:.2f}%")
    
    # Print overall model performance
    print("\nOverall Model Performance:")
    print("Model\t\tTotal\tCorrect\tAccuracy%")
    print("-" * 40)
    for model_name in sorted(model_totals.keys()):
        stats = model_totals[model_name]
        accuracy = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{model_name:12}\t{stats['total']}\t{stats['correct']}\t{accuracy:.2f}%")
    
    # Print token usage summary
    print("\nToken Usage Summary:")
    print("Model\t\tRuns\tInput Tokens\tOutput Tokens\tAvg Input\tAvg Output")
    print("-" * 75)
    for model_name in sorted(model_totals.keys()):
        stats = model_totals[model_name]
        avg_input = stats['input_tokens'] / stats['total'] if stats['total'] > 0 else 0
        avg_output = stats['output_tokens'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{model_name:12}\t{stats['total']}\t{stats['input_tokens']:,}\t\t{stats['output_tokens']:,}\t\t{avg_input:.0f}\t{avg_output:.0f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate frontier models on math problems')
    parser.add_argument('--prob_family', type=str, default='geometry_rotation',
                       help='Problem family name (default: logic_puzzles_grid_knight, combinatory_distribution, combinatory_pattern_matching, geometry_rotation_h)')
    parser.add_argument('--cache_db', type=str, default='evaluation/geometry_rotation.db',
                       help='Path to cache database (default: evaluation_cache.db)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples per difficulty level (default: 10)')
    parser.add_argument('--num_runs', type=int, default=64,
                       help='Number of runs per sample (default: 64)')
    parser.add_argument('--max_workers', type=int, default=40,
                       help='Maximum number of multiprocessing workers (default: 40)')
    parser.add_argument('--model', type=str, default='o4mini', choices=list(model_to_func.keys()),
                       help='Specific model to run. Options: %(choices)s. If not specified, runs all models.')
    parser.add_argument('--summary_only', action='store_true',
                       help='Only generate summary statistics from existing cache')
    
    args = parser.parse_args()
    
    if args.summary_only:
        generate_summary_stats(args.cache_db, args.prob_family)
    else:
        evaluate_models_on_problem_family(
            prob_family=args.prob_family,
            cache_db_path=args.cache_db,
            num_samples=args.num_samples,
            num_runs=args.num_runs,
            max_workers=args.max_workers,
            model_name=args.model
        )

if __name__ == "__main__":
    main()