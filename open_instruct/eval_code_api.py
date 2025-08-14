#!/usr/bin/env python3
"""
Modified evaluation script that adds API-based model support to the existing HF evaluation framework
"""

import asyncio
import csv
import json
import os
import sys
import ray
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from vllm import SamplingParams
from sqlitedict import SqliteDict

# API clients
from anthropic import Anthropic
import anthropic
from openai import OpenAI
import requests
import dotenv
dotenv.load_dotenv()

# Original imports
from dataset_transformation import (
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
)
from ground_truth_utils import (
    build_all_verifiers,
    soft_format_reward_func,
)
from model_utils import ModelConfig, apply_verifiable_reward
from rl_utils2 import Timer
from utils import ArgumentParserPlus
from vllm_utils3 import create_vllm_engines


@dataclass
class APIConfig:
    """Configuration for API-based models"""
    
    # Model selection
    model_type: str = "huggingface"  # "huggingface", "deepseek", "anthropic", "openai"
    
    # API keys (direct, not Azure)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    
    # Model names
    anthropic_model_name: str = "claude-3-5-sonnet-20241022"  # Latest Claude model
    openai_model_name: str = "gpt-4o"  # Can be gpt-4o, gpt-4o-mini, o1-preview, o1-mini, etc.
    deepseek_model_name: str = "deepseek-reasoner"
    
    # Anthropic specific settings
    anthropic_thinking_budget: int = 30000
    
    # Generation parameters
    api_temperature: float = 1.0
    api_max_tokens: int = 32000
    api_timeout: int = 60
    api_max_retries: int = 3
    
    # Caching
    use_cache: bool = True
    cache_db_path: str = "api_evaluation_cache.db"


@dataclass
class EvalArgs:
    """Configuration for evaluation runs - same as original with added API support"""
    
    # NEW: API Configuration
    api_config: APIConfig = field(default_factory=APIConfig)
    
    # Dataset configuration (unchanged)
    dataset_mixer_eval_list: List[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    dataset_mixer_eval_list_splits: List[str] = field(default_factory=lambda: ["test"])
    dataset_transform_fn: List[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_filter_v1"])
    dataset_cache_mode: str = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_eval_hash: Optional[str] = None
    dataset_skip_cache: bool = False
    use_last_n_eval_samples: bool = True
    
    # All other configs remain exactly the same...
    num_runs: int = 1
    seed: int = 42
    response_length: int = 256
    temperature: float = 0.0
    vllm_top_p: float = 1.0
    stop_strings: Optional[List[str]] = None
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    vllm_enable_prefix_caching: bool = False
    vllm_gpu_memory_utilization: float = 0.9
    single_gpu_mode: bool = False
    apply_r1_style_format_reward: bool = True
    r1_style_format_reward: float = 1.0
    apply_verifiable_reward: bool = True
    verification_reward: float = 1.0
    additive_format_reward: bool = True
    only_reward_good_outputs: bool = False
    non_stop_penalty: bool = False
    non_stop_penalty_value: float = -1.0
    tools: Optional[List[str]] = None
    max_tool_calls: List[int] = field(default_factory=lambda: [5])
    code_tool_api_endpoint: Optional[str] = None
    search_api_endpoint: Optional[str] = None
    number_documents_to_search: int = 10
    code_api_url: Optional[str] = None
    code_max_execution_time: float = 1.0
    manufactoria_api_url: Optional[str] = None
    manufactoria_max_execution_time: float = 1.0
    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    llm_judge_max_tokens: int = 2048
    llm_judge_temperature: float = 1.0
    llm_judge_timeout: int = 60
    output_dir: str = "eval_results"
    save_results: bool = True
    hf_entity: Optional[str] = None


class APIClientManager:
    """NEW: Manages API clients for different model providers"""
    
    def __init__(self, api_config: APIConfig):
        self.config = api_config
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients based on configuration"""
        
        if self.config.model_type == "anthropic":
            self.clients["anthropic"] = Anthropic(
                api_key=self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        
        elif self.config.model_type == "openai":
            self.clients["openai"] = OpenAI(
                api_key=self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            )
        
        elif self.config.model_type == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            self.clients["deepseek"] = OpenAI(
                api_key=self.config.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
    
    async def generate_response(self, prompt: str, run_id: str = "") -> tuple:
        """Generate response using the configured API model"""
        
        # Create cache key
        cache_key = f"{self.config.model_type}_{hash(prompt)}_{run_id}"
        
        # Check cache first
        if self.config.use_cache:
            with SqliteDict(self.config.cache_db_path, autocommit=True) as cache:
                if cache_key in cache:
                    cached_result = cache[cache_key]
                    return (cached_result["response"], cached_result.get("thinking"), 
                           cached_result.get("input_tokens", 0), cached_result.get("output_tokens", 0))
        
        # Generate new response
        for attempt in range(self.config.api_max_retries):
            try:
                if self.config.model_type == "anthropic":
                    response, thinking, input_tokens, output_tokens = await self._generate_anthropic(prompt)
                elif self.config.model_type == "openai":
                    response, thinking, input_tokens, output_tokens = await self._generate_openai(prompt)
                elif self.config.model_type == "deepseek":
                    response, thinking, input_tokens, output_tokens = await self._generate_deepseek(prompt)
                else:
                    raise ValueError(f"Unsupported model type: {self.config.model_type}")
                
                # Cache the result
                if self.config.use_cache and response is not None:
                    result = {
                        "response": response,
                        "thinking": thinking,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "timestamp": time.time()
                    }
                    with SqliteDict(self.config.cache_db_path, autocommit=True) as cache:
                        cache[cache_key] = result
                
                return response, thinking, input_tokens, output_tokens
                
            except Exception as e:
                if attempt == self.config.api_max_retries - 1:
                    print(f'Error in generate_response for {self.config.model_type}')
                    traceback.print_exc()
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None, None, 0, 0
    
    async def _generate_anthropic(self, prompt: str) -> tuple:
        """Generate response using Anthropic API"""
        response = self.clients["anthropic"].messages.create(
            model=self.config.anthropic_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.api_max_tokens,
            temperature=self.config.api_temperature,
        )
        
        response_text = response.content[0].text if response.content else None
        return (response_text, None, 
                response.usage.input_tokens, response.usage.output_tokens)
    
    async def _generate_openai(self, prompt: str) -> tuple:
        """Generate response using OpenAI API"""
        response = self.clients["openai"].chat.completions.create(
            model=self.config.openai_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.config.api_max_tokens,
            temperature=self.config.api_temperature,
        )
        
        return (response.choices[0].message.content, None,
                response.usage.prompt_tokens, response.usage.completion_tokens)
    
    async def _generate_deepseek(self, prompt: str) -> tuple:
        """Generate response using DeepSeek API (OpenAI-compatible)"""
        response = self.clients["deepseek"].chat.completions.create(
            model=self.config.deepseek_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.config.api_max_tokens,
            temperature=self.config.api_temperature,
        )
        
        return (response.choices[0].message.content, None,
                response.usage.prompt_tokens, response.usage.completion_tokens)


class CodeEvaluator:
    def __init__(self, args: EvalArgs, tokenizer_config: TokenizerConfig, model_config: ModelConfig):
        self.args = args
        self.tokenizer_config = tokenizer_config
        self.model_config = model_config
        self.tokenizer = None
        
        # NEW: Check if using API model
        self.use_api = self.args.api_config.model_type != "huggingface"
        if self.use_api:
            self.api_client = APIClientManager(self.args.api_config)
        else:
            self.api_client = None
        
        # Ensure tokenizer_config has valid values before any use
        if self.tokenizer_config.tokenizer_name_or_path is None:
            self.tokenizer_config.tokenizer_name_or_path = self.model_config.model_name_or_path
            self.tokenizer_config.tokenizer_revision = self.model_config.model_revision
        
        self.vllm_engines = None
        self.reward_fn_mapping = None
        self.eval_dataset = None
        
        # Parse dataset configurations for individual processing
        self.dataset_configs = self.parse_dataset_configs()
    
    # All existing methods remain the same until generate_responses...
    def parse_dataset_configs(self) -> List[Dict]:
        """Parse dataset_mixer_eval_list into individual dataset configurations"""
        configs = []
        
        for i in range(0, len(self.args.dataset_mixer_eval_list), 2):
            dataset_name = self.args.dataset_mixer_eval_list[i]
            frac_or_num_samples = self.args.dataset_mixer_eval_list[i+1]
            
            # Clean up dataset name for folder usage            
            folder_name = dataset_name.split("/")[-1].replace("-", "_")
            
            configs.append({
                "dataset_name": dataset_name,
                "frac_or_num_samples": frac_or_num_samples,
                "folder_name": folder_name,
                "output_dir": os.path.join(self.args.output_dir, folder_name)
            })
        
        return configs
        
    def setup(self):
        """Initialize all components needed for evaluation"""
        print("🔧 Setting up evaluator...")
        
        # Initialize tokenizer (always needed for dataset processing)
        from transformers import AutoTokenizer
        tokenizer_path = (self.model_config.model_name_or_path if not self.use_api 
                         else "microsoft/DialoGPT-medium")  # Default tokenizer for API models
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            revision=self.model_config.model_revision if not self.use_api else None,
            trust_remote_code=self.tokenizer_config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load evaluation dataset (same as original)
        print("📚 Loading evaluation dataset...")
        tc = self.tokenizer_config
        eval_transform_fn_args = [
            {},
            {"need_contain_labels": True},  # No length filtering for eval
        ]
        
        self.eval_dataset = get_cached_dataset_tulu(
            self.args.dataset_mixer_eval_list,
            self.args.dataset_mixer_eval_list_splits,
            tc,
            self.args.dataset_transform_fn,
            eval_transform_fn_args,
            hf_entity=self.args.hf_entity,
            dataset_cache_mode=self.args.dataset_cache_mode,
            dataset_config_hash=self.args.dataset_config_eval_hash,
            dataset_local_cache_dir=self.args.dataset_local_cache_dir,
            dataset_skip_cache=self.args.dataset_skip_cache,
            use_last_n=self.args.use_last_n_eval_samples,
        )
        
        # Add dataset source tracking (same as original)
        sources = []
        for i in range(0, len(self.args.dataset_mixer_eval_list), 2):
            dataset_name = self.args.dataset_mixer_eval_list[i]
            frac_or_num_samples = self.args.dataset_mixer_eval_list[i+1]
            
            if "." in frac_or_num_samples:
                num_samples = int(float(frac_or_num_samples) * len(self.eval_dataset))
            else:
                num_samples = int(frac_or_num_samples)
            
            sources.extend([dataset_name] * num_samples)
        
        self.eval_dataset = self.eval_dataset.add_column("__dataset_source__", sources)
        
        # Initialize VLLM engines only for HF models
        if not self.use_api:
            print("🚀 Initializing VLLM engines...")
            max_len = 512 + self.args.response_length  # Rough estimate
            
            # Setup tools if specified (same as original)
            tool_objects = {}
            if self.args.tools:
                for tool in self.args.tools:
                    if tool.lower() == "code":
                        from tool_utils.tool_vllm import PythonCodeTool
                        tool_obj = PythonCodeTool(
                            start_str="<code>",
                            end_str="</code>",
                            api_endpoint=self.args.code_tool_api_endpoint,
                        )
                        tool_objects[tool_obj.end_str] = tool_obj
                    elif tool.lower() == "search":
                        from search_utils.search_tool import SearchTool
                        tool_obj = SearchTool(
                            start_str="<query>",
                            end_str="</query>",
                            api_endpoint=self.args.search_api_endpoint,
                            number_documents_to_search=self.args.number_documents_to_search,
                        )
                        tool_objects[tool_obj.end_str] = tool_obj
                        
            self.vllm_engines = create_vllm_engines(
                self.args.vllm_num_engines,
                self.args.vllm_tensor_parallel_size,
                self.args.vllm_enforce_eager,
                self.model_config.model_name_or_path,
                self.model_config.model_revision,
                self.args.seed,
                self.args.vllm_enable_prefix_caching,
                max_len,
                self.args.vllm_gpu_memory_utilization,
                self.args.single_gpu_mode,
                tools=tool_objects,
                max_tool_calls=self.args.max_tool_calls,
            )
            
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init(dashboard_host="0.0.0.0", include_dashboard=False)
        else:
            print(f"🌐 Using API model: {self.args.api_config.model_type}")
        
        # Initialize reward functions (same as original)
        print("🏆 Setting up reward functions...")
        self.reward_fn_mapping = build_all_verifiers(self.args)
        
        print("✅ Setup complete!")
    
    def generate_responses(self, prompt_token_ids: List[List[int]], run_id: int = 0) -> tuple:
        """Generate responses using either VLLM engines or API clients"""
        
        if self.use_api:
            # For API models: decode token IDs to text prompts first
            prompts = self.tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=True)
            
            # Generate responses sequentially using API
            decoded_responses = []
            response_ids = []
            finish_reasons = []
            masks = []
            infos = [[], [], [], [], [], []]  # num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds
            
            for i, prompt in enumerate(prompts):
                try:
                    # Run each API call
                    result = asyncio.run(self.api_client.generate_response(prompt, f"{run_id}_{i}"))
                    response_text, thinking, input_tokens, output_tokens = result
                    
                    decoded_responses.append(response_text or "")
                    
                    # Create token IDs for API responses (needed for reward computation)
                    if response_text:
                        tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
                        response_ids.append(tokens)
                    else:
                        response_ids.append([])
                    
                    finish_reasons.append("stop" if response_text else "error")
                    masks.append([1] * len(response_ids[-1]))
                    
                    # Add default values for tool info (API models don't use tools in this context)
                    infos[0].append(0)      # num_calls
                    infos[1].append(False)  # timeouts
                    infos[2].append(False)  # tool_errors
                    infos[3].append("")     # tool_outputs
                    infos[4].append(0.0)    # tool_runtimes
                    infos[5].append(False)  # tool_calleds
                    
                except Exception as e:
                    print(f"API call failed for prompt {i}: {e}")
                    decoded_responses.append("")
                    response_ids.append([])
                    finish_reasons.append("error")
                    masks.append([])
                    # Add default values for tool info
                    for j, info_list in enumerate(infos):
                        if j in [1, 2, 5]:  # Boolean fields
                            info_list.append(False)
                        elif j == 3:  # String field
                            info_list.append("")
                        else:  # Numeric fields
                            info_list.append(0)
            
            return response_ids, finish_reasons, masks, infos
            
        else:
            # For HF models: use the ORIGINAL VLLM logic (unchanged!)
            generation_config = SamplingParams(
                temperature=self.args.temperature,
                top_p=self.args.vllm_top_p,
                max_tokens=self.args.response_length,
                include_stop_str_in_output=True,
                skip_special_tokens=False,
                stop=self.args.stop_strings,
            )
            
            # Split queries between engines
            queries_per_engine = (len(prompt_token_ids) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
            split_queries = [
                prompt_token_ids[i:i + queries_per_engine] 
                for i in range(0, len(prompt_token_ids), queries_per_engine)
            ]
            
            # Generate responses in parallel - using standard generate.remote() method
            futures = []
            for i, (engine, queries) in enumerate(zip(self.vllm_engines, split_queries)):
                if queries:  # Only process non-empty query lists
                    future = engine.generate.remote(sampling_params=generation_config, prompt_token_ids=queries, use_tqdm=False)
                    futures.append(future)
            
            # Collect and process results - same pattern as grpo_fast_code.py
            all_outputs = ray.get(futures)
            
            response_ids = []
            finish_reasons = []
            masks = []
            num_calls = []
            timeouts = []
            tool_errors = []
            tool_outputs = []
            tool_runtimes = []
            tool_calleds = []
            
            # Check if we're using tools
            tool_use = self.args.tools is not None and len(self.args.tools) > 0
            
            for outputs in all_outputs:
                response_ids.extend([list(out.token_ids) for output in outputs for out in output.outputs])
                finish_reasons.extend([out.finish_reason for output in outputs for out in output.outputs])
                if tool_use:
                    masks.extend([out.mask for output in outputs for out in output.outputs])
                    num_calls.extend([out.num_calls for output in outputs for out in output.outputs])
                    timeouts.extend([out.timeout for output in outputs for out in output.outputs])
                    tool_errors.extend([out.tool_error for output in outputs for out in output.outputs])
                    tool_outputs.extend([out.tool_output for output in outputs for out in output.outputs])
                    tool_runtimes.extend([out.tool_runtime for output in outputs for out in output.outputs])
                    tool_calleds.extend([out.tool_called for output in outputs for out in output.outputs])
            
            # If not using tools, provide default values
            if not tool_use:
                masks = [[1] * len(response_ids[i]) for i in range(len(response_ids))]
                num_calls = [0] * len(response_ids)
                timeouts = [False] * len(response_ids)
                tool_errors = [False] * len(response_ids)
                tool_outputs = [""] * len(response_ids)
                tool_runtimes = [0.0] * len(response_ids)
                tool_calleds = [False] * len(response_ids)
            
            # Package infos in the expected format
            infos = [num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds]
            
            return response_ids, finish_reasons, masks, infos
