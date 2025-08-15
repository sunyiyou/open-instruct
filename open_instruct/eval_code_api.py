#!/usr/bin/env python3
"""
Modified evaluation script that adds API-based model support to the existing HF evaluation framework
API mode in this version:
  - Never initializes VLLM or uses GPUs
  - Can fully avoid tokenizer usage (no decode/encode) with APIConfig.no_tokenizer=True
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
from open_instruct.dataset_transformation import(
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
)
from open_instruct.ground_truth_utils import (
    build_all_verifiers,
    soft_format_reward_func,
)
from open_instruct.model_utils import ModelConfig, apply_verifiable_reward
from open_instruct.rl_utils2 import Timer
from open_instruct.utils import ArgumentParserPlus
from open_instruct.vllm_utils3 import create_vllm_engines


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
    anthropic_model_name: str = "claude-3-5-sonnet-20241022"
    openai_model_name: str = "gpt-4o"
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

    # NEW: completely avoid tokenizer when using API
    no_tokenizer: bool = True


@dataclass
class EvalArgs:
    """Configuration for evaluation runs - same as original with added API support"""

    model_type: str = "huggingface"  # "huggingface", "deepseek", "anthropic", "openai"

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
    """Manages API clients for different model providers"""

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
        cache_key = f"{self.config.model_type}_{hash(prompt)}_{run_id}"

        # Cache
        if self.config.use_cache:
            with SqliteDict(self.config.cache_db_path, autocommit=True) as cache:
                if cache_key in cache:
                    cached = cache[cache_key]
                    return (cached["response"], cached.get("thinking"),
                            cached.get("input_tokens", 0), cached.get("output_tokens", 0))

        # Call provider
        for attempt in range(self.config.api_max_retries):
            try:
                if self.config.model_type == "anthropic":
                    response, thinking, itok, otok = await self._generate_anthropic(prompt)
                elif self.config.model_type == "openai":
                    response, thinking, itok, otok = await self._generate_openai(prompt)
                elif self.config.model_type == "deepseek":
                    response, thinking, itok, otok = await self._generate_deepseek(prompt)
                else:
                    raise ValueError(f"Unsupported model type: {self.config.model_type}")

                if self.config.use_cache and response is not None:
                    with SqliteDict(self.config.cache_db_path, autocommit=True) as cache:
                        cache[cache_key] = {
                            "response": response,
                            "thinking": thinking,
                            "input_tokens": itok,
                            "output_tokens": otok,
                            "timestamp": time.time(),
                        }
                return response, thinking, itok, otok
            except Exception:
                if attempt == self.config.api_max_retries - 1:
                    print(f'Error in generate_response for {self.config.model_type}')
                    traceback.print_exc()
                else:
                    await asyncio.sleep(2 ** attempt)  # backoff
        return None, None, 0, 0

    async def _generate_anthropic(self, prompt: str) -> tuple:
        """Generate response using Anthropic API"""
        
        resp = self.clients["anthropic"].messages.create(
            # model=self.config.anthropic_model_name,
            model="claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": prompt}],
            # max_tokens=self.config.api_max_tokens,
            max_tokens=32000,
            timeout=60 * 20,
            thinking={
                "type": "enabled",
                "budget_tokens": 30000
            },
            temperature=self.config.api_temperature,

        )
        # text = resp.content[0].text if resp.content else None
        
        text = None
        thinking = None

        # Extract text and thinking from response content
        if resp.content:
            for block in resp.content:
                if hasattr(block, 'text'):  # TextBlock
                    text = block.text
                elif hasattr(block, 'thinking'):  # ThinkingBlock
                    thinking = block.thinking
                    print(f"🧠 THINKING LENGTH: {len(thinking) if thinking else 0} chars")

        # Fallback if no text found
        if text is None:
            print("⚠️ No text block found in response")
            text = ""

        print(f"✅ ANTHROPIC RESPONSE GENERATED SUCCESSFULLY!")
        print(f"📝 Response length: {len(text) if text else 0} characters")
        print(f"🔍 First 300 chars: {text[:300] if text else 'None'}...")
        print(f"📊 Input tokens: {getattr(resp.usage, 'input_tokens', 0)}")
        print(f"📊 Output tokens: {getattr(resp.usage, 'output_tokens', 0)}")

        # breakint()

        return (text, None, getattr(resp.usage, "input_tokens", 0), getattr(resp.usage, "output_tokens", 0))

    async def _generate_openai(self, prompt: str) -> tuple:
        """Generate response using OpenAI API"""
        resp = self.clients["openai"].chat.completions.create(
            model=self.config.openai_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.config.api_max_tokens,
            temperature=self.config.api_temperature,
        )
        text = resp.choices[0].message.content if resp.choices else None
        return (text, None,
                getattr(resp.usage, "prompt_tokens", 0),
                getattr(resp.usage, "completion_tokens", 0))

    async def _generate_deepseek(self, prompt: str) -> tuple:
        """Generate response using DeepSeek API (OpenAI-compatible)"""
        resp = self.clients["deepseek"].chat.completions.create(
            model=self.config.deepseek_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.config.api_max_tokens,
            temperature=self.config.api_temperature,
        )
        text = resp.choices[0].message.content if resp.choices else None
        return (text, None,
                getattr(resp.usage, "prompt_tokens", 0),
                getattr(resp.usage, "completion_tokens", 0))


class CodeEvaluator:
    def __init__(self, args: EvalArgs, tokenizer_config: TokenizerConfig, model_config: ModelConfig):

        self.args = args
        self.tokenizer_config = tokenizer_config
        self.model_config = model_config
        self.tokenizer = None

        # NEW: API mode?
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

        self.dataset_configs = self.parse_dataset_configs()

    def parse_dataset_configs(self) -> List[Dict]:
        """Parse dataset_mixer_eval_list into individual dataset configurations"""
        configs = []
        for i in range(0, len(self.args.dataset_mixer_eval_list), 2):
            dataset_name = self.args.dataset_mixer_eval_list[i]
            frac_or_num_samples = self.args.dataset_mixer_eval_list[i+1]
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

        # NEW: in API mode, hard-disable CUDA so nothing touches GPUs
        if self.use_api:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            print(f"🌐 Using API model: {self.args.api_config.model_type}")

        # Initialize tokenizer only if needed
        if (not self.use_api) or (self.use_api and not self.args.api_config.no_tokenizer):
            from transformers import AutoTokenizer
            tokenizer_path = (self.model_config.model_name_or_path if not self.use_api
                              else "microsoft/DialoGPT-medium")  # harmless small default
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                revision=self.model_config.model_revision if not self.use_api else None,
                trust_remote_code=self.tokenizer_config.trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None  # absolutely no tokenizer use in API mode

        # Load evaluation dataset
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

        # Add dataset source tracking
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
            max_len = 512 + self.args.response_length
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

            if not ray.is_initialized():
                ray.init(dashboard_host="0.0.0.0", include_dashboard=False)

        print("🏆 Setting up reward functions...")
        self.reward_fn_mapping = build_all_verifiers(self.args)

        print("✅ Setup complete!")

    # NEW: helper to extract text prompts without tokenizers
    def _extract_prompts_as_text(self, ds):
        cols = set(ds.column_names)
        if "prompt" in cols:
            return ds["prompt"]
        if "messages" in cols:
            return [" | ".join([m["content"] for m in msgs if m.get("role") == "user"]) for msgs in ds["messages"]]
        # If only token IDs exist and tokenizer disabled, we can't proceed
        if self.tokenizer is None:
            raise RuntimeError(
                "Dataset has only token IDs but tokenizer is disabled. "
                "Provide a 'prompt' or 'messages' column for API mode, or set api_config.no_tokenizer=False."
            )
        # Last resort: decode using tokenizer (only if enabled)
        return self.tokenizer.batch_decode(ds[INPUT_IDS_PROMPT_KEY], skip_special_tokens=True)

    async def generate_responses(
        self,
        prompt_token_ids: Optional[List[List[int]]] = None,
        prompts_text: Optional[List[str]] = None,
        run_id: int = 0
    ) -> tuple:
        """
        Generate responses using either VLLM engines or API clients.
        Returns: (response_ids, finish_reasons, masks, infos, decoded_texts)
        decoded_texts is non-empty only for API path; HF path returns [].
        """
        # breakpoint()
        if self.use_api:
            # Build prompt texts
            if prompts_text is None:
                if self.tokenizer is None:
                    raise RuntimeError("API mode without tokenizer needs prompts_text.")
                prompts = self.tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=True)
            else:
                prompts = prompts_text

            decoded_texts = []
            response_ids, finish_reasons, masks = [], [], []
            infos = [[], [], [], [], [], []]  # num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds

            for i, prompt in enumerate(prompts):
                try:
                    # response_text, thinking, input_tokens, output_tokens = asyncio.run(
                    #     self.api_client.generate_response(prompt, f"{run_id}_{i}")
                    # )
                    response_text, thinking, input_tokens, output_tokens = await self.api_client.generate_response(prompt, f"{run_id}_{i}")


                    text = response_text or ""
                    decoded_texts.append(text)

                    # In tokenizer-less API mode, don't re-encode to IDs
                    if self.tokenizer is None:
                        response_ids.append([])
                        masks.append([])
                    else:
                        toks = self.tokenizer.encode(text, add_special_tokens=False) if text else []
                        response_ids.append(toks)
                        masks.append([1] * len(toks))

                    finish_reasons.append("stop" if text else "error")

                    # default tool info
                    infos[0].append(0)
                    infos[1].append(False)
                    infos[2].append(False)
                    infos[3].append("")
                    infos[4].append(0.0)
                    infos[5].append(False)

                except Exception as e:
                    print(f"API call failed for prompt {i}: {e}")
                    decoded_texts.append("")
                    response_ids.append([])
                    finish_reasons.append("error")
                    masks.append([])
                    for arr, val in zip(infos, [0, False, False, "", 0.0, False]):
                        arr.append(val)

            return response_ids, finish_reasons, masks, infos, decoded_texts

        # HF/VLLM path (unchanged)
        generation_config = SamplingParams(
            temperature=self.args.temperature,
            top_p=self.args.vllm_top_p,
            max_tokens=self.args.response_length,
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            stop=self.args.stop_strings,
        )

        queries_per_engine = (len(prompt_token_ids) + len(self.vllm_engines) - 1) // len(self.vllm_engines)
        split_queries = [
            prompt_token_ids[i:i + queries_per_engine]
            for i in range(0, len(prompt_token_ids), queries_per_engine)
        ]

        futures = []
        for i, (engine, queries) in enumerate(zip(self.vllm_engines, split_queries)):
            if queries:
                future = engine.generate.remote(sampling_params=generation_config, prompt_token_ids=queries, use_tqdm=False)
                futures.append(future)

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

        if not tool_use:
            masks = [[1] * len(response_ids[i]) for i in range(len(response_ids))]
            num_calls = [0] * len(response_ids)
            timeouts = [False] * len(response_ids)
            tool_errors = [False] * len(response_ids)
            tool_outputs = [""] * len(response_ids)
            tool_runtimes = [0.0] * len(response_ids)
            tool_calleds = [False] * len(response_ids)

        infos = [num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds]
        decoded_texts: List[str] = []  # not used on HF path
        return response_ids, finish_reasons, masks, infos, decoded_texts

    async def compute_rewards(
        self,
        responses: List[torch.Tensor],
        decoded_responses: List[str],
        ground_truths: List[Union[str, List[str]]],
        datasets: List[str],
        finish_reasons: List[str],
        infos: List[List[int]],
        queries: Optional[List[str]] = None,
        original_sources: Optional[List[str]] = None,
    ) -> tuple:
        """Compute rewards for generated responses"""

        num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = infos
        good_outputs = [
            len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
            for i in range(len(tool_outputs))
        ]

        scores = [0] * len(decoded_responses)
        metrics = {}

        dataset_scores = defaultdict(list)
        dataset_indices = defaultdict(list)

        sources_to_use = original_sources if original_sources is not None else datasets
        for i, dataset_source in enumerate(sources_to_use):
            dataset_indices[dataset_source].append(i)

        # Format rewards
        if self.args.apply_r1_style_format_reward:
            with Timer("Computing format rewards"):
                format_scores = soft_format_reward_func(decoded_responses, self.args.r1_style_format_reward)
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]
                metrics["format_scores"] = np.array(format_scores).mean()
                for ds, indices in dataset_indices.items():
                    ds_format_scores = [format_scores[i] for i in indices]
                    if ds_format_scores:
                        display_name = ds.split('/')[-1]
                        metrics[f"{display_name}/format_scores"] = np.array(ds_format_scores).mean()

        # Verifiable rewards
        if self.args.apply_verifiable_reward:
            with Timer("Computing verifiable rewards"):
                verifiable_rewards, per_func_rewards, additional_metrics = await apply_verifiable_reward(
                    self.reward_fn_mapping,
                    responses,
                    decoded_responses,
                    ground_truths,
                    datasets,
                    reward_mult=self.args.verification_reward,
                    queries=queries,
                )

                if len(verifiable_rewards) != len(scores):
                    raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")

                for i in range(len(verifiable_rewards)):
                    if not self.args.only_reward_good_outputs or (good_outputs[i] and self.args.only_reward_good_outputs):
                        if self.args.apply_r1_style_format_reward and self.args.additive_format_reward:
                            scores[i] = verifiable_rewards[i] + scores[i]
                        elif self.args.apply_r1_style_format_reward and not self.args.additive_format_reward:
                            scores[i] = verifiable_rewards[i] if format_scores[i] == 1 else 0
                        else:
                            scores[i] = verifiable_rewards[i]

                np_verifiable = np.array(verifiable_rewards)
                metrics["avg_score"] = np_verifiable.mean()

                for ds, indices in dataset_indices.items():
                    ds_ver = [verifiable_rewards[i] for i in indices]
                    if ds_ver:
                        display_name = ds.split('/')[-1]
                        ds_np = np.array(ds_ver)
                        metrics[f"{display_name}/avg_score"] = ds_np.mean()

                per_func_lists = defaultdict(list)
                for reward_dict in per_func_rewards:
                    for key, value in reward_dict.items():
                        per_func_lists[key].append(value)
                for key, value in per_func_lists.items():
                    np_value = np.array(value)
                    metrics[f"{key}_reward"] = np_value.mean()

                additional_metrics_lists = defaultdict(list)
                for addm in additional_metrics:
                    for key, value in addm.items():
                        additional_metrics_lists[key].append(value)
                for key, values in additional_metrics_lists.items():
                    if values:
                        np_value = np.array(values)
                        metrics[f"{key}"] = np_value.mean()

                for ds, indices in dataset_indices.items():
                    display_name = ds.split('/')[-1]
                    ds_add = defaultdict(list)
                    for i in indices:
                        if i < len(additional_metrics):
                            for key, value in additional_metrics[i].items():
                                ds_add[key].append(value)
                    for key, values in ds_add.items():
                        if values:
                            np_value = np.array(values)
                            metrics[f"{display_name}/{key}"] = np_value.mean()

        if self.args.non_stop_penalty:
            with Timer("Applying non-stop penalty"):
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = self.args.non_stop_penalty_value

        for ds, indices in dataset_indices.items():
            display_name = ds.split('/')[-1]
            ds_scores = [scores[i] for i in indices]
            if ds_scores:
                metrics[f"{display_name}/scores"] = np.array(ds_scores).mean()

        return scores, metrics

    def check_existing_runs(self, dataset_output_dir: str) -> int:
        if not os.path.exists(dataset_output_dir):
            return 0
        existing_runs = 0
        for filename in os.listdir(dataset_output_dir):
            if filename.startswith("run_") and filename.endswith("_detailed.csv"):
                try:
                    run_id = int(filename.split("_")[1])
                    existing_runs = max(existing_runs, run_id + 1)
                except (ValueError, IndexError):
                    continue
        return existing_runs

    def load_existing_results(self, num_existing_runs: int, dataset_output_dir: str) -> List[Dict]:
        existing_results = []
        for run_id in range(num_existing_runs):
            summary_csv_path = os.path.join(dataset_output_dir, f"run_{run_id}_summary.csv")
            if os.path.exists(summary_csv_path):
                summary_df = pd.read_csv(summary_csv_path)
                overall_row = summary_df[summary_df["dataset"] == "OVERALL"].iloc[0]
                result = {
                    "run_id": run_id,
                    "metrics": {},
                    "stop_rate": 1.0 - overall_row["tool_timeout_rate"] - overall_row["tool_error_rate"],
                    "avg_sequence_length": float(overall_row.get("avg_sequence_length", 0.0)),
                    "total_samples": int(overall_row["total_samples"]),
                }
                metrics_columns = [
                    "avg_score",
                    "manufactoria_all_pass",
                    "manufactoria_pass_rate",
                    "avg_tool_calls",
                    "tool_timeout_rate",
                    "tool_error_rate",
                ]
                for column in metrics_columns:
                    if column in overall_row:
                        result["metrics"][column] = overall_row[column]
                for _, row in summary_df.iterrows():
                    if row["dataset"] != "OVERALL":
                        dataset_name = row["dataset"].split('/')[-1]
                        for column in metrics_columns:
                            if column in row:
                                result["metrics"][f"{dataset_name}/{column}"] = row[column]
                                if column == "avg_score":
                                    result["metrics"][f"{dataset_name}/scores"] = row[column]
                existing_results.append(result)
            else:
                print(f"⚠️  Warning: Could not find summary file for run {run_id}")
        return existing_results

    async def run_evaluation(self) -> Dict:
        # breakpoint()
        print(f"🎯 Starting evaluation with {self.args.num_runs} runs for {len(self.dataset_configs)} datasets")
        all_dataset_results = {}
        for dataset_config in self.dataset_configs:
            dataset_name = dataset_config["dataset_name"]
            folder_name = dataset_config["folder_name"]
            output_dir = dataset_config["output_dir"]

            print(f"\n📂 Processing dataset: {dataset_name}")
            print(f"📁 Folder: {folder_name}")
            print(f"📍 Output: {output_dir}")

            existing_runs = self.check_existing_runs(output_dir)
            if existing_runs > 0:
                print(f"📁 Found {existing_runs} existing runs for {dataset_name}")
                if existing_runs >= self.args.num_runs:
                    print(f"✅ Already have {existing_runs} runs (requested {self.args.num_runs}). Skipping.")
                    existing_results = self.load_existing_results(existing_runs, output_dir)
                    aggregated = self.aggregate_results(existing_results)
                    all_dataset_results[dataset_name] = aggregated
                    continue
                else:
                    remaining_runs = self.args.num_runs - existing_runs
                    print(f"🔄 Need {remaining_runs} more runs to reach {self.args.num_runs} total")
                    start_run_id = existing_runs
            else:
                print(f"🎯 Starting fresh evaluation with {self.args.num_runs} runs")
                start_run_id = 0
                remaining_runs = self.args.num_runs

            dataset_results = await self.run_single_dataset_evaluation(
                dataset_config, existing_runs, start_run_id, remaining_runs
            )
            all_dataset_results[dataset_name] = dataset_results

        return self.create_combined_summary(all_dataset_results)

    async def run_single_dataset_evaluation(
        self,
        dataset_config: Dict,
        existing_runs: int,
        start_run_id: int,
        remaining_runs: int
    ) -> Dict:
        dataset_name = dataset_config["dataset_name"]
        output_dir = dataset_config["output_dir"]

        all_results = []
        if existing_runs > 0:
            existing_results = self.load_existing_results(existing_runs, output_dir)
            all_results.extend(existing_results)

        single_dataset_eval = await self.create_single_dataset_eval(dataset_config)

        for i in range(remaining_runs):
            run_id = start_run_id + i
            result = await self.run_single_evaluation_for_dataset(run_id, single_dataset_eval, dataset_name, output_dir)
            all_results.append(result)

            metrics = result["metrics"]
            print(f"  Run {run_id + 1} Summary:")
            print(f"    Average Score: {metrics.get('avg_score', 0.0):.3f}")
            if 'manufactoria_all_pass' in metrics:
                print(f"    Manufactoria All Pass: {metrics['manufactoria_all_pass']:.3f}")
            if 'manufactoria_pass_rate' in metrics:
                print(f"    Manufactoria Pass Rate: {metrics['manufactoria_pass_rate']:.3f}")
            print(f"    Stop Rate: {result['stop_rate']:.3f}")

        aggregated = self.aggregate_results(all_results)
        self.save_dataset_results(aggregated, output_dir, dataset_config["folder_name"])
        return aggregated

    async def create_single_dataset_eval(self, dataset_config: Dict):
        single_dataset_mixer = [dataset_config["dataset_name"], dataset_config["frac_or_num_samples"]]
        tc = self.tokenizer_config
        eval_transform_fn_args = [{}, {"need_contain_labels": True}]
        single_eval_dataset = get_cached_dataset_tulu(
            single_dataset_mixer,
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
        dataset_name = dataset_config["dataset_name"]
        sources = [dataset_name] * len(single_eval_dataset)
        single_eval_dataset = single_eval_dataset.add_column("__dataset_source__", sources)
        return single_eval_dataset

    async def run_single_evaluation_for_dataset(self, run_id: int, eval_dataset, dataset_name: str, output_dir: str = None) -> Dict:
        print(f"  🧪 Running evaluation {run_id + 1}/{self.args.num_runs} for {dataset_name}")

        eval_dataset_subset = eval_dataset

        # Inputs & labels
        eval_ground_truths = eval_dataset_subset[GROUND_TRUTHS_KEY]
        eval_dataset_names = eval_dataset_subset[DATASET_SOURCE_KEY]
        eval_original_sources = eval_dataset_subset["__dataset_source__"]

        # Messages if present
        eval_messages = None
        if "messages" in eval_dataset_subset.column_names:
            eval_messages = eval_dataset_subset["messages"]

        # Prepare prompts (IDs or text) depending on mode
        if self.use_api and self.args.api_config.no_tokenizer:
            eval_prompts_text = self._extract_prompts_as_text(eval_dataset_subset)
            eval_prompt_token_ids = None
        else:
            eval_prompt_token_ids = eval_dataset_subset[INPUT_IDS_PROMPT_KEY]
            eval_prompts_text = None

        # Generate
        with Timer(f"Generating responses for {dataset_name}"):
            response_ids, finish_reasons, masks, infos, decoded_texts = await self.generate_responses(
            prompt_token_ids=eval_prompt_token_ids,
            prompts_text=eval_prompts_text,
            run_id=run_id)

            # response_ids, finish_reasons, masks, infos, decoded_texts = self.generate_responses(
            #     prompt_token_ids=eval_prompt_token_ids,
            #     prompts_text=eval_prompts_text,
            #     run_id=run_id
            # )

        # Build decoded_responses & decoded_prompts safely
        if self.use_api and self.args.api_config.no_tokenizer:
            decoded_responses = decoded_texts
            decoded_prompts = eval_prompts_text
        else:
            decoded_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            decoded_prompts = self.tokenizer.batch_decode(eval_prompt_token_ids, skip_special_tokens=True)

        # Rewards
        with Timer(f"Computing rewards for {dataset_name}"):
            scores, metrics = await self.compute_rewards(
                response_ids,
                decoded_responses,
                eval_ground_truths,
                eval_dataset_names,
                finish_reasons,
                infos,
                original_sources=eval_original_sources,
            )

        # Metrics
        sequence_lengths = np.array([len(r) for r in response_ids])
        stop_rate = sum(int(fr == "stop") for fr in finish_reasons) / max(1, len(finish_reasons))

        results = {
            "run_id": run_id,
            "metrics": metrics,
            "stop_rate": float(stop_rate),
            "avg_sequence_length": float(sequence_lengths.mean()) if len(sequence_lengths) else 0.0,
            "total_samples": len(decoded_responses),
        }

        if self.args.save_results:
            detailed_results = {
                "responses": decoded_responses,
                "prompts": decoded_prompts,
                "ground_truths": eval_ground_truths,
                "scores": scores,
                "finish_reasons": finish_reasons,
                "datasets": eval_original_sources,
                "sequence_lengths": sequence_lengths.tolist(),
                "infos": infos,
                "messages": eval_messages,
            }
            results["detailed"] = detailed_results
            if output_dir:
                self.save_single_run_results(results, output_dir)

        return results

    def save_single_run_results(self, run_result: Dict, output_dir: str):
        if not self.args.save_results or "detailed" not in run_result:
            return

        os.makedirs(output_dir, exist_ok=True)

        run_id = run_result["run_id"]
        detailed = run_result["detailed"]
        run_metrics = run_result["metrics"]

        num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = detailed["infos"]

        failure_reasons = []
        for i in range(len(detailed["responses"])):
            reasons = []
            if detailed["finish_reasons"][i] != "stop":
                reasons.append(f"non_stop_finish({detailed['finish_reasons'][i]})")
            if timeouts[i]:
                reasons.append("tool_timeout")
            if tool_errors[i]:
                reasons.append("tool_error")
            if detailed["scores"][i] <= 0:
                reasons.append("incorrect_solution")
            failure_reasons.append("; ".join(reasons) if reasons else "none")

        # prompt text extraction is already handled earlier; use what's saved
        prompt_texts = detailed["prompts"]

        df_data = {
            "prompt": prompt_texts,
            "response": detailed["responses"],
            "ground_truth": detailed["ground_truths"],
            "score": detailed["scores"],
            "finish_reason": detailed["finish_reasons"],
            "failure_reason": failure_reasons,
            "dataset_source": detailed["datasets"],
            "sequence_length": detailed["sequence_lengths"],
            "num_tool_calls": num_calls,
            "tool_timeout": timeouts,
            "tool_error": tool_errors,
            "tool_output": tool_outputs,
            "tool_runtime": tool_runtimes,
            "tool_called": tool_calleds,
        }

        available_metrics = ["manufactoria_all_pass", "manufactoria_pass_rate", "format_scores"]
        for metric in available_metrics:
            if metric in run_metrics:
                df_data[metric] = [run_metrics[metric]] * len(detailed["responses"])

        df = pd.DataFrame(df_data)
        csv_path = os.path.join(output_dir, f"run_{run_id}_detailed.csv")
        df.to_csv(csv_path, index=False)

        dataset_summary = []
        unique_datasets = df["dataset_source"].unique()
        for dataset in unique_datasets:
            dataset_df = df[df["dataset_source"] == dataset]
            summary_row = {
                "dataset": dataset,
                "total_samples": len(dataset_df),
                "avg_score": dataset_df["score"].mean(),
                "avg_tool_calls": dataset_df["num_tool_calls"].mean(),
                "tool_timeout_rate": dataset_df["tool_timeout"].mean(),
                "tool_error_rate": dataset_df["tool_error"].mean(),
                "avg_sequence_length": dataset_df["sequence_length"].mean(),
            }
            for metric in available_metrics:
                if metric in df.columns:
                    summary_row[metric] = dataset_df[metric].mean()
            dataset_summary.append(summary_row)

        summary_df = pd.DataFrame(dataset_summary)

        overall_summary = {
            "dataset": "OVERALL",
            "total_samples": len(df),
            "avg_score": df["score"].mean(),
            "avg_tool_calls": df["num_tool_calls"].mean(),
            "tool_timeout_rate": df["tool_timeout"].mean(),
            "tool_error_rate": df["tool_error"].mean(),
            "avg_sequence_length": df["sequence_length"].mean(),
        }
        for metric in available_metrics:
            if metric in df.columns:
                overall_summary[metric] = df[metric].mean()

        summary_df = pd.concat([summary_df, pd.DataFrame([overall_summary])], ignore_index=True)
        summary_csv_path = os.path.join(output_dir, f"run_{run_id}_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        print(f"💾 Run {run_id} results saved to {output_dir}")

    def create_combined_summary(self, all_dataset_results: Dict) -> Dict:
        combined_summary = {
            "datasets": list(all_dataset_results.keys()),
            "num_datasets": len(all_dataset_results),
            "per_dataset_results": all_dataset_results,
            "combined_metrics": {}
        }
        all_metrics = defaultdict(list)
        for dataset_name, dataset_results in all_dataset_results.items():
            for metric_name, metric_stats in dataset_results["metrics"].items():
                if isinstance(metric_stats, dict) and "mean" in metric_stats:
                    all_metrics[metric_name].append(metric_stats["mean"])
        for metric_name, values in all_metrics.items():
            if values:
                combined_summary["combined_metrics"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "datasets": list(all_dataset_results.keys())
                }
        return combined_summary

    def aggregate_results(self, all_results: List[Dict]) -> Dict:
        print("📊 Aggregating results across runs...")
        all_metrics = defaultdict(list)
        for result in all_results:
            for key, value in result["metrics"].items():
                all_metrics[key].append(value)

        aggregated_metrics = {}
        for key, values in all_metrics.items():
            values = np.array(values)
            aggregated_metrics[key] = {
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            }

        stop_rates = [r["stop_rate"] for r in all_results]
        seq_lengths = [r["avg_sequence_length"] for r in all_results]

        aggregated = {
            "num_runs": len(all_results),
            "metrics": aggregated_metrics,
            "stop_rate": {
                "mean": np.mean(stop_rates) if stop_rates else 0.0,
                "std": np.std(stop_rates) if stop_rates else 0.0,
            },
            "avg_sequence_length": {
                "mean": np.mean(seq_lengths) if seq_lengths else 0.0,
                "std": np.std(seq_lengths) if seq_lengths else 0.0,
            },
            "individual_runs": all_results,
        }
        return aggregated

    def save_dataset_results(self, results: Dict, output_dir: str, folder_name: str):
        if not self.args.save_results:
            return
        os.makedirs(output_dir, exist_ok=True)

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        metrics_results = {
            "num_runs": results["num_runs"],
            "metrics": results["metrics"],
            "stop_rate": results["stop_rate"],
            "avg_sequence_length": results["avg_sequence_length"],
        }
        metrics_results["individual_runs"] = []
        for run_result in results["individual_runs"]:
            run_metrics = {
                "run_id": run_result["run_id"],
                "metrics": run_result["metrics"],
                "stop_rate": run_result["stop_rate"],
                "avg_sequence_length": run_result["avg_sequence_length"],
                "total_samples": run_result["total_samples"],
            }
            metrics_results["individual_runs"].append(run_metrics)

        with open(os.path.join(output_dir, "aggregated_metrics.json"), "w") as f:
            json.dump(convert_numpy(metrics_results), f, indent=2)

        print(f"💾 Aggregated results saved to {output_dir}")
        print(f"  📂 Dataset folder: {folder_name}")

    def save_combined_results(self, combined_results: Dict):
        if not self.args.save_results:
            return
        os.makedirs(self.args.output_dir, exist_ok=True)

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        with open(os.path.join(self.args.output_dir, "combined_summary.json"), "w") as f:
            json.dump(convert_numpy(combined_results), f, indent=2)

        print(f"💾 Combined summary saved to {self.args.output_dir}/combined_summary.json")


def print_results_summary(results: Dict):
    print("\n" + "="*60)
    print("🎯 EVALUATION RESULTS SUMMARY")
    print("="*60)

    print(f"Number of runs: {results['num_runs']}")
    print(f"Samples per run: {results['individual_runs'][0]['total_samples']}")
    print()
    print("📊 KEY METRICS:")
    key_metrics = ["avg_score", "format_scores"]
    for metric in key_metrics:
        if metric in results["metrics"]:
            stats = results["metrics"][metric]
            print(f"  {metric}:")
            print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    manufactoria_metrics = ["manufactoria_all_pass", "manufactoria_pass_rate"]
    manufactoria_found = False
    for metric in manufactoria_metrics:
        if metric in results["metrics"]:
            if not manufactoria_found:
                print("\n🏭 MANUFACTORIA METRICS:")
                manufactoria_found = True
            stats = results["metrics"][metric]
            display_name = metric.replace("manufactoria_", "").replace("_", " ").title()
            print(f"  {display_name}:")
            print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    print(f"\n🛑 Stop Rate: {results['stop_rate']['mean']:.4f} ± {results['stop_rate']['std']:.4f}")
    print(f"📏 Avg Sequence Length: {results['avg_sequence_length']['mean']:.1f} ± {results['avg_sequence_length']['std']:.1f}")

    print("\n🔍 PER-DATASET BREAKDOWN:")
    datasets = set()
    for metric_name in results["metrics"].keys():
        if "/" in metric_name:
            dataset_name = metric_name.split("/")[0]
            datasets.add(dataset_name)
    for dataset_name in sorted(datasets):
        print(f"\n  📁 {dataset_name}:")
        reward_metric = f"{dataset_name}/avg_score"
        if reward_metric in results["metrics"]:
            stats = results["metrics"][reward_metric]
            print(f"    Average Score: {stats['mean']:.4f} ± {stats['std']:.4f}")
        for manufactoria_metric in ["manufactoria_all_pass", "manufactoria_pass_rate"]:
            full_metric = f"{dataset_name}/{manufactoria_metric}"
            if full_metric in results["metrics"]:
                stats = results["metrics"][full_metric]
                display_name = manufactoria_metric.replace("manufactoria_", "").replace("_", " ").title()
                print(f"    {display_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("="*60)


def print_combined_results_summary(results: Dict):
    print("\n" + "="*80)
    print("🎯 COMBINED EVALUATION RESULTS SUMMARY")
    print("="*80)

    print(f"Number of datasets: {results['num_datasets']}")
    print(f"Datasets evaluated: {', '.join(results['datasets'])}")
    print()
    print("📊 COMBINED KEY METRICS:")
    key_metrics = ["avg_score", "format_scores"]
    for metric in key_metrics:
        if metric in results["combined_metrics"]:
            stats = results["combined_metrics"][metric]
            print(f"  {metric}:")
            print(f"    Mean across datasets: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\n🔍 PER-DATASET BREAKDOWN:")
    for dataset_name in results["datasets"]:
        if dataset_name in results["per_dataset_results"]:
            dataset_results = results["per_dataset_results"][dataset_name]
            print(f"\n  📁 {dataset_name}:")
            print(f"    Runs completed: {dataset_results['num_runs']}")
            if "avg_score" in dataset_results["metrics"]:
                reward_stats = dataset_results["metrics"]["avg_score"]
                print(f"    Average Score: {reward_stats['mean']:.4f} ± {reward_stats['std']:.4f}")
            for manufactoria_metric in ["manufactoria_all_pass", "manufactoria_pass_rate"]:
                if manufactoria_metric in dataset_results["metrics"]:
                    stats = dataset_results["metrics"][manufactoria_metric]
                    display_name = manufactoria_metric.replace("manufactoria_", "").replace("_", " ").title()
                    print(f"    {display_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("="*80)


async def main():
    """Main evaluation function"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = ArgumentParserPlus((EvalArgs, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()

    print("DEBUG:", "model_type=", args.model_type, "api_config.model_type=", args.api_config.model_type)

    # Back-compat: if user passed --model_type anthropic/openai/deepseek,
    # automatically treat it as API mode unless api_config.model_type was explicitly set.
    _api_providers = {"anthropic", "openai", "deepseek"}
    if (args.api_config.model_type == "huggingface") and (args.model_type in _api_providers):
        args.api_config.model_type = args.model_type
        print(f"DEBUG: switching to API mode via model_type={args.model_type}")

    # Hard-disable CUDA for API mode as an early guard
    if args.api_config.model_type != "huggingface":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    print("🚀 Starting Code Evaluation")
    print(f"Model: {model_config.model_name_or_path}")
    print(f"Dataset: {args.dataset_mixer_eval_list}")
    print(f"Number of runs: {args.num_runs}")

    evaluator = CodeEvaluator(args, tokenizer_config, model_config)
    print(f"📂 Will evaluate {len(evaluator.dataset_configs)} datasets:")
    for config in evaluator.dataset_configs:
        print(f"  - {config['dataset_name']} → {config['folder_name']}")
    evaluator.setup()

    print("Sample allocation per dataset:")
    for config in evaluator.dataset_configs:
        frac_or_num = config['frac_or_num_samples']
        if "." in frac_or_num:
            print(f"  - {config['dataset_name']}: {float(frac_or_num)*100:.1f}% of dataset")
        else:
            print(f"  - {config['dataset_name']}: {frac_or_num} samples")
    print()

    results = await evaluator.run_evaluation()

    evaluator.save_combined_results(results)

    print_combined_results_summary(results)

    if ray.is_initialized():
        ray.shutdown()
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
