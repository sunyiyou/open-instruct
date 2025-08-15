#!/usr/bin/env python3
"""
Clean evaluation script extracted from grpo_fast_code.py
Runs evaluation on a given model with specified parameters and outputs results for n runs.
"""

import asyncio
import csv
import json
import os
import sys
import ray

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from vllm import SamplingParams


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
class EvalArgs:
    """Configuration for evaluation runs"""
    
    # Dataset configuration
    dataset_mixer_eval_list: List[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    """A list of datasets (local or HF) to sample from for evaluation."""
    dataset_mixer_eval_list_splits: List[str] = field(default_factory=lambda: ["test"])
    """The dataset splits to use for evaluation"""
    dataset_transform_fn: List[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_filter_v1"])
    """The list of transform functions to apply to the dataset."""
    dataset_cache_mode: str = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_eval_hash: Optional[str] = None
    """The hash of the dataset configuration for evaluation."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    use_last_n_eval_samples: bool = True
    """Whether to use the last N samples from each evaluation dataset instead of the first N."""
    
    # Evaluation configuration
    num_runs: int = 1
    """Number of evaluation runs to perform"""
    seed: int = 42
    """Random seed for reproducibility"""
    
    # Generation configuration
    response_length: int = 256
    """Maximum response length"""
    temperature: float = 0.0
    """Sampling temperature (0.0 for greedy)"""
    vllm_top_p: float = 1.0
    """Top-p sampling parameter"""
    stop_strings: Optional[List[str]] = None
    """List of strings that stop generation"""
    
    # VLLM configuration
    vllm_num_engines: int = 1
    """Number of VLLM engines to use"""
    vllm_tensor_parallel_size: int = 1
    """Tensor parallel size for VLLM"""
    vllm_enforce_eager: bool = False
    """Whether to enforce eager execution in VLLM"""
    vllm_enable_prefix_caching: bool = False
    """Whether to enable prefix caching in VLLM"""
    vllm_gpu_memory_utilization: float = 0.9
    """GPU memory utilization for VLLM"""
    single_gpu_mode: bool = False
    """Whether to use single GPU mode"""
    
    # Reward configuration
    apply_r1_style_format_reward: bool = True
    """Whether to apply R1-style format rewards"""
    r1_style_format_reward: float = 1.0
    """R1-style format reward value"""
    apply_verifiable_reward: bool = True
    """Whether to apply verifiable rewards"""
    verification_reward: float = 1.0
    """Verification reward multiplier"""
    additive_format_reward: bool = True
    """Whether format reward is additive"""
    only_reward_good_outputs: bool = False
    """Whether to only reward outputs that executed successfully"""
    non_stop_penalty: bool = False
    """Whether to apply penalty for non-stop finish reasons"""
    non_stop_penalty_value: float = -1.0
    """Penalty value for non-stop finish reasons"""
    
    # Tool configuration
    tools: Optional[List[str]] = None
    """List of tools to use (e.g., ['code'])"""
    max_tool_calls: List[int] = field(default_factory=lambda: [5])
    """Maximum number of tool calls"""
    code_tool_api_endpoint: Optional[str] = None
    """API endpoint for code tool"""
    search_api_endpoint: Optional[str] = None
    """API endpoint for search tool"""
    number_documents_to_search: int = 10
    """Number of documents to search"""
    
    # Verifier configuration
    code_api_url: Optional[str] = None
    """API URL for code verification"""
    code_max_execution_time: float = 1.0
    """Maximum execution time for code verification"""
    manufactoria_api_url: Optional[str] = None
    """API URL for Manufactoria verification"""
    manufactoria_max_execution_time: float = 1.0
    """Maximum execution time for Manufactoria verification"""
    
    # LLM judge configuration
    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    """Model to use for LLM judge"""
    llm_judge_max_tokens: int = 2048
    """Maximum tokens for LLM judge"""
    llm_judge_temperature: float = 1.0
    """Temperature for LLM judge"""
    llm_judge_timeout: int = 60
    """Timeout for LLM judge"""
    
    # Output configuration
    output_dir: str = "eval_results"
    """Directory to save evaluation results"""
    save_results: bool = True
    """Whether to save detailed results"""
    hf_entity: Optional[str] = None
    """HuggingFace entity for dataset access"""


class CodeEvaluator:
    def __init__(self, args: EvalArgs, tokenizer_config: TokenizerConfig, model_config: ModelConfig):
        self.args = args
        self.tokenizer_config = tokenizer_config
        self.model_config = model_config
        self.tokenizer = None
        
        # Ensure tokenizer_config has valid values before any use
        if self.tokenizer_config.tokenizer_name_or_path is None:
            self.tokenizer_config.tokenizer_name_or_path = self.model_config.model_name_or_path
            self.tokenizer_config.tokenizer_revision = self.model_config.model_revision
        self.vllm_engines = None
        self.reward_fn_mapping = None
        self.eval_dataset = None
        
        # Parse dataset configurations for individual processing
        self.dataset_configs = self.parse_dataset_configs()
    
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
    
    def clean_dataset_name(self, dataset_name: str) -> str:
        """Clean dataset name for use as folder name"""
        clean_name = dataset_name.replace("/", "_").replace("-", "_")
        if "_" in clean_name:
            clean_name = clean_name.split("_")[-1]
        return clean_name
        
    def setup(self):
        """Initialize all components needed for evaluation"""
        print("🔧 Setting up evaluator...")
        
        # Initialize tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name_or_path,
            revision=self.model_config.model_revision,
            trust_remote_code=self.tokenizer_config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
        
        # Initialize VLLM engines
        print("🚀 Initializing VLLM engines...")
        max_len = 512 + self.args.response_length  # Rough estimate
        
        # Setup tools if specified
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
        
        # Initialize reward functions
        print("🏆 Setting up reward functions...")
        self.reward_fn_mapping = build_all_verifiers(self.args)
        
        print("✅ Setup complete!")
        
    def generate_responses(self, prompt_token_ids: List[List[int]]) -> tuple:
        """Generate responses using VLLM engines"""
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
        
        # Track per-dataset metrics using original sources if provided
        dataset_scores = defaultdict(list)
        dataset_indices = defaultdict(list)
        
        # Use original_sources for per-dataset breakdown if available, otherwise fall back to datasets
        sources_to_use = original_sources if original_sources is not None else datasets
        for i, dataset_source in enumerate(sources_to_use):
            dataset_indices[dataset_source].append(i)

        # Apply format rewards
        if self.args.apply_r1_style_format_reward:
            with Timer("Computing format rewards"):
                format_scores = soft_format_reward_func(decoded_responses, self.args.r1_style_format_reward)
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]
                metrics["format_scores"] = np.array(format_scores).mean()
                
                # Per-dataset format scores
                for ds, indices in dataset_indices.items():
                    ds_format_scores = [format_scores[i] for i in indices]
                    if ds_format_scores:
                        display_name = ds.split('/')[-1]
                        metrics[f"{display_name}/format_scores"] = np.array(ds_format_scores).mean()

        # Apply verifiable rewards
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
                            
                np_verifiable_rewards = np.array(verifiable_rewards)
                metrics["avg_score"] = np_verifiable_rewards.mean()

                # Per-dataset verifiable rewards
                for ds, indices in dataset_indices.items():
                    ds_verifiable_rewards = [verifiable_rewards[i] for i in indices]
                    if ds_verifiable_rewards:
                        display_name = ds.split('/')[-1]
                        ds_np_rewards = np.array(ds_verifiable_rewards)
                        metrics[f"{display_name}/avg_score"] = ds_np_rewards.mean()

                # Per-function rewards
                per_func_lists = defaultdict(list)
                for reward_dict in per_func_rewards:
                    for key, value in reward_dict.items():
                        per_func_lists[key].append(value)
                        
                for key, value in per_func_lists.items():
                    np_value = np.array(value)
                    metrics[f"{key}_reward"] = np_value.mean()
                
                # Log global additional metrics (e.g., manufactoria all_pass vs pass_rate)
                additional_metrics_lists = defaultdict(list)
                for additional_metrics_dict in additional_metrics:
                    for key, value in additional_metrics_dict.items():
                        additional_metrics_lists[key].append(value)
                
                for key, values in additional_metrics_lists.items():
                    if values:  # Only log if we have values
                        np_value = np.array(values)
                        metrics[f"{key}"] = np_value.mean()
                
                # Log additional metrics per dataset (e.g., manufactoria all_pass vs pass_rate)
                for ds, indices in dataset_indices.items():
                    display_name = ds.split('/')[-1]
                    ds_additional_metrics = defaultdict(list)
                    
                    # Collect additional metrics for this dataset
                    for i in indices:
                        if i < len(additional_metrics):
                            for key, value in additional_metrics[i].items():
                                ds_additional_metrics[key].append(value)
                    
                    # Log per-dataset additional metrics
                    for key, values in ds_additional_metrics.items():
                        if values:  # Only log if we have values
                            np_value = np.array(values)
                            metrics[f"{display_name}/{key}"] = np_value.mean()

        # Apply non-stop penalty
        if self.args.non_stop_penalty:
            with Timer("Applying non-stop penalty"):
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = self.args.non_stop_penalty_value

        # Final per-dataset scores
        for ds, indices in dataset_indices.items():
            display_name = ds.split('/')[-1]
            ds_scores = [scores[i] for i in indices]
            if ds_scores:
                metrics[f"{display_name}/scores"] = np.array(ds_scores).mean()

        # Return per-sample additional_metrics if available
        per_sample_additional_metrics = None
        if self.args.apply_verifiable_reward and 'additional_metrics' in locals():
            per_sample_additional_metrics = additional_metrics
        
        return scores, metrics, per_sample_additional_metrics
    

    
    def check_existing_runs(self, dataset_output_dir: str) -> int:
        """Check how many evaluation runs already exist in the dataset-specific output directory"""
        if not os.path.exists(dataset_output_dir):
            return 0
        
        existing_runs = 0
        # Check for existing detailed CSV files (run_N_detailed.csv pattern)
        for filename in os.listdir(dataset_output_dir):
            if filename.startswith("run_") and filename.endswith("_detailed.csv"):
                try:
                    # Extract run ID from filename
                    run_id = int(filename.split("_")[1])
                    existing_runs = max(existing_runs, run_id + 1)
                except (ValueError, IndexError):
                    continue
        
        return existing_runs

    def load_existing_results(self, num_existing_runs: int, dataset_output_dir: str) -> List[Dict]:
        """Load existing evaluation results from CSV files"""
        existing_results = []
        
        for run_id in range(num_existing_runs):
            # Load metrics from summary CSV if available
            summary_csv_path = os.path.join(dataset_output_dir, f"run_{run_id}_summary.csv")
            if os.path.exists(summary_csv_path):
                summary_df = pd.read_csv(summary_csv_path)
                overall_row = summary_df[summary_df["dataset"] == "OVERALL"].iloc[0]
                
                # Start with basic result structure
                result = {
                    "run_id": run_id,
                    "metrics": {},
                    "stop_rate": 1.0 - overall_row["tool_timeout_rate"] - overall_row["tool_error_rate"],  # Approximation
                    "avg_sequence_length": float(overall_row.get("avg_sequence_length", 0.0)),  # Load from summary CSV
                    "total_samples": int(overall_row["total_samples"]),
                }
                
                # Load ALL available metrics from the overall row
                # List of known CSV columns to load as metrics (excluding avg_sequence_length since it's already handled above)
                metrics_columns = [
                    "avg_score",
                    "manufactoria_all_pass", 
                    "manufactoria_pass_rate",
                    "avg_tool_calls",
                    "tool_timeout_rate",
                    "tool_error_rate",
                ]
                
                # Load overall metrics
                for column in metrics_columns:
                    if column in overall_row:
                        result["metrics"][column] = overall_row[column]
                
                # Load per-dataset metrics
                for _, row in summary_df.iterrows():
                    if row["dataset"] != "OVERALL":
                        dataset_name = row["dataset"].split('/')[-1]
                        
                        # Load all available per-dataset metrics
                        for column in metrics_columns:
                            if column in row:
                                result["metrics"][f"{dataset_name}/{column}"] = row[column]
                                # Also map avg_score to scores for consistency with existing code
                                if column == "avg_score":
                                    result["metrics"][f"{dataset_name}/scores"] = row[column]
                
                existing_results.append(result)
            else:
                print(f"⚠️  Warning: Could not find summary file for run {run_id}")
        
        return existing_results

    async def run_evaluation(self) -> Dict:
        """Run multiple evaluation runs for each dataset separately and aggregate results"""
        print(f"🎯 Starting evaluation with {self.args.num_runs} runs for {len(self.dataset_configs)} datasets")
        
        all_dataset_results = {}
        
        for dataset_config in self.dataset_configs:
            dataset_name = dataset_config["dataset_name"]
            folder_name = dataset_config["folder_name"]
            output_dir = dataset_config["output_dir"]
            
            print(f"\n📂 Processing dataset: {dataset_name}")
            print(f"📁 Folder: {folder_name}")
            print(f"📍 Output: {output_dir}")
            
            # Create dataset-specific evaluation dataset and save original dataset info first
            single_dataset_eval = await self.create_single_dataset_eval(dataset_config)
            self.save_original_dataset_jsonl(single_dataset_eval, output_dir)
            
            # Check for existing runs for this specific dataset
            existing_runs = self.check_existing_runs(output_dir)
            if existing_runs > 0:
                print(f"📁 Found {existing_runs} existing runs for {dataset_name}")
                
                if existing_runs >= self.args.num_runs:
                    print(f"✅ Already have {existing_runs} runs (requested {self.args.num_runs}). Skipping.")
                    # Load existing results and add to aggregate
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
            
            # Run evaluation for this specific dataset
            dataset_results = await self.run_single_dataset_evaluation(
                dataset_config, existing_runs, start_run_id, remaining_runs, single_dataset_eval
            )
            all_dataset_results[dataset_name] = dataset_results
        
        # Create combined summary
        return self.create_combined_summary(all_dataset_results)
    
    async def run_single_dataset_evaluation(
        self, 
        dataset_config: Dict, 
        existing_runs: int, 
        start_run_id: int, 
        remaining_runs: int,
        single_dataset_eval = None
    ) -> Dict:
        """Run evaluation for a single dataset"""
        dataset_name = dataset_config["dataset_name"]
        output_dir = dataset_config["output_dir"]
        
        all_results = []
        
        # Load existing results if any
        if existing_runs > 0:
            existing_results = self.load_existing_results(existing_runs, output_dir)
            all_results.extend(existing_results)
        
        # Use the pre-created dataset (or create it if not provided for backward compatibility)
        if single_dataset_eval is None:
            single_dataset_eval = await self.create_single_dataset_eval(dataset_config)
        
        # Run additional evaluations
        for i in range(remaining_runs):
            run_id = start_run_id + i
            result = await self.run_single_evaluation_for_dataset(run_id, single_dataset_eval, dataset_name, output_dir)
            all_results.append(result)
            
            # Print summary for this run
            metrics = result["metrics"]
            print(f"  Run {run_id + 1} Summary:")
            print(f"    Average Score: {metrics.get('avg_score', 0.0):.3f}")

            # Show both Manufactoria metrics if available
            if 'manufactoria_all_pass' in metrics:
                print(f"    Manufactoria All Pass: {metrics['manufactoria_all_pass']:.3f}")
            if 'manufactoria_pass_rate' in metrics:
                print(f"    Manufactoria Pass Rate: {metrics['manufactoria_pass_rate']:.3f}")
            
            print(f"    Stop Rate: {result['stop_rate']:.3f}")
        
        # Aggregate results for this dataset
        aggregated = self.aggregate_results(all_results)
        
        # Save results for this dataset
        self.save_dataset_results(aggregated, output_dir, dataset_config["folder_name"])
        
        return aggregated
    
    async def create_single_dataset_eval(self, dataset_config: Dict):
        """Create evaluation dataset for a single dataset"""
        # Create a temporary dataset mixer list with just this dataset
        single_dataset_mixer = [dataset_config["dataset_name"], dataset_config["frac_or_num_samples"]]
        
        # Use the same dataset loading logic but for single dataset
        tc = self.tokenizer_config
        eval_transform_fn_args = [
            {},
            {"need_contain_labels": True},
        ]
        
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
        
        # Add dataset source tracking
        dataset_name = dataset_config["dataset_name"]
        sources = [dataset_name] * len(single_eval_dataset)
        single_eval_dataset = single_eval_dataset.add_column("__dataset_source__", sources)
        
        return single_eval_dataset
    
    async def run_single_evaluation_for_dataset(self, run_id: int, eval_dataset, dataset_name: str, output_dir: str = None) -> Dict:
        """Run a single evaluation for a specific dataset"""
        print(f"  🧪 Running evaluation {run_id + 1}/{self.args.num_runs} for {dataset_name}")
        
        # Use all samples from the dataset (already sized according to dataset_mixer_eval_list)
        eval_dataset_subset = eval_dataset
            
        eval_prompt_token_ids = eval_dataset_subset[INPUT_IDS_PROMPT_KEY]
        eval_ground_truths = eval_dataset_subset[GROUND_TRUTHS_KEY]
        eval_dataset_names = eval_dataset_subset[DATASET_SOURCE_KEY]
        eval_original_sources = eval_dataset_subset["__dataset_source__"]
        
        # Extract original messages if available
        eval_messages = None
        if "messages" in eval_dataset_subset.column_names:
            eval_messages = eval_dataset_subset["messages"]
        
        # Generate responses
        with Timer(f"Generating responses for {dataset_name}"):
            response_ids, finish_reasons, masks, infos = self.generate_responses(eval_prompt_token_ids)
        
        # Decode responses
        decoded_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        # Decode prompts for better readability
        decoded_prompts = self.tokenizer.batch_decode(eval_prompt_token_ids, skip_special_tokens=True)
        
        # Compute rewards
        with Timer(f"Computing rewards for {dataset_name}"):
            scores, metrics, per_sample_additional_metrics = await self.compute_rewards(
                response_ids,
                decoded_responses, 
                eval_ground_truths,
                eval_dataset_names,
                finish_reasons,
                infos,
                original_sources=eval_original_sources,
            )
        
        # Calculate additional metrics
        sequence_lengths = np.array([len(response) for response in response_ids])
        stop_rate = sum(int(finish_reason == "stop") for finish_reason in finish_reasons) / len(finish_reasons)
        
        # Compile results
        results = {
            "run_id": run_id,
            "metrics": metrics,
            "stop_rate": float(stop_rate),
            "avg_sequence_length": float(sequence_lengths.mean()),
            "total_samples": len(decoded_responses),
        }
        
        # Save detailed results if requested
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
                "additional_metrics": per_sample_additional_metrics,
            }
            results["detailed"] = detailed_results
            
            # Save individual run results immediately if output_dir is provided
            if output_dir:
                self.save_single_run_results(results, output_dir)
            
        return results
    
    def save_original_dataset_jsonl(self, eval_dataset, output_dir: str):
        """Save the original dataset information as JSONL file"""
        os.makedirs(output_dir, exist_ok=True)
        
        jsonl_path = os.path.join(output_dir, "original_dataset.jsonl")
        
        # Check if JSONL file already exists to avoid overwriting
        if os.path.exists(jsonl_path):
            return
            
        print(f"💾 Saving original dataset information to {jsonl_path}")
        
        with open(jsonl_path, "w") as f:
            for i in range(len(eval_dataset)):
                # Extract original data for this sample
                sample_data = {
                    "index": f"{i}",
                    "dataset_source": eval_dataset["__dataset_source__"][i],
                }
                
                # Extract criteria from messages (the part after the task description)
                if "messages" in eval_dataset.column_names:
                    messages = eval_dataset["messages"][i]
                    criteria = ""
                    if messages and len(messages) > 0:
                        # Look for the user message content
                        for msg in messages:
                            if msg.get("role") == "user" and "content" in msg:
                                content = msg["content"]
                                # Extract the part after the task description
                                task_prefix = "Your task is to design a factory with code with following functionality:\n\n"
                                if task_prefix in content:
                                    criteria = content.split(task_prefix, 1)[1].strip()
                                else:
                                    # Fallback: use the full content if the prefix isn't found
                                    criteria = content.strip()
                                break
                    sample_data["criteria"] = criteria
                
                # Add ground truth
                if GROUND_TRUTHS_KEY in eval_dataset.column_names:
                    sample_data["ground_truth"] = eval_dataset[GROUND_TRUTHS_KEY][i]
                
                # Add dataset source information
                if DATASET_SOURCE_KEY in eval_dataset.column_names:
                    sample_data["dataset"] = eval_dataset[DATASET_SOURCE_KEY][i]
                
                # Add any other original columns that might be useful
                # We want to preserve original metadata like: dataset, difficulty, id, problem_family, name, etc.
                excluded_columns = {
                    "__dataset_source__", "messages", GROUND_TRUTHS_KEY, DATASET_SOURCE_KEY, 
                    INPUT_IDS_PROMPT_KEY, "input_ids", "attention_mask", "labels",
                    "input_ids_chosen", "input_ids_rejected", "labels_chosen", "labels_rejected"
                }
                
                for col in eval_dataset.column_names:
                    if col not in excluded_columns and col not in sample_data:
                        # Keep original metadata columns like dataset, difficulty, id, problem_family, name, etc.
                        try:
                            value = eval_dataset[col][i]
                            # Only include if it's a serializable type
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                sample_data[col] = value
                        except Exception as e:
                            # Skip any columns that can't be serialized
                            print(f"Warning: Skipping column '{col}' for sample {i}: {e}")
                            pass
                
                # Write as JSONL (one JSON object per line)
                json.dump(sample_data, f, ensure_ascii=False)
                f.write("\n")
        
        print(f"✅ Saved {len(eval_dataset)} samples to {jsonl_path}")

    def save_single_run_results(self, run_result: Dict, output_dir: str):
        """Save results for a single evaluation run immediately"""
        if not self.args.save_results or "detailed" not in run_result:
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        run_id = run_result["run_id"]
        detailed = run_result["detailed"]
        run_metrics = run_result["metrics"]
        
        # Extract tool information from infos
        num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = detailed["infos"]
        
        # Create failure reasons
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
        
        # Extract prompt text from messages if available
        prompt_texts = []
        if detailed["messages"] is not None:
            for msgs in detailed["messages"]:
                # Extract the user prompt (typically the last user message or all but assistant)
                user_msgs = [msg["content"] for msg in msgs if msg["role"] == "user"]
                prompt_texts.append(" | ".join(user_msgs))
        else:
            # Fallback to decoded prompts
            prompt_texts = detailed["prompts"]
        
        # Save as CSV with comprehensive information
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
        
        # Add per-sample additional metrics if available
        if detailed.get("additional_metrics") is not None:
            # Extract manufactoria metrics per sample
            manufactoria_all_pass = []
            manufactoria_pass_rate = []
            
            for i, sample_metrics in enumerate(detailed["additional_metrics"]):
                if sample_metrics:
                    # Extract manufactoria metrics (the keys are prefixed with dataset name)
                    all_pass = None
                    pass_rate = None
                    
                    for key, value in sample_metrics.items():
                        if key.endswith("_all_pass"):
                            all_pass = value
                        elif key.endswith("_pass_rate"):
                            pass_rate = value
                    
                    manufactoria_all_pass.append(all_pass)
                    manufactoria_pass_rate.append(pass_rate)
                else:
                    manufactoria_all_pass.append(None)
                    manufactoria_pass_rate.append(None)
            
            # Add the per-sample metrics to the DataFrame
            df_data["manufactoria_all_pass"] = manufactoria_all_pass
            df_data["manufactoria_pass_rate"] = manufactoria_pass_rate
        else:
            # Fallback to global metrics if per-sample metrics are not available
            available_metrics = ["manufactoria_all_pass", "manufactoria_pass_rate"]
            for metric in available_metrics:
                if metric in run_metrics:
                    df_data[metric] = [run_metrics[metric]] * len(detailed["responses"])
        
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(output_dir, f"run_{run_id}_detailed.csv")
        df.to_csv(csv_path, index=False)
        
        # Save a summary CSV with pass rates by dataset
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
                "manufactoria_all_pass": dataset_df["manufactoria_all_pass"].mean(),
                "manufactoria_pass_rate": dataset_df["manufactoria_pass_rate"].mean(),
            }
            
            dataset_summary.append(summary_row)
        
        summary_df = pd.DataFrame(dataset_summary)
        
        # Add overall summary row
        overall_summary = {
            "dataset": "OVERALL",
            "total_samples": len(df),
            "avg_score": df["score"].mean(),
            "avg_tool_calls": df["num_tool_calls"].mean(),
            "tool_timeout_rate": df["tool_timeout"].mean(),
            "tool_error_rate": df["tool_error"].mean(),
            "avg_sequence_length": df["sequence_length"].mean(),
            "manufactoria_all_pass": df["manufactoria_all_pass"].mean(),
            "manufactoria_pass_rate": df["manufactoria_pass_rate"].mean(),
        }
        
        summary_df = pd.concat([summary_df, pd.DataFrame([overall_summary])], ignore_index=True)
        
        summary_csv_path = os.path.join(output_dir, f"run_{run_id}_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        
        print(f"💾 Run {run_id} results saved to {output_dir}")
    
    def create_combined_summary(self, all_dataset_results: Dict) -> Dict:
        """Create a combined summary across all datasets"""
        combined_summary = {
            "datasets": list(all_dataset_results.keys()),
            "num_datasets": len(all_dataset_results),
            "per_dataset_results": all_dataset_results,
            "combined_metrics": {}
        }
        
        # Aggregate metrics across all datasets
        all_metrics = defaultdict(list)
        for dataset_name, dataset_results in all_dataset_results.items():
            for metric_name, metric_stats in dataset_results["metrics"].items():
                if isinstance(metric_stats, dict) and "mean" in metric_stats:
                    all_metrics[metric_name].append(metric_stats["mean"])
        
        # Compute overall statistics
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
        """Aggregate results from multiple runs"""
        print("📊 Aggregating results across runs...")
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        for result in all_results:
            for key, value in result["metrics"].items():
                all_metrics[key].append(value)
        
        # Compute statistics
        aggregated_metrics = {}
        for key, values in all_metrics.items():
            values = np.array(values)
            aggregated_metrics[key] = {
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            }
        
        # Aggregate other metrics
        stop_rates = [r["stop_rate"] for r in all_results]
        seq_lengths = [r["avg_sequence_length"] for r in all_results]
        
        aggregated = {
            "num_runs": len(all_results),
            "metrics": aggregated_metrics,
            "stop_rate": {
                "mean": np.mean(stop_rates),
                "std": np.std(stop_rates),
            },
            "avg_sequence_length": {
                "mean": np.mean(seq_lengths),
                "std": np.std(seq_lengths),
            },
            "individual_runs": all_results,
        }
        
        return aggregated
    
    def save_dataset_results(self, results: Dict, output_dir: str, folder_name: str):
        """Save aggregated evaluation results for a single dataset (individual runs already saved)"""
        if not self.args.save_results:
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
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
            
        # Create aggregated metrics results (without detailed samples since they're already saved)
        metrics_results = {
            "num_runs": results["num_runs"],
            "metrics": results["metrics"],
            "stop_rate": results["stop_rate"],
            "avg_sequence_length": results["avg_sequence_length"],
        }
        
        # Add individual run metrics (without detailed samples)
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
        
        # Save aggregated metrics (without samples)
        with open(os.path.join(output_dir, "aggregated_metrics.json"), "w") as f:
            json.dump(convert_numpy(metrics_results), f, indent=2)
        
        print(f"💾 Aggregated results saved to {output_dir}")
        print(f"  📂 Dataset folder: {folder_name}")
        
    def save_combined_results(self, combined_results: Dict):
        """Save combined results across all datasets"""
        if not self.args.save_results:
            return
            
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
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
        
        # Save combined summary
        with open(os.path.join(self.args.output_dir, "combined_summary.json"), "w") as f:
            json.dump(convert_numpy(combined_results), f, indent=2)
        
        print(f"💾 Combined summary saved to {self.args.output_dir}/combined_summary.json")


def print_results_summary(results: Dict):
    """Print a formatted summary of results"""
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
    
    # Show Manufactoria-specific metrics if available
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
    # Collect dataset names first
    datasets = set()
    for metric_name in results["metrics"].keys():
        if "/" in metric_name:
            dataset_name = metric_name.split("/")[0]
            datasets.add(dataset_name)
    
    for dataset_name in sorted(datasets):
        print(f"\n  📁 {dataset_name}:")
        
        # Show main reward metric
        reward_metric = f"{dataset_name}/avg_score"
        if reward_metric in results["metrics"]:
            stats = results["metrics"][reward_metric]
            print(f"    Average Score: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Show Manufactoria-specific metrics if available
        for manufactoria_metric in ["manufactoria_all_pass", "manufactoria_pass_rate"]:
            full_metric = f"{dataset_name}/{manufactoria_metric}"
            if full_metric in results["metrics"]:
                stats = results["metrics"][full_metric]
                display_name = manufactoria_metric.replace("manufactoria_", "").replace("_", " ").title()
                print(f"    {display_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("="*60)


def print_combined_results_summary(results: Dict):
    """Print a formatted summary of combined results across datasets"""
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
            
            # Show main metrics for this dataset
            if "avg_score" in dataset_results["metrics"]:
                reward_stats = dataset_results["metrics"]["avg_score"]
                print(f"    Average Score: {reward_stats['mean']:.4f} ± {reward_stats['std']:.4f}")
            
            # Show Manufactoria-specific metrics if available
            for manufactoria_metric in ["manufactoria_all_pass", "manufactoria_pass_rate"]:
                if manufactoria_metric in dataset_results["metrics"]:
                    stats = dataset_results["metrics"][manufactoria_metric]
                    display_name = manufactoria_metric.replace("manufactoria_", "").replace("_", " ").title()
                    print(f"    {display_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("="*80)


async def main():
    """Main evaluation function"""
    # Set environment variable to avoid tokenizer warnings
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = ArgumentParserPlus((EvalArgs, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    
    print("🚀 Starting Code Evaluation")
    print(f"Model: {model_config.model_name_or_path}")
    print(f"Dataset: {args.dataset_mixer_eval_list}")
    print(f"Number of runs: {args.num_runs}")
    
    # Initialize evaluator
    evaluator = CodeEvaluator(args, tokenizer_config, model_config)
    print(f"📂 Will evaluate {len(evaluator.dataset_configs)} datasets:")
    for config in evaluator.dataset_configs:
        print(f"  - {config['dataset_name']} → {config['folder_name']}")
    evaluator.setup()
    
    # Print sample information based on dataset configuration
    print("Sample allocation per dataset:")
    for config in evaluator.dataset_configs:
        frac_or_num = config['frac_or_num_samples']
        if "." in frac_or_num:
            print(f"  - {config['dataset_name']}: {float(frac_or_num)*100:.1f}% of dataset")
        else:
            print(f"  - {config['dataset_name']}: {frac_or_num} samples")
    print()
    
    # Ray is already initialized in evaluator.setup()
    
    # Run evaluation
    results = await evaluator.run_evaluation()
    
    # Save combined results
    evaluator.save_combined_results(results)
    
    # Print summary
    print_combined_results_summary(results)
    
    # Cleanup
    ray.shutdown()
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
