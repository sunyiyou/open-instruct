#!/usr/bin/env python3
"""
Flask web application for viewing evaluation results from eval_code.py
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, request, abort, url_for
import numpy as np

app = Flask(__name__)

# Configuration
EVAL_RESULTS_DIR = Path("eval")
DATA_DIR = Path(__file__).parent / EVAL_RESULTS_DIR


class DataLoader:
    """Utility class for loading evaluation results data"""
    
    @staticmethod
    def get_all_models() -> List[Dict[str, Any]]:
        """Get list of all models with their basic info"""
        models = []
        if not DATA_DIR.exists():
            return models
            
        for model_dir in DATA_DIR.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("eval_results_"):
                model_name = model_dir.name.replace("eval_results_", "")
                
                # Count datasets and runs
                datasets = DataLoader.get_datasets_for_model(model_name)
                total_runs = sum(len(d["runs"]) for d in datasets)
                
                # Calculate average score and sequence length across all datasets
                avg_scores = []
                avg_seq_lengths = []
                for dataset in datasets:
                    if dataset["aggregated_metrics"]:
                        avg_score = dataset["aggregated_metrics"]["metrics"].get("avg_score", {}).get("mean", 0)
                        avg_seq_length = dataset["aggregated_metrics"]["metrics"].get("avg_sequence_length", {}).get("mean", 0)
                        avg_scores.append(avg_score)
                        avg_seq_lengths.append(avg_seq_length)
                
                overall_avg_score = np.mean(avg_scores) if avg_scores else 0
                overall_avg_seq_length = np.mean(avg_seq_lengths) if avg_seq_lengths else 0
                
                models.append({
                    "name": model_name,
                    "dir_name": model_dir.name,
                    "num_datasets": len(datasets),
                    "total_runs": total_runs,
                    "avg_score": overall_avg_score,
                    "avg_sequence_length": overall_avg_seq_length
                })
        
        return sorted(models, key=lambda x: x["name"])
    
    @staticmethod
    def get_datasets_for_model(model_name: str) -> List[Dict[str, Any]]:
        """Get all datasets for a specific model"""
        model_dir = DATA_DIR / f"eval_results_{model_name}"
        datasets = []
        
        if not model_dir.exists():
            return datasets
            
        for dataset_dir in model_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_info = DataLoader.get_dataset_info(model_name, dataset_dir.name)
                if dataset_info:
                    datasets.append(dataset_info)
        
        return sorted(datasets, key=lambda x: x["name"])
    
    @staticmethod
    def get_dataset_info(model_name: str, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific dataset"""
        dataset_dir = DATA_DIR / f"eval_results_{model_name}" / dataset_name
        
        if not dataset_dir.exists():
            return None
            
        # Load aggregated metrics
        aggregated_file = dataset_dir / "aggregated_metrics.json"
        aggregated_metrics = None
        if aggregated_file.exists():
            with open(aggregated_file, 'r') as f:
                aggregated_metrics = json.load(f)
        
        # Get individual runs
        runs = []
        for file in dataset_dir.glob("run_*_summary.csv"):
            run_id = int(file.stem.split("_")[1])
            
            # Load summary CSV
            summary_df = pd.read_csv(file)
            overall_row = summary_df[summary_df["dataset"] == "OVERALL"].iloc[0]
            
            runs.append({
                "run_id": run_id,
                "summary": overall_row.to_dict(),
                "has_detailed": (dataset_dir / f"run_{run_id}_detailed.csv").exists()
            })
        
        runs = sorted(runs, key=lambda x: x["run_id"])
        
        return {
            "name": dataset_name,
            "model_name": model_name,
            "aggregated_metrics": aggregated_metrics,
            "runs": runs,
            "num_runs": len(runs)
        }
    
    @staticmethod
    def get_samples_for_dataset(model_name: str, dataset_name: str) -> List[Dict[str, Any]]:
        """Get all samples for a dataset with aggregated results across runs"""
        dataset_dir = DATA_DIR / f"eval_results_{model_name}" / dataset_name
        
        if not dataset_dir.exists():
            return []
        
        # Load original dataset info if available
        original_file = dataset_dir / "original_dataset.jsonl"
        original_data = {}
        if original_file.exists():
            with open(original_file, 'r') as f:
                for i, line in enumerate(f):
                    data = json.loads(line.strip())
                    original_data[i] = data
        
        # Aggregate results across all runs
        sample_results = {}
        
        # Find all detailed CSV files
        for detailed_file in dataset_dir.glob("run_*_detailed.csv"):
            try:
                run_id = int(detailed_file.stem.split("_")[1])
                detailed_df = pd.read_csv(detailed_file)
                
                for idx, row in detailed_df.iterrows():
                    sample_id = idx  # Use row index as sample ID
                    
                    if sample_id not in sample_results:
                        sample_results[sample_id] = {
                            "sample_id": sample_id,
                            "original_data": original_data.get(sample_id, {}),
                            "runs": {},
                            "best_score": 0,
                            "worst_score": 1,
                            "avg_score": 0,
                            "success_rate": 0,
                            "total_runs": 0
                        }
                    
                    # Add this run's result
                    sample_results[sample_id]["runs"][run_id] = {
                        "run_id": run_id,
                        "prompt": row.get("prompt", ""),
                        "response": row.get("response", ""),
                        "ground_truth": row.get("ground_truth", ""),
                        "score": row.get("score", 0),
                        "manufactoria_pass_rate": row.get("manufactoria_pass_rate", 0),
                        "finish_reason": row.get("finish_reason", ""),
                        "failure_reason": row.get("failure_reason", "none"),
                        "dataset_source": row.get("dataset_source", ""),
                        "sequence_length": row.get("sequence_length", 0),
                        "num_tool_calls": row.get("num_tool_calls", 0),
                        "tool_timeout": row.get("tool_timeout", False),
                        "tool_error": row.get("tool_error", False),
                        "tool_output": row.get("tool_output", ""),
                        "tool_runtime": row.get("tool_runtime", 0.0),
                        "tool_called": row.get("tool_called", False),
                    }
                    
            except (ValueError, IndexError, Exception) as e:
                print(f"Error processing {detailed_file}: {e}")
                continue
        
        # Calculate aggregated statistics for each sample
        samples = []
        for sample_id, sample_data in sample_results.items():
            runs = list(sample_data["runs"].values())
            if not runs:
                continue
                
            scores = [run["score"] for run in runs]
            pass_rates = [run["manufactoria_pass_rate"] for run in runs]
            seq_lengths = [run["sequence_length"] for run in runs]
            tool_runtimes = [run["tool_runtime"] for run in runs]
            sample_data.update({
                "total_runs": len(runs),
                "best_score": max(scores),
                "worst_score": min(scores),
                "avg_score": np.mean(scores),
                "avg_pass_rate": np.mean(pass_rates),
                "score_std": np.std(scores) if len(scores) > 1 else 0,
                "pass_rate_std": np.std(pass_rates) if len(pass_rates) > 1 else 0,
                "success_rate": sum(1 for score in scores if score > 0) / len(scores),
                "avg_sequence_length": np.mean(seq_lengths),
                "min_sequence_length": min(seq_lengths),
                "max_sequence_length": max(seq_lengths),
                "seq_length_std": np.std(seq_lengths) if len(seq_lengths) > 1 else 0,
                "total_tool_calls": sum(run["num_tool_calls"] for run in runs),
                "avg_tool_runtime": np.mean(tool_runtimes),
                "timeout_rate": sum(1 for run in runs if run["tool_timeout"]) / len(runs),
                "error_rate": sum(1 for run in runs if run["tool_error"]) / len(runs),
            })
            
            samples.append(sample_data)
        
        return sorted(samples, key=lambda x: x["sample_id"])
    
    @staticmethod
    def get_sample_details(model_name: str, dataset_name: str, sample_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific sample across all runs"""
        samples = DataLoader.get_samples_for_dataset(model_name, dataset_name)
        
        for sample in samples:
            if sample["sample_id"] == sample_id:
                return {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "sample_id": sample_id,
                    "sample_data": sample
                }
        
        return None


@app.route('/')
def home():
    """Home/Overview Page - High-level summary across all models and datasets"""
    models = DataLoader.get_all_models()
    
    # Calculate overall statistics
    total_models = len(models)
    total_datasets = sum(m["num_datasets"] for m in models)
    total_runs = sum(m["total_runs"] for m in models)
    
    return render_template('home.html', 
                         models=models,
                         total_models=total_models,
                         total_datasets=total_datasets,
                         total_runs=total_runs)


@app.route('/model/<model_name>')
def model_detail(model_name: str):
    """Model Detail Page - Comprehensive view of a specific model's performance"""
    datasets = DataLoader.get_datasets_for_model(model_name)
    
    if not datasets:
        abort(404)
    
    # Calculate model-level statistics
    total_runs = sum(d["num_runs"] for d in datasets)
    avg_scores = []
    avg_seq_lengths = []
    
    for dataset in datasets:
        if dataset["aggregated_metrics"]:
            avg_score = dataset["aggregated_metrics"]["metrics"].get("avg_score", {}).get("mean", 0)
            avg_seq_length = dataset["aggregated_metrics"]["metrics"].get("avg_sequence_length", {}).get("mean", 0)
            avg_scores.append(avg_score)
            avg_seq_lengths.append(avg_seq_length)
    
    model_avg_score = np.mean(avg_scores) if avg_scores else 0
    model_score_std = np.std(avg_scores) if len(avg_scores) > 1 else 0
    model_avg_seq_length = np.mean(avg_seq_lengths) if avg_seq_lengths else 0
    model_seq_length_std = np.std(avg_seq_lengths) if len(avg_seq_lengths) > 1 else 0
    
    return render_template('model_detail.html',
                         model_name=model_name,
                         datasets=datasets,
                         total_runs=total_runs,
                         model_avg_score=model_avg_score,
                         model_score_std=model_score_std,
                         model_avg_seq_length=model_avg_seq_length,
                         model_seq_length_std=model_seq_length_std)


@app.route('/model/<model_name>/dataset/<dataset_name>')
def dataset_detail(model_name: str, dataset_name: str):
    """Dataset Detail Page - Sample browser for a specific dataset"""
    dataset_info = DataLoader.get_dataset_info(model_name, dataset_name)
    samples = DataLoader.get_samples_for_dataset(model_name, dataset_name)
    
    if not dataset_info:
        abort(404)
    
    return render_template('dataset_detail.html',
                         model_name=model_name,
                         dataset_name=dataset_name,
                         dataset_info=dataset_info,
                         samples=samples)


@app.route('/model/<model_name>/dataset/<dataset_name>/sample/<int:sample_id>')
def sample_detail(model_name: str, dataset_name: str, sample_id: int):
    """Sample Detail Page - Individual sample analysis across all runs"""
    sample_details = DataLoader.get_sample_details(model_name, dataset_name, sample_id)
    
    if not sample_details:
        abort(404)
    
    sample_data = sample_details["sample_data"]
    runs = list(sample_data["runs"].values())
    runs = sorted(runs, key=lambda x: x["run_id"])
    
    # Calculate additional statistics
    scores = [run["score"] for run in runs]
    failure_reasons = [run["failure_reason"] for run in runs if run["failure_reason"] != "none"]
    
    failure_summary = {}
    for reason in failure_reasons:
        failure_summary[reason] = failure_summary.get(reason, 0) + 1
    
    # Add success count
    success_count = sum(1 for score in scores if score > 0)
    if success_count > 0:
        failure_summary["Success"] = success_count
    
    score_stats = {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": min(scores),
        "max": max(scores)
    }
    
    return render_template('sample_detail.html',
                         model_name=model_name,
                         dataset_name=dataset_name,
                         sample_id=sample_id,
                         sample_data=sample_data,
                         runs=runs,
                         failure_summary=failure_summary,
                         score_stats=score_stats,
                         total_runs=len(runs))


@app.template_filter('round_float')
def round_float(value, decimals=4):
    """Template filter to round floating point numbers"""
    if isinstance(value, (int, float)) and not np.isnan(value):
        return round(float(value), decimals)
    return value


@app.template_filter('percentage')
def percentage(value, decimals=1):
    """Template filter to convert decimal to percentage"""
    if isinstance(value, (int, float)) and not np.isnan(value):
        return f"{float(value) * 100:.{decimals}f}%"
    return value


@app.template_filter('format_large_number')
def format_large_number(value):
    """Template filter to format large numbers with commas"""
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    return value


@app.template_filter('preserve_newlines')
def preserve_newlines(value):
    """Template filter to preserve newlines for display"""
    if isinstance(value, str):
        # Replace \n with actual newlines and handle common escape sequences
        value = value.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        return value
    return value


@app.template_filter('safe_markdown_id')
def safe_markdown_id(value):
    """Template filter to create safe IDs for markdown containers"""
    if isinstance(value, (str, int)):
        return str(value).replace(' ', '_').replace('.', '_').replace('-', '_')
    return value


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)
