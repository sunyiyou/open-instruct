"""
Manufactoria Inline Feedback for vLLM Generation

This module extends the ToolUseLLM pattern to provide inline feedback for Manufactoria DSL code.
Instead of detecting tool calls, it detects EOS tokens during Manufactoria generation, evaluates
the DSL against test cases, and provides feedback by replacing the EOS token with test results.
"""

import asyncio
import copy
import json
import logging
import re
import time
import warnings
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import requests
from tqdm import tqdm
from vllm import (
    LLM,
    PoolingParams,
    PoolingRequestOutput,
    PromptType,
    RequestOutput,
    SamplingParams,
    TokensPrompt,
)
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest
from vllm.prompt_adapter.request import PromptAdapterRequest

logger = logging.getLogger(__name__)


@dataclass
class FeedbackOutput:
    """Output from Manufactoria feedback generation."""
    feedback_text: str
    test_results: List[Dict]
    all_passed: bool
    error: Optional[str] = None
    timeout: bool = False
    runtime: float = 0.0


@dataclass
class ManufactoriaFeedbackConfig:
    """Configuration for Manufactoria feedback generation."""
    api_url: str = "http://localhost:8071"
    enable_feedback: bool = True
    max_iterations: int = 2
    feedback_datasets: List[str] = None
    timeout_seconds: float = 5.0
    feedback_format: str = "detailed"  # "detailed" or "minimal"
    feedback_on_pass: bool = True  # Whether to provide feedback when all tests pass
    
    def __post_init__(self):
        if self.feedback_datasets is None:
            self.feedback_datasets = ["manufactoria"]


class ManufactoriaFeedbackLLM(LLM):
    """
    LLM wrapper that provides inline feedback for Manufactoria DSL generations.
    
    When generating for Manufactoria datasets, this class:
    1. Detects EOS tokens during generation
    2. Extracts DSL code from the generated text
    3. Evaluates DSL against provided test cases via API
    4. Replaces EOS with feedback if tests fail
    5. Allows model to continue generating from feedback
    
    Similar to ToolUseLLM but focused on EOS replacement rather than tool detection.
    """
    
    def __init__(self, 
                 config: Optional[ManufactoriaFeedbackConfig] = None,
                 **kwargs):
        """
        Initialize ManufactoriaFeedbackLLM.
        
        Args:
            config: Configuration for feedback behavior
            **kwargs: Arguments passed to base LLM class
        """
        super().__init__(**kwargs)
        
        self.config = config or ManufactoriaFeedbackConfig()
        self.feedback_datasets = set(self.config.feedback_datasets)
        
        # Thread pool for async feedback generation
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Track pending feedback requests
        self.pending_feedback_futures = {}
        
        # Store test cases and dataset info per request
        self.request_test_cases = {}
        self.request_datasets = {}
        
        # Override sampling params to ensure n=1 for feedback consistency
        self.single_n_sampling_params = None
    
    def set_request_metadata(self, request_id: str, test_cases: List[Dict], dataset: str):
        """
        Associate test cases and dataset with a generation request.
        
        Args:
            request_id: Unique identifier for the request
            test_cases: List of test case dictionaries for Manufactoria API
            dataset: Dataset name (e.g., "manufactoria")
        """
        self.request_test_cases[request_id] = test_cases
        self.request_datasets[request_id] = dataset
    
    def set_batch_metadata(self, prompt_idx: int, test_cases: List[Dict], dataset: str, n_samples: int):
        """
        Set metadata for all completions of a prompt when n > 1.
        
        This creates metadata for all request IDs: "0-0", "0-1", "0-2", etc.
        
        Args:
            prompt_idx: Index of the prompt in the batch (0, 1, 2, ...)
            test_cases: List of test case dictionaries for Manufactoria API
            dataset: Dataset name (e.g., "manufactoria")  
            n_samples: Number of samples per prompt (typically from SamplingParams.n)
        """
        for j in range(n_samples):
            request_id = f"{prompt_idx}-{j}"
            self.set_request_metadata(request_id, test_cases, dataset)
    
    def is_feedback_enabled_for_request(self, request_id: str) -> bool:
        """Check if feedback should be generated for this request."""
        if not self.config.enable_feedback:
            return False
        
        dataset = self.request_datasets.get(request_id, "")
        return dataset.lower() in self.feedback_datasets
    
    def _should_provide_feedback(self, output, eos_token_id: Optional[int]) -> bool:
        """
        Check if we should provide feedback for this output.
        
        We provide feedback when:
        1. Generation finished naturally (EOS token or stop sequence)
        2. Output contains potential DSL code
        """
        # Check if output ends with EOS token
        if eos_token_id is not None and output.token_ids and output.token_ids[-1] == eos_token_id:
            return True
        
        # Check if output ends with stop sequence (common in chat models)
        if hasattr(output, 'finish_reason') and output.finish_reason in ['stop', 'eos']:
            return True
        
        # For vLLM, check if the output was stopped by a stop sequence
        # by looking at the stop_reason or finish_reason
        if hasattr(output, 'stop_reason') and output.stop_reason is not None:
            return True
            
        return False
    
    def extract_dsl_code(self, text: str) -> Optional[str]:
        """
        Extract Manufactoria DSL code from generated text.
        
        Uses the same pattern as ManufactoriaVerifier.extract_manufactoria_code.
        """
        # Find content between ``` markers
        pattern = r"```(?:manufactoria)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if not matches:
            # If no code blocks, return the entire text as potential DSL
            return text.strip()
        
        # Return the last match, stripped of whitespace
        return matches[-1].strip()
    
    def has_feedback_after_last_dsl(self, text: str) -> bool:
        """
        Check if there's already feedback after the last DSL code block.
        
        This prevents generating duplicate feedback when the model doesn't
        produce new DSL code after receiving feedback.
        
        Args:
            text: The full generated text to analyze
            
        Returns:
            True if feedback already exists after the last DSL code, False otherwise
        """
        # Find all code blocks and feedback blocks (both detailed and minimal formats)
        code_pattern = r"```(?:manufactoria)?(.*?)```"
        detailed_feedback_pattern = r"--- Test Results ---.*?--- End Results ---"
        
        code_matches = list(re.finditer(code_pattern, text, re.DOTALL))
        detailed_feedback_matches = list(re.finditer(detailed_feedback_pattern, text, re.DOTALL))
        
        # Combine all feedback matches
        all_feedback_matches = detailed_feedback_matches
        
        if not code_matches:
            # No DSL code found, check if there's any feedback
            has_feedback = len(all_feedback_matches) > 0
            if has_feedback:
                logger.debug("No DSL code found but feedback exists - skipping duplicate feedback")
            return has_feedback
        
        # Get the position of the last DSL code block
        last_code_end = code_matches[-1].end()
        
        # Check if any feedback blocks come after the last DSL code
        for feedback_match in all_feedback_matches:
            if feedback_match.start() > last_code_end:
                logger.debug(f"Found existing feedback after last DSL code at position {feedback_match.start()} - skipping duplicate feedback")
                return True
        
        return False
    
    async def generate_feedback(self, dsl_code: str, test_cases: List[Dict]) -> FeedbackOutput:
        """
        Generate feedback for DSL code by testing against test cases.
        
        Args:
            dsl_code: The DSL code to test
            test_cases: List of test case dictionaries
            
        Returns:
            FeedbackOutput with feedback text and test results
        """
        start_time = time.time()
        
        payload = {
            "dsl": dsl_code,
            "test_cases": test_cases,
            "max_execution_time": 1.0
        }
        
        try:
            # Make async request to Manufactoria API
            def make_request():
                response = requests.post(
                    f"{self.config.api_url}/test_solution",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.config.timeout_seconds
                )
                response.raise_for_status()
                return response.json()
            
            result = await asyncio.to_thread(make_request)
            runtime = time.time() - start_time
            
            # Check if the DSL code is valid first
            if not result.get("valid", True):
                # Handle syntax/parse errors
                error_message = result.get("message", "Unknownerror")
                feedback_text = self.format_syntax_error(error_message)
                
                return FeedbackOutput(
                    feedback_text=feedback_text,
                    test_results=[],
                    all_passed=False,
                    runtime=runtime
                )
            
            all_passed = result.get("all_passed", False)
            test_results = result.get("results", [])
            
            if all_passed:
                # Generate positive feedback when all tests pass (if enabled)
                if self.config.feedback_on_pass:
                    feedback_text = self.format_passed_tests(test_results)
                else:
                    feedback_text = ""  # No feedback when all tests pass (if disabled)
                
                return FeedbackOutput(
                    feedback_text=feedback_text,
                    test_results=test_results,
                    all_passed=True,
                    runtime=runtime
                )
            
            # Generate feedback for failed tests
            feedback_text = self.format_failed_tests(test_results)
            
            return FeedbackOutput(
                feedback_text=feedback_text,
                test_results=test_results,
                all_passed=False,
                runtime=runtime
            )
            
        except asyncio.TimeoutError:
            runtime = time.time() - start_time
            logger.warning(f"Manufactoria API timeout after {runtime:.2f}s")
            return FeedbackOutput(
                feedback_text="",
                test_results=[],
                all_passed=False,
                error="API timeout",
                timeout=True,
                runtime=runtime
            )
        except Exception as e:
            runtime = time.time() - start_time
            logger.warning(f"Manufactoria API error: {e}")
            return FeedbackOutput(
                feedback_text="",
                test_results=[],
                all_passed=False,
                error=str(e),
                runtime=runtime
            )
    
    def format_passed_tests(self, test_results: List[Dict]) -> str:
        """
        Format positive feedback when all tests pass.
        
        Args:
            test_results: List of test result dictionaries from API
            
        Returns:
            Formatted positive feedback string
        """
        if not test_results:
            return "\n\n✅ All tests passed!"
        
        if self.config.feedback_format == "minimal":
            return f"\n\n✅ All {len(test_results)} tests passed!"
        
        # Detailed positive feedback format
        feedback_lines = ["\n\n--- Test Results ---"]
        feedback_lines.append(f"✅ All {len(test_results)} tests PASSED!")
        feedback_lines.append("--- End Results ---\n")
        return "\n".join(feedback_lines)
    
    def format_failed_tests(self, test_results: List[Dict]) -> str:
        """
        Format failed test results into readable feedback text.
        
        Args:
            test_results: List of test result dictionaries from API
            
        Returns:
            Formatted feedback string to append to generation
        """
        failed_tests = [r for r in test_results if not r.get('passed', False)]
        if not failed_tests:
            return ""
        
        if self.config.feedback_format == "minimal":
            return f"\n\n--- Test Results ---\n❌ FAILED: {len(failed_tests)}/{len(test_results)} tests\n--- End Results ---\nGiven the test results"
        
        # Detailed feedback format
        feedback_lines = ["\n\n--- Test Results ---"]
        
        for i, test in enumerate(failed_tests[:3]):  # Limit to first 3 failed tests
            feedback_lines.append(f"Test {i+1} FAILED:")
            feedback_lines.append(f"  Input: '{test.get('input', '')}'")
            
            # Check if this test is checking output or just acceptance
            if test.get('check_output', True):  # Default to True for backward compatibility
                feedback_lines.extend([
                    f"  Expected Ouput: '{test.get('expected_output', '')}'",
                    f"  Actual Output: '{test.get('actual_output', '')}'",
                ])
            else:
                feedback_lines.extend([
                    f"  Expected Acceptance: {test.get('expected_accepted', 'Unknown')}",
                    f"  Actual Acceptance: {test.get('actual_accepted', 'Unknown')}",
                ])
            
            # if test.get('rejection_reason'):
            #     feedback_lines.append(f"  Reason: {test['rejection_reason']}")
        
        if len(failed_tests) > 3:
            feedback_lines.append(f"... and {len(failed_tests) - 3} more failures")
        
        feedback_lines.append("--- End Results ---\n\nGiven the test results")
        return "\n".join(feedback_lines)
    
    def format_syntax_error(self, error_message: str) -> str:
        """
        Format syntax/parse error feedback.
        
        Args:
            error_message: The error message from the API
            
        Returns:
            Formatted feedback string for syntax errors
        """
        # if self.config.feedback_format == "minimal":
        #     return f"\n\n--- Test Results ---\n❌ {error_message}\n--- End Results ---\n\nGiven the test results"
        
        # Detailed feedback format
        feedback_lines = [
            "\n\n--- Test Results ---",
            f"❌ {error_message}",
            "--- End Results ---\n\nGiven the test results"
        ]
        return "\n".join(feedback_lines)
    
    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams, Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[list[int]] = None,
    ) -> None:
        """Use ToolUseLLM approach for multiple samples to maintain feedback state isolation."""
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params must be the same.")
        if isinstance(lora_request, list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request must be the same.")

        # Apply guided params similar to ToolUseLLM
        for sp in params if isinstance(params, list) else [params]:
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

        # Use ToolUseLLM approach: create individual requests for each completion
        # This ensures each completion has isolated feedback state
        assert not isinstance(params, list), "ManufactoriaFeedbackLLM doesn't support per-prompt params"
        self.single_n_sampling_params = copy.deepcopy(params)
        self.single_n_sampling_params.n = 1
        
        # Add requests to the engine (like ToolUseLLM)
        for i, prompt in enumerate(prompts):
            for j in range(params.n):
                request_id = f"{i}-{j}"
                self.llm_engine.add_request(
                    request_id,
                    prompt,
                    self.single_n_sampling_params,
                    lora_request=lora_request[i] if isinstance(lora_request, Sequence) else lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority[i] if priority else 0,
                )
    
    def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """
        Run the engine with Manufactoria feedback support.
        
        This method extends the base generation loop to:
        1. Detect EOS tokens in feedback-enabled requests
        2. Extract and evaluate DSL code
        3. Replace EOS with feedback if tests fail
        4. Continue generation from feedback
        """
        # Initialize progress bar
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix="feedback: 0 calls",
            )

        # Tracking variables
        outputs: List[Union[RequestOutput, PoolingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        tokenizer = self.get_tokenizer()
        
        # Feedback-specific tracking
        feedback_iterations = defaultdict(int)
        feedback_calls = 0
        concat_outputs = {}
        masks = defaultdict(list)
        
        # Get EOS token ID for detection
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_token_id is None:
            logger.warning("No EOS token ID found, feedback may not work properly")
        
        while True:
            # Process pending feedback futures
            dict_keys_to_delete = []
            for req_id, (future, last_o, last_output) in self.pending_feedback_futures.items():
                if future.done():
                    feedback_result = future.result()
                    
                    # Get current state
                    last_prompt_token_ids = last_output.prompt_token_ids
                    last_token_ids = last_o.token_ids
                    
                    if feedback_result.feedback_text:
                        # STEP 1: Remove EOS token from the generated sequence
                        # We need to remove the EOS token that caused generation to stop
                        current_token_ids = concat_outputs[req_id].outputs[0].token_ids
                        current_masks = masks[req_id]
                        
                        # Check if we need to remove stop tokens/sequences
                        tokens_to_remove = 0
                        eos_token_removed = False
                        
                        # Check for EOS token
                        if (eos_token_id is not None and 
                            current_token_ids and 
                            current_token_ids[-1] == eos_token_id):
                            tokens_to_remove = 1
                            eos_token_removed = True
                        
                        # Remove stop tokens if found
                        if tokens_to_remove > 0 and len(current_token_ids) >= tokens_to_remove:
                            current_token_ids = current_token_ids[:-tokens_to_remove]
                            current_masks = current_masks[:-tokens_to_remove]
                            concat_outputs[req_id].outputs[0].token_ids = current_token_ids
                            masks[req_id] = current_masks
                        
                        # STEP 2: Tokenize feedback text
                        feedback_token_ids = tokenizer.encode(
                            feedback_result.feedback_text, add_special_tokens=False
                        )
                        
                        # Update token tracking
                        feedback_calls += 1
                        
                        # STEP 3: Check context length limits (use updated current_token_ids)
                        prompt_and_feedback_tokens = last_prompt_token_ids + current_token_ids + feedback_token_ids
                        excess = len(prompt_and_feedback_tokens) - self.llm_engine.model_config.max_model_len
                        if excess > 0:
                            feedback_token_ids = feedback_token_ids[:-excess]
                            can_continue = False
                        else:
                            can_continue = True
                        
                        # STEP 4: Check max_tokens limit
                        remaining = self.single_n_sampling_params.max_tokens - len(current_masks)
                        if remaining <= 0:
                            feedback_token_ids = []
                            can_continue = False
                        elif len(feedback_token_ids) > remaining:
                            feedback_token_ids = feedback_token_ids[:remaining]
                        
                        # STEP 5: Replace EOS with feedback tokens
                        concat_outputs[req_id].outputs[0].token_ids.extend(feedback_token_ids)
                        masks[req_id].extend([0] * len(feedback_token_ids))  # Mask feedback tokens
                        
                        # STEP 6: If all tests passed, add EOS token back to properly terminate sequence
                        # This ensures that when all tests pass and feedback_on_pass=True, the sequence
                        # is properly terminated instead of being left open-ended
                        if feedback_result.all_passed and eos_token_removed and eos_token_id is not None:
                            # Check if we have space for the EOS token
                            remaining_after_feedback = self.single_n_sampling_params.max_tokens - len(masks[req_id])
                            if remaining_after_feedback > 0:
                                concat_outputs[req_id].outputs[0].token_ids.append(eos_token_id)
                                masks[req_id].append(1)  # Mark EOS as model-generated (not feedback)
                                logger.debug(f"Added EOS token back after passed tests feedback for request {req_id}")
                            can_continue = False  # Don't continue generation when all tests pass
                        else:
                            # Continue generation if possible (for failed tests)
                            new_sample_tokens = self.single_n_sampling_params.max_tokens - len(masks[req_id])
                            can_continue = can_continue and new_sample_tokens > 0
                        
                        if can_continue:
                            try:
                                new_sampling_params = copy.deepcopy(self.single_n_sampling_params)
                                new_sampling_params.max_tokens = new_sample_tokens
                                self.llm_engine.add_request(
                                    req_id,
                                    TokensPrompt(prompt_token_ids=prompt_and_feedback_tokens),
                                    new_sampling_params,
                                )
                            except Exception as e:
                                logger.error(f"Error continuing generation after feedback: {e}")
                    
                    dict_keys_to_delete.append(req_id)
            
            # Clean up completed futures
            for req_id in dict_keys_to_delete:
                del self.pending_feedback_futures[req_id]
            
            # Process engine steps
            if self.llm_engine.has_unfinished_requests():
                step_outputs = self.llm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        assert len(output.outputs) <= 1  # n=1 enforced
                        o = output.outputs[0]
                        output_processed = False
                        
                        # Initialize or update concat_outputs
                        if output.request_id not in concat_outputs:
                            concat_outputs[output.request_id] = output
                        else:
                            concat_outputs[output.request_id].outputs[0].token_ids.extend(o.token_ids)
                        
                        # Track model-generated tokens
                        masks[output.request_id].extend([1] * len(o.token_ids))
                        
                        # Check for feedback opportunity when generation naturally finishes
                        # (i.e., EOS token generated or stop sequence hit)
                        if (self.is_feedback_enabled_for_request(output.request_id) and
                            feedback_iterations[output.request_id] < self.config.max_iterations and
                            self._should_provide_feedback(o, eos_token_id)):
                            
                            # Extract DSL code from generated text
                            full_text = tokenizer.decode(concat_outputs[output.request_id].outputs[0].token_ids)
                            
                            # Check if there's already feedback after the last DSL code
                            # This prevents duplicate feedback when no new DSL is generated
                            if self.has_feedback_after_last_dsl(full_text):
                                # Skip feedback - there's already feedback after the last DSL code
                                logger.debug(f"Skipping duplicate feedback for request {output.request_id} (iteration {feedback_iterations[output.request_id]})")
                            else:
                                dsl_code = self.extract_dsl_code(full_text)
                                test_cases = self.request_test_cases.get(output.request_id, [])
                                
                                if dsl_code and test_cases:
                                    logger.debug(f"Scheduling feedback generation for request {output.request_id} (iteration {feedback_iterations[output.request_id] + 1})")
                                    # Schedule feedback generation
                                    future = self.executor.submit(
                                        lambda: asyncio.run(self.generate_feedback(dsl_code, test_cases))
                                    )
                                    self.pending_feedback_futures[output.request_id] = (future, o, output)
                                    feedback_iterations[output.request_id] += 1
                                    output_processed = True
                                else:
                                    logger.debug(f"No DSL code or test cases found for request {output.request_id} - skipping feedback")
                        
                        if not output_processed:
                            outputs.append(output)
                            if use_tqdm:
                                if isinstance(output, RequestOutput):
                                    # Update progress bar
                                    n = len(output.outputs)
                                    assert output.prompt_token_ids is not None
                                    total_in_toks += len(output.prompt_token_ids) * n
                                    in_spd = total_in_toks / pbar.format_dict["elapsed"]
                                    total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                                    out_spd = total_out_toks / pbar.format_dict["elapsed"]
                                    pbar.postfix = f"feedback: {feedback_calls} calls, est. speed input: {in_spd:.2f} toks/s, output: {out_spd:.2f} toks/s"
                                    pbar.update(n)
                                else:
                                    pbar.update(1)
            
            # Check if done
            if not self.llm_engine.has_unfinished_requests() and len(self.pending_feedback_futures) == 0:
                break
        
        if use_tqdm:
            pbar.close()
        
        # Add masks to outputs
        for req_id in masks:
            if req_id in concat_outputs:
                setattr(concat_outputs[req_id].outputs[0], "mask", masks[req_id])
                setattr(concat_outputs[req_id].outputs[0], "feedback_iterations", feedback_iterations[req_id])
                
                # Validate mask length
                if len(masks[req_id]) != len(concat_outputs[req_id].outputs[0].token_ids):
                    logger.error(
                        f"Mask length {len(masks[req_id])} does not match "
                        f"token IDs length {len(concat_outputs[req_id].outputs[0].token_ids)}"
                    )
                concat_outputs[req_id].outputs[0].text = tokenizer.decode(concat_outputs[req_id].outputs[0].token_ids, skip_special_tokens=False)

        # Merge outputs and return
        merged_outputs = {}
        for req_id in concat_outputs:
            if len(concat_outputs[req_id].outputs[0].token_ids) > self.single_n_sampling_params.max_tokens:
                raise ValueError(
                    f"ManufactoriaFeedbackLLM generated more response tokens than max_tokens! "
                    f"len(concat_outputs[req_id].outputs[0].token_ids): {len(concat_outputs[req_id].outputs[0].token_ids)}"
                )
            real_req_id, _ = req_id.split("-")
            if real_req_id not in merged_outputs:
                merged_outputs[real_req_id] = concat_outputs[req_id]
            else:
                merged_outputs[real_req_id].outputs.append(concat_outputs[req_id].outputs[0])
        
        final_outputs = sorted(
            merged_outputs.values(), 
            key=lambda x: (int(x.request_id.split("-")[0]), int(x.request_id.split("-")[1]))
        )
        return final_outputs


if __name__ == "__main__":
    # Example usage
    from rich.console import Console
    from transformers import AutoTokenizer
    
    console = Console()
    
    # Test configuration
    config = ManufactoriaFeedbackConfig(
        api_url="http://localhost:1235",
        enable_feedback=True,
        max_iterations=2,
        feedback_format="detailed"
    )
    
    # Initialize LLM with feedback
    llm = ManufactoriaFeedbackLLM(
        config=config,
        model="microsoft/Phi-3-mini-4k-instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )
    
    # Example test cases
    test_cases = [
        {
            "input": "R",
            "expected_output": "",
            "expected_accepted": True,
            "check_output": False,
            "description": "Red robot should be accepted"
        },
        {
            "input": "B", 
            "expected_output": "",
            "expected_accepted": False,
            "check_output": False,
            "description": "Blue robot should be rejected"
        }
    ]
    
    # Example prompt
    system_prompt = """You are a Manufactoria DSL expert. Write DSL code to solve the given problem."""
    user_prompt = """Write Manufactoria DSL code that accepts red robots (R) but rejects blue robots (B).

```manufactoria
START start:
    IF R NEXT accept
    IF B NEXT reject

END accept
END reject
```"""
    
    prompt = f"{system_prompt}\n\n{user_prompt}"
    
    # Set up generation with multiple samples
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=512,
        stop=["</s>", "<|endoftext|>"],
        n=3  # Generate 3 completions per prompt
    )
    
    # Associate test cases with all completions for prompt 0
    llm.set_batch_metadata(prompt_idx=0, test_cases=test_cases, dataset="manufactoria", n_samples=sampling_params.n)
    
    console.print("🔧 Testing ManufactoriaFeedbackLLM...")
    console.print(f"Prompt: {prompt}")
    
    # Generate with feedback
    outputs = llm.generate([prompt], sampling_params)
    
    for i, output in enumerate(outputs):
        console.rule(f"Output {i}")
        console.print(f"Generated text: {output.outputs[0].text}")
        if hasattr(output.outputs[0], 'mask'):
            console.print(f"Mask length: {len(output.outputs[0].mask)}")
            console.print(f"Feedback iterations: {getattr(output.outputs[0], 'feedback_iterations', 0)}")
