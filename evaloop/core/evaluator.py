"""Main evaluator class for EvaLoop framework."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
from pathlib import Path

from .config import EvaluationConfig
from ..models.factory import ModelFactory
from ..tasks.factory import TaskFactory
from ..evaluation.cycle import BatchEvaluationCycle
from ..evaluation.py_testing import EvalPlusCodeTester, load_mbpp_dataset


class EvaLoopEvaluator:
    """Main evaluator class for running EvaLoop experiments."""
    
    # Constants
    TASK_OUTPUT_KEY_TEMPLATE = "task_{}_output"
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Evaluation configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.task_factory = TaskFactory()
        
        self.logger.info(f"Initialized EvaLoopEvaluator with config: {config.experiment_name}")
    
    def run(self) -> List[Dict[str, Any]]:
        """
        Run the loop selected by ``config.loop`` (e.g. code generation <-> summarization).
        
        Returns:
            List of evaluation results (single model result in list for consistency).
        """
        self.logger.info(f"Starting {self.config.loop} evaluation")
        
        # Setup components
        model = self._create_model()
        tasks = self._create_tasks()
        prompts, py_problems = self._load_dataset()
        tester = self._create_tester(py_problems)
        
        # Run evaluation
        model_result = self._run_evaluation_cycles(model, tasks, prompts, py_problems, tester)
        
        # Save and return results
        self._save_results([model_result], self.config.loop)
        return [model_result]

    def run_code_generation_summarization(self) -> List[Dict[str, Any]]:
        """Backward-compatible alias of run(); runs the loop selected by ``config.loop``."""
        return self.run()
    
    def run_code_translation(self) -> Dict[str, Any]:
        """
        Run code translation evaluation.
        
        Note: This method is currently not implemented. 
        It's kept as an interface for future extension.
        
        Returns:
            Dictionary containing evaluation results.
        """
        raise NotImplementedError("Code translation evaluation is not currently supported")
    
    def _create_model(self):
        """Create and configure the model."""
        model_config = self.config.get_model_config()
        model = self.model_factory.create_model(model_config)
        self.logger.info(f"Created model for evaluation: {model}")
        return model
    
    def _create_tasks(self):
        """Create task instances."""
        task_configs = self.config.get_task_configs(self.config.loop)
        tasks = [self.task_factory.create_task(config) for config in task_configs]
        self.logger.info(f"Created {len(tasks)} tasks for evaluation")
        return tasks
    
    def _load_dataset(self):
        """Load and prepare the dataset."""
        py_problems, ground_truth = load_mbpp_dataset()
        self.logger.info(f"Loaded MBPP dataset with {len(py_problems)} problems")
        
        # Prepare prompts from dataset
        prompts = []
        for task_id, problem in py_problems.items():
            prompts.append({
                "task_id": task_id,
                "prompt": problem["prompt"]
            })
        if self.config.limit:
            # First N tasks in dataset order; meant for smoke tests, not for reported runs
            prompts = prompts[: self.config.limit]
        
        self.logger.info(f"Prepared {len(prompts)} prompts from MBPP dataset")
        return prompts, py_problems
    
    def _create_tester(self, py_problems):
        """Create the code tester."""
        _, ground_truth = load_mbpp_dataset()  # Get ground truth
        return EvalPlusCodeTester(py_problems, ground_truth)
    
    def _run_evaluation_cycles(self, model, tasks, prompts, py_problems, tester) -> Dict[str, Any]:
        """Run the main evaluation cycles."""
        self.logger.info(f"Evaluating model: {model}")
        
        # Create evaluation cycle runner
        cycle_runner = BatchEvaluationCycle(model, tasks)
        
        # Initialize results structure
        model_result = self._initialize_model_result(model, prompts)
        
        # Run cycles until max_cycles or all prompts fail
        current_prompts = prompts.copy()
        current_cycle = 0
        
        while current_cycle < self.config.max_cycles and current_prompts:
            current_cycle += 1
            self.logger.info(f"Starting cycle {current_cycle}/{self.config.max_cycles}")
            
            # Run single cycle, re-running prompts whose model calls failed
            cycle_output = cycle_runner.run_single_cycle(current_prompts, current_cycle)
            self._retry_errored_prompts(cycle_runner, model, cycle_output, current_prompts, current_cycle)
            
            # Process results and prepare for next cycle
            current_prompts = self._process_cycle_results(
                cycle_output, model_result, tasks, py_problems, tester, current_cycle
            )
            # Checkpoint, so an aborted run keeps every completed cycle
            self._save_results([model_result], self.config.loop)
            
            if not current_prompts:
                self.logger.info(f"All prompts failed or completed after cycle {current_cycle}")
                break
        
        # Finalize results
        self._finalize_model_result(model_result)
        
        return model_result
    
    @staticmethod
    def _has_task_error(cycle_result: Dict[str, Any]) -> bool:
        """True if a model call of this cycle failed (an infrastructure error, not a model output)."""
        return any(key.startswith("task_") and key.endswith("_error") for key in cycle_result)

    def _retry_errored_prompts(self, cycle_runner, model, cycle_output, prompts, cycle_number):
        """Re-run, in place, the cycle of every prompt whose model call raised.

        Failures are retried one prompt at a time, so a single failing prompt cannot fail a
        whole batch again. Prompts still failing after ``cycle_retries`` keep their error.
        """
        results = cycle_output["cycle_results"]
        prompt_by_id = {prompt["task_id"]: prompt for prompt in prompts}
        workers = getattr(model, "max_workers", 1)

        def rerun(prompt):
            return cycle_runner.run_single_cycle([prompt], cycle_number)["cycle_results"][0]

        for attempt in range(1, self.config.cycle_retries + 1):
            errored = [i for i, result in enumerate(results) if self._has_task_error(result)]
            if not errored:
                return
            self.logger.warning(
                f"Cycle {cycle_number}: re-running {len(errored)} prompt(s) whose model calls failed "
                f"(retry {attempt}/{self.config.cycle_retries})"
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                reruns = list(pool.map(rerun, [prompt_by_id[results[i]["task_id"]] for i in errored]))
            for index, result in zip(errored, reruns):
                results[index] = result
    
    def _initialize_model_result(self, model, prompts) -> Dict[str, Any]:
        """Initialize the model result structure."""
        model_result = {
            "model": str(model),
            # Loop budget M; needed by the analyzer to normalize the ASL metrics
            "max_cycles": self.config.max_cycles,
            # Provenance: which loop and settings produced this run
            "loop": self.config.loop,
            "config": self.config.to_dict(),
            # False in per-cycle checkpoints; True once every cycle has run
            "complete": False,
            "prompt_results": []
        }
        
        for prompt in prompts:
            prompt_result = {
                "task_id": prompt["task_id"],
                "initial_prompt": prompt["prompt"],
                "cycles": [],
                "successful_cycles": 0,
                "max_cycles_reached": False
            }
            model_result["prompt_results"].append(prompt_result)
        
        return model_result
    
    def _process_cycle_results(self, cycle_output, model_result, tasks, py_problems, tester, current_cycle) -> List[Dict[str, Any]]:
        """Process cycle results and return prompts for next cycle."""
        next_cycle_prompts = []
        api_errors = 0
        
        for cycle_result in cycle_output["cycle_results"]:
            task_id = cycle_result["task_id"]
            problem = py_problems.get(task_id)
            
            # Find corresponding prompt result
            prompt_result = next(r for r in model_result["prompt_results"] if r["task_id"] == task_id)

            # A model call that still failed after retries says nothing about the model:
            # stop this task's loop without counting the cycle as a failed generation.
            if self._has_task_error(cycle_result):
                api_errors += 1
                cycle_result["success"] = None
                prompt_result["cycles"].append(cycle_result)
                prompt_result["api_error_cycle"] = current_cycle
                self.logger.error(f"{task_id}: model call failed in cycle {current_cycle}; task stopped")
                continue
            
            # Test and validate the code generation output
            success = self._test_and_validate_output(cycle_result, tasks, task_id, problem, tester)
            cycle_result["success"] = success
            
            # Add cycle result to prompt result
            prompt_result["cycles"].append(cycle_result)
            
            # Update successful cycles and prepare for next cycle if successful
            if success:
                prompt_result["successful_cycles"] = current_cycle
                next_prompt = self._prepare_next_cycle_prompt(cycle_result, tasks, task_id)
                if next_prompt:
                    next_cycle_prompts.append(next_prompt)

        results = cycle_output["cycle_results"]
        if results and api_errors / len(results) > self.config.max_api_error_rate:
            # Systemic failure (outage, exhausted credits, ...): keep what we have and stop
            self._save_results([model_result], self.config.loop)
            raise RuntimeError(
                f"{api_errors}/{len(results)} prompts of cycle {current_cycle} failed with API errors "
                f"(> {self.config.max_api_error_rate:.0%}); aborting. Completed cycles are saved."
            )
        
        return next_cycle_prompts
    
    def _test_and_validate_output(self, cycle_result, tasks, task_id, problem, tester) -> bool:
        """Test and validate the code generation output."""
        # Find code generation task
        code_gen_index = self._find_code_generation_task_index(tasks)
        
        if code_gen_index is None:
            return False
        
        output_key = self.TASK_OUTPUT_KEY_TEMPLATE.format(code_gen_index + 1)
        
        if output_key not in cycle_result:
            return False
        
        # Test the generated code
        code_output = cycle_result[output_key]
        test_results = self._test_python_code(code_output, task_id, problem, tester)
        
        # Store test results
        # Tester errors return no details; .get keeps them from crashing the whole run
        test_results["plus_details"] = "".join(map(str, test_results.get("plus_details", [])))
        cycle_result["test_results"] = test_results
        
        return test_results.get("success", False)
    
    def _find_code_generation_task_index(self, tasks) -> int:
        """Find the index of the task whose output should be functionally tested."""
        return next((i for i, task in enumerate(tasks)
                     if getattr(task, "produces_testable_code", False)), None)
    
    def _prepare_next_cycle_prompt(self, cycle_result, tasks, task_id) -> Dict[str, Any]:
        """Prepare prompt for the next cycle."""
        last_task_index = len(tasks)
        last_output_key = self.TASK_OUTPUT_KEY_TEMPLATE.format(last_task_index)
        
        if last_output_key in cycle_result:
            return {
                "task_id": task_id,
                "prompt": cycle_result[last_output_key]
            }
        
        return None
    
    def _finalize_model_result(self, model_result):
        """Finalize model results with summary statistics."""
        # Mark prompts that reached max cycles
        for prompt_result in model_result["prompt_results"]:
            prompt_result["max_cycles_reached"] = (
                prompt_result["successful_cycles"] == self.config.max_cycles
            )
        
        # Tasks stopped by a model call that failed after retries (see _process_cycle_results)
        model_result["api_error_tasks"] = [
            r["task_id"] for r in model_result["prompt_results"] if "api_error_cycle" in r
        ]
        model_result["complete"] = True
        
        # Calculate average successful cycles
        if model_result["prompt_results"]:
            avg_cycles = sum(
                r["successful_cycles"] for r in model_result["prompt_results"]
            ) / len(model_result["prompt_results"])
            model_result["average_successful_cycles"] = avg_cycles
    
    def _test_python_code(self, output: str, task_id: str, problem: Dict, tester: EvalPlusCodeTester) -> Dict[str, Any]:
        """Test code output using the EvalPlus tester."""
        try:
            test_results = tester.test_code(
                output, 
                task_id, 
                problem["entry_point"]
            )
            return test_results
        except Exception as e:
            self.logger.error(f"Error testing code for task {task_id}: {e}")
            return {"success": False, "error": str(e)}
    
    
    def _save_results(self, results: List[Dict[str, Any]], experiment_type: str):
        """Save evaluation results to file."""
        output_file = Path(self.config.output_dir) / f"{self.config.experiment_name}_{experiment_type}_results.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")
    
