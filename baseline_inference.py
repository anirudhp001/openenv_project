#!/usr/bin/env python3
"""
Baseline inference script for DataPipelineEnv.

This script provides a simple baseline agent that can be used to establish
reproducible performance benchmarks across all task difficulties.
"""

import argparse
import json
import random
import sys
from typing import Optional

from src.openenv_project import (
    DataPipelineEnv,
    Action,
    ActionType,
    get_task_config,
    grade_episode,
    list_tasks
)


class BaselineAgent:
    """
    Simple baseline agent for DataPipelineEnv.
    
    This agent uses a heuristic-based approach:
    1. Always schedule pending jobs if workers are available
    2. Retry failed jobs immediately
    3. Scale up when queue is building up (if scaling enabled)
    4. Scale down when resources are underutilized (if scaling enabled)
    5. Wait otherwise
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def get_action(self, env: DataPipelineEnv) -> Action:
        """
        Get the next action based on current state.
        
        Args:
            env: The environment instance.
            
        Returns:
            Action to take.
        """
        state = env.state  # Use the state attribute directly
        
        # Priority 1: Retry failed jobs
        if state.failed_jobs:
            # Retry the first failed job
            return Action(
                action_type=ActionType.RETRY_FAILED,
                job_id=state.failed_jobs[0].job_id
            )
        
        # Priority 2: Schedule pending jobs
        if state.pending_jobs:
            # Schedule the highest priority pending job
            sorted_jobs = sorted(
                state.pending_jobs, 
                key=lambda j: j.priority.value, 
                reverse=True
            )
            return Action(
                action_type=ActionType.SCHEDULE_JOB,
                job_id=sorted_jobs[0].job_id
            )
        
        # Priority 3: Scale up if queue might build up
        if env.task_config.enable_scaling:
            # Scale up if we have capacity and jobs are arriving
            if (state.active_workers < state.max_workers and 
                env.task_config.job_arrival_rate > 0.3):
                if state.resource_utilization > 70:
                    return Action(action_type=ActionType.SCALE_UP)
            
            # Scale down if resources are underutilized
            if (state.active_workers > 3 and 
                state.resource_utilization < 20 and
                not state.pending_jobs):
                return Action(action_type=ActionType.SCALE_DOWN)
        
        # Default: Wait for jobs to complete
        return Action(action_type=ActionType.WAIT)


def run_episode(
    task_id: str, 
    agent: BaselineAgent, 
    seed: Optional[int] = None,
    verbose: bool = False
) -> dict:
    """
    Run a single episode with the baseline agent.
    
    Args:
        task_id: The task to run.
        agent: The agent to use.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress.
        
    Returns:
        Dictionary with episode results.
    """
    # Get task configuration
    task_config = get_task_config(task_id)
    
    # Create environment
    env = DataPipelineEnv(task_config=task_config, seed=seed)
    
    # Reset environment
    state = env.reset()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_config.name}")
        print(f"Description: {task_config.description}")
        print(f"Max steps: {task_config.max_steps}")
        print(f"{'='*60}\n")
    
    # Run episode
    total_reward = 0.0
    step = 0
    
    while not env.done:
        # Get action from agent
        action = agent.get_action(env)
        
        # Execute action
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        if verbose and step % 10 == 0:
            print(f"Step {step}: reward={reward:.3f}, "
                  f"completed={info['total_jobs_completed']}, "
                  f"pending={state.queue_length}")
    
    # Grade the episode
    grade_result = grade_episode(env, task_id)
    
    # Compile results
    results = {
        "task_id": task_id,
        "seed": seed,
        "total_steps": step,
        "total_reward": total_reward,
        "score": grade_result.score,
        "passed": grade_result.passed,
        "metrics": grade_result.metrics.model_dump(),
        "feedback": grade_result.feedback
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Episode Complete!")
        print(f"Total Reward: {total_reward:.3f}")
        print(f"Score: {grade_result.score:.3f}")
        print(f"Passed: {grade_result.passed}")
        print(f"\nMetrics:")
        print(f"  Completion Rate: {grade_result.metrics.completion_rate:.1%}")
        print(f"  Resource Efficiency: {grade_result.metrics.resource_efficiency:.1%}")
        print(f"  SLA Compliance: {grade_result.metrics.sla_compliance:.1%}")
        print(f"  Cost Efficiency: {grade_result.metrics.cost_efficiency:.1%}")
        print(f"  Throughput: {grade_result.metrics.throughput:.3f} jobs/step")
        print(f"{'='*60}")
    
    return results


def run_benchmark(
    episodes_per_task: int = 10,
    seeds: Optional[list[int]] = None,
    verbose: bool = False
) -> dict:
    """
    Run benchmark across all tasks.
    
    Args:
        episodes_per_task: Number of episodes to run per task.
        seeds: List of seeds to use. If None, uses default seeds.
        verbose: Whether to print progress.
        
    Returns:
        Dictionary with benchmark results.
    """
    if seeds is None:
        seeds = list(range(episodes_per_task))
    
    agent = BaselineAgent(seed=42)  # Fixed seed for agent randomness
    
    all_results = []
    
    for task in list_tasks():
        task_id = task["id"]
        task_results = []
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# Running benchmark for task: {task['name']}")
            print(f"{'#'*60}")
        
        for i, seed in enumerate(seeds):
            result = run_episode(task_id, agent, seed=seed, verbose=verbose)
            task_results.append(result)
            all_results.append(result)
            
            if not verbose:
                print(f"  {task_id} episode {i+1}/{episodes_per_task}: "
                      f"score={result['score']:.3f}")
        
        # Print task summary
        scores = [r["score"] for r in task_results]
        passed = sum(1 for r in task_results if r["passed"])
        print(f"\n{task_id} Summary:")
        print(f"  Episodes: {len(task_results)}")
        print(f"  Mean Score: {sum(scores)/len(scores):.3f}")
        print(f"  Min Score: {min(scores):.3f}")
        print(f"  Max Score: {max(scores):.3f}")
        print(f"  Pass Rate: {passed/len(task_results):.1%}")
    
    # Overall summary
    all_scores = [r["score"] for r in all_results]
    all_passed = sum(1 for r in all_results if r["passed"])
    
    print(f"\n{'='*60}")
    print("Overall Benchmark Results:")
    print(f"  Total Episodes: {len(all_results)}")
    print(f"  Mean Score: {sum(all_scores)/len(all_scores):.3f}")
    print(f"  Pass Rate: {all_passed/len(all_results):.1%}")
    print(f"{'='*60}")
    
    return {
        "episodes_per_task": episodes_per_task,
        "seeds_used": seeds,
        "results": all_results
    }


def main():
    """Main entry point for the baseline inference script."""
    parser = argparse.ArgumentParser(
        description="Baseline inference for DataPipelineEnv"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task to run (default: all)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run per task (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Set base seed
    random.seed(args.seed)
    
    # Determine which tasks to run
    if args.task == "all":
        tasks = [t["id"] for t in list_tasks()]
    else:
        tasks = [args.task]
    
    # Generate seeds
    seeds = [args.seed + i for i in range(args.episodes)]
    
    # Run benchmark
    agent = BaselineAgent(seed=args.seed)
    all_results = []
    
    for task_id in tasks:
        task_results = []
        
        for seed in seeds:
            result = run_episode(
                task_id, 
                agent, 
                seed=seed, 
                verbose=args.verbose
            )
            task_results.append(result)
            all_results.append(result)
            
            if not args.verbose:
                print(f"{task_id} (seed={seed}): "
                      f"score={result['score']:.3f}, "
                      f"passed={result['passed']}")
        
        # Print task summary
        scores = [r["score"] for r in task_results]
        passed = sum(1 for r in task_results if r["passed"])
        print(f"\n{task_id} Summary:")
        print(f"  Mean Score: {sum(scores)/len(scores):.3f} ± "
              f"{(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5:.3f}")
        print(f"  Min/Max: {min(scores):.3f} / {max(scores):.3f}")
        print(f"  Pass Rate: {passed/len(task_results):.1%}")
    
    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "task": args.task,
                "episodes": args.episodes,
                "base_seed": args.seed,
                "results": all_results
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Final summary
    if len(tasks) > 1:
        all_scores = [r["score"] for r in all_results]
        all_passed = sum(1 for r in all_results if r["passed"])
        print(f"\nOverall: Mean={sum(all_scores)/len(all_scores):.3f}, "
              f"Pass Rate={all_passed/len(all_results):.1%}")


if __name__ == "__main__":
    main()