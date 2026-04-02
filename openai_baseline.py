#!/usr/bin/env python3
"""
OpenAI API baseline inference for DataPipelineEnv.

This script uses the OpenAI API to run a language model agent against
the DataPipelineEnv environment. It reads API credentials from the
OPENAI_API_KEY environment variable.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python openai_baseline.py --task all --model gpt-4 --episodes 3
"""

import argparse
import json
import os
import sys
import time
from typing import Optional, List, Dict, Any

from openai import OpenAI

from src.openenv_project import (
    DataPipelineEnv,
    Action,
    ActionType,
    get_task_config,
    grade_episode,
    list_tasks,
    EnvironmentState
)


# System prompt for the AI agent
SYSTEM_PROMPT = """You are an AI agent managing a data pipeline operations environment. Your goal is to efficiently process data jobs while meeting SLAs and managing resources.

## Environment Overview
You manage a cluster of workers that process data jobs. Jobs arrive over time and must be scheduled onto available workers. Each job has:
- A priority (low, medium, high, critical)
- Resource requirements (CPU, memory)
- An estimated duration
- A deadline (SLA)

## Available Actions
1. **schedule_job** - Schedule a pending job onto an available worker. Use this when there are pending jobs and available workers.
2. **wait** - Do nothing this step, let current jobs progress. Use when all workers are busy or no pending jobs.
3. **retry_failed** - Retry a failed job. Use when there are failed jobs.
4. **cancel_job** - Cancel a job to free resources. Use for low-priority jobs when resources are needed for higher priority work.
5. **scale_up** - Add more workers (costs money). Use when queue is building up and utilization is high.
6. **scale_down** - Remove idle workers (saves money). Use when resources are underutilized.
7. **allocate_resource** - Boost resources for a running job. Use for urgent jobs near their deadline.

## Strategy Guidelines
- Always prioritize critical and high-priority jobs
- Keep workers utilized - idle workers waste money
- Schedule jobs promptly to avoid queue buildup
- Retry failed jobs quickly
- Monitor SLA deadlines - jobs past their deadline hurt your score
- Scale up when demand is high, scale down when demand is low
- Balance throughput with cost efficiency

## Response Format
Respond with a JSON object containing:
{
    "action_type": "action_name",
    "job_id": "job_id_if_applicable",
    "reasoning": "brief explanation of your decision"
}

Choose the best action based on the current state."""


class OpenAIAgent:
    """AI agent powered by OpenAI's language models."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the OpenAI agent.
        
        Args:
            model: Model name to use (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
    
    def reset(self):
        """Reset the conversation history for a new episode."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    def format_state(self, state: EnvironmentState) -> str:
        """Format the environment state as a string for the LLM."""
        state_dict = state.model_dump()
        
        # Simplify for LLM
        summary = f"""Current State (timestep {state.time_step}):

Queue: {state.queue_length} pending jobs
Running: {len(state.running_jobs)} jobs in progress
Completed: {len(state.completed_jobs)} jobs done
Failed: {len(state.failed_jobs)} jobs failed
SLA Violations: {state.sla_violations}
Resource Utilization: {state.resource_utilization:.1f}%
Cost So Far: ${state.cost_so_far:.2f}
Active Workers: {state.active_workers}/{state.max_workers}

Pending Jobs:"""
        
        for job in state.pending_jobs[:5]:  # Limit to 5 for context
            summary += f"\n  - {job.job_id[:8]}: priority={job.priority.value}, type={job.job_type.value}, est_duration={job.estimated_duration}"
        
        if len(state.pending_jobs) > 5:
            summary += f"\n  ... and {len(state.pending_jobs) - 5} more"
        
        summary += "\n\nRunning Jobs:"
        for job in state.running_jobs[:5]:
            summary += f"\n  - {job.job_id[:8]}: progress={job.progress:.0%}, time_remaining={job.time_remaining}"
        
        if state.failed_jobs:
            summary += "\n\nFailed Jobs (need retry):"
            for job in state.failed_jobs[:3]:
                summary += f"\n  - {job.job_id[:8]}: priority={job.priority.value}, retries={job.retry_count}"
        
        return summary
    
    def get_action(self, state: EnvironmentState) -> Action:
        """
        Get an action from the LLM based on current state.
        
        Args:
            state: Current environment state
            
        Returns:
            Action to take
        """
        state_summary = self.format_state(state)
        
        user_message = f"""{state_summary}

What action should I take next? Respond with JSON."""
        
        self.conversation_history.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.1,
                max_tokens=200
            )
            
            assistant_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Parse the response
            try:
                # Extract JSON from response
                if "```json" in assistant_message:
                    assistant_message = assistant_message.split("```json")[1].split("```")[0].strip()
                elif "```" in assistant_message:
                    assistant_message = assistant_message.split("```")[1].split("```")[0].strip()
                
                action_data = json.loads(assistant_message)
                action_type = ActionType(action_data.get("action_type", "wait"))
                job_id = action_data.get("job_id")
                
                return Action(action_type=action_type, job_id=job_id)
            
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Failed to parse LLM response: {e}")
                print(f"Raw response: {assistant_message[:200]}")
                return Action(action_type=ActionType.WAIT)
        
        except Exception as e:
            print(f"Warning: OpenAI API error: {e}")
            return Action(action_type=ActionType.WAIT)


def run_episode_with_openai(
    task_id: str,
    agent: OpenAIAgent,
    seed: Optional[int] = None,
    verbose: bool = False
) -> dict:
    """
    Run a single episode with the OpenAI agent.
    
    Args:
        task_id: Task to run
        agent: OpenAI agent instance
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Dictionary with episode results
    """
    config = get_task_config(task_id)
    env = DataPipelineEnv(task_config=config, seed=seed)
    env.reset()
    agent.reset()
    
    total_reward = 0.0
    step = 0
    
    if verbose:
        print(f"\nRunning {task_id} task...")
    
    while not env.done:
        state = env.state
        
        # Get action from LLM
        action = agent.get_action(state)
        
        # Execute action
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        
        if verbose and step % 5 == 0:
            print(f"  Step {step}: reward={reward:.3f}, completed={info['total_jobs_completed']}")
        
        # Rate limiting - be nice to the API
        time.sleep(0.5)
    
    # Grade the episode
    grade_result = grade_episode(env, task_id)
    
    return {
        "task_id": task_id,
        "seed": seed,
        "total_steps": step,
        "total_reward": round(total_reward, 3),
        "score": round(grade_result.score, 3),
        "passed": grade_result.passed,
        "metrics": grade_result.metrics.model_dump(),
        "model": agent.model
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OpenAI API baseline inference for DataPipelineEnv"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task to run (default: all)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        choices=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        help="OpenAI model to use (default: gpt-4)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes per task (default: 1)"
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
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    # Create agent
    agent = OpenAIAgent(model=args.model)
    
    # Determine tasks
    if args.task == "all":
        tasks = [t["id"] for t in list_tasks()]
    else:
        tasks = [args.task]
    
    # Run episodes
    all_results = []
    
    for task_id in tasks:
        for ep in range(args.episodes):
            seed = args.seed + ep
            result = run_episode_with_openai(
                task_id, agent, seed=seed, verbose=args.verbose
            )
            all_results.append(result)
            
            print(f"{task_id} (episode {ep+1}): score={result['score']:.3f}, passed={result['passed']}")
    
    # Summary
    if len(all_results) > 1:
        by_task = {}
        for r in all_results:
            tid = r["task_id"]
            if tid not in by_task:
                by_task[tid] = []
            by_task[tid].append(r["score"])
        
        print("\n" + "="*50)
        print("Results Summary:")
        for tid, scores in by_task.items():
            avg = sum(scores) / len(scores)
            print(f"  {tid}: avg_score={avg:.3f}, episodes={len(scores)}")
        
        all_scores = [r["score"] for r in all_results]
        print(f"\nOverall: mean_score={sum(all_scores)/len(all_scores):.3f}")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model,
                "task": args.task,
                "episodes": args.episodes,
                "results": all_results
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()