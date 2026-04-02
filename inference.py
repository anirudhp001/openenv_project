#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
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

# Mandatory environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = os.getenv("BENCHMARK", "openenv_project")
MAX_STEPS = 50

# System prompt for the AI agent
SYSTEM_PROMPT = """You are an AI agent managing a data pipeline operations environment. Your goal is to efficiently process data jobs while meeting SLAs and managing resources.

## Environment Overview
You manage a cluster of workers that process data jobs. Jobs arrive over time and must be scheduled onto available workers. Each job has:
- A priority (low, medium, high, critical)
- Resource requirements (CPU, memory)
- An estimated duration
- A deadline (SLA)

## Available Actions
1. schedule_job - Schedule a pending job onto an available worker.
2. wait - Do nothing this step, let current jobs progress.
3. retry_failed - Retry a failed job.
4. cancel_job - Cancel a job to free resources.
5. scale_up - Add more workers.
6. scale_down - Remove idle workers.
7. allocate_resource - Boost resources for a running job.

## Response Format
Respond with a JSON object containing:
{
    "action_type": "action_name",
    "job_id": "job_id_if_applicable"
}
Choose the best action based on the current state."""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", 
        flush=True
    )

class OpenAIAgent:
    def __init__(self):
        token = HF_TOKEN
        if not token:
            print("Warning: HF_TOKEN not found. Using empty string for local compatibility.")
            token = "dummy-token"
            
        self.client = OpenAI(base_url=API_BASE_URL, api_key=token)
        self.model = MODEL_NAME
        self.conversation_history = []
    
    def reset(self):
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def format_state(self, state: EnvironmentState) -> str:
        summary = f"Current State:\nQueue: {state.queue_length} pending jobs\nRunning: {len(state.running_jobs)} jobs\nFailed: {len(state.failed_jobs)} jobs"
        if state.pending_jobs:
            summary += f"\nFirst pending: {state.pending_jobs[0].job_id}"
        if state.failed_jobs:
            summary += f"\nFirst failed: {state.failed_jobs[0].job_id}"
        return summary
    
    def get_action_data(self, state: EnvironmentState) -> tuple[Action, str, Optional[str]]:
        state_summary = self.format_state(state)
        user_message = f"{state_summary}\nWhat action should I take next? Respond with exactly one JSON object."
        self.conversation_history.append({"role": "user", "content": user_message})
        
        error = None
        action_text = "wait"
        action_obj = Action(action_type=ActionType.WAIT)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.1,
                max_tokens=150,
                stream=False
            )
            
            assistant_message = (response.choices[0].message.content or "").strip()
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            try:
                # Basic JSON extraction
                content = assistant_message
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                action_data = json.loads(content)
                act_type_str = action_data.get("action_type", "wait")
                job_id = action_data.get("job_id")
                
                action_type = ActionType(act_type_str)
                action_obj = Action(action_type=action_type, job_id=job_id)
                
                action_text = act_type_str
                if job_id:
                    action_text += f"({job_id})"
                    
            except Exception as e:
                error = f"Parse Error: {str(e)}"
                action_text = "parse-error-wait"
        except Exception as e:
            error = f"API Error: {str(e)}"
            action_text = "api-error-wait"
            
        return action_obj, action_text, error


def main():
    agent = OpenAIAgent()
    tasks = ["easy", "medium", "hard"]
    
    for task_id in tasks:
        # Initialize episode logs
        log_start(task=task_id, env=BENCHMARK, model=agent.model)
        
        config = get_task_config(task_id)
        config.max_steps = MAX_STEPS 
        env = DataPipelineEnv(task_config=config, seed=42)
        env.reset()
        agent.reset()
        
        rewards = []
        steps_taken = 0
        score = 0.0
        success = False

        try:
            for step in range(1, MAX_STEPS + 1):
                if env.done:
                    break
                
                action_obj, action_text, action_error = agent.get_action_data(env.state)
                
                # Execute environment step
                _, reward, done, _ = env.step(action_obj)
                
                # Save step data
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_text, reward=reward, done=done, error=action_error)
                
                if done:
                    break
                    
                time.sleep(0.1) # Be nice to API and environment
                
            grade_result = grade_episode(env, task_id)
            score = grade_result.score
            success = grade_result.passed

        except Exception as e:
            print(f"[DEBUG] Runtime error occurred: {e}", file=sys.stderr)
            
        finally:
            # We don't have an async env.close(), but we log the end predictably.
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
