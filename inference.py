#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from datetime import datetime
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

def log_structured(event_type: str, data: dict):
    """Log structured events as required by Phase 3."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    log_entry = {
        "timestamp": timestamp,
        "event": event_type.strip("[]"),
        **data
    }
    print(f"{event_type} {timestamp} - {json.dumps(log_entry)}", flush=True)


class OpenAIAgent:
    def __init__(self):
        api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        model_name = os.environ.get("MODEL_NAME", "gpt-4")
        hf_token = os.environ.get("HF_TOKEN")
        
        if not hf_token:
            print("Warning: HF_TOKEN not found. Using empty string for compatibility if local.")
            hf_token = "dummy-token"
            
        self.client = OpenAI(base_url=api_base_url, api_key=hf_token)
        self.model = model_name
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
    
    def get_action(self, state: EnvironmentState) -> Action:
        state_summary = self.format_state(state)
        user_message = f"{state_summary}\nWhat action should I take next? Respond with JSON."
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
            
            try:
                if "```json" in assistant_message:
                    assistant_message = assistant_message.split("```json")[1].split("```")[0].strip()
                elif "```" in assistant_message:
                    assistant_message = assistant_message.split("```")[1].split("```")[0].strip()
                
                action_data = json.loads(assistant_message)
                action_type = ActionType(action_data.get("action_type", "wait"))
                return Action(action_type=action_type, job_id=action_data.get("job_id"))
            except Exception:
                return Action(action_type=ActionType.WAIT)
        except Exception as e:
            print(f"API Error: {e}", file=sys.stderr)
            return Action(action_type=ActionType.WAIT)


def main():
    agent = OpenAIAgent()
    
    tasks = ["easy", "medium", "hard"]
    
    for task_id in tasks:
        log_structured("[START]", {"task_id": task_id, "model": agent.model})
        config = get_task_config(task_id)
        # Limit to 50 max steps to finish quickly
        config.max_steps = 50 
        env = DataPipelineEnv(task_config=config, seed=42)
        env.reset()
        agent.reset()
        
        total_reward = 0.0
        step = 0
        
        while not env.done:
            state = env.state
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            log_structured("[STEP]", {
                "step": step,
                "action": action.action_type.value,
                "reward": reward,
                "done": done
            })
            
            time.sleep(0.1) # Be nice to resources
            
        grade_result = grade_episode(env, task_id)
        log_structured("[END]", {
            "task_id": task_id,
            "total_steps": step,
            "score": grade_result.score,
            "passed": grade_result.passed
        })

if __name__ == "__main__":
    main()
