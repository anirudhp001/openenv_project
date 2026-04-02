"""
Reward function helpers for DataPipelineEnv.

This module provides additional reward function utilities and alternative
reward shaping strategies that can be used with the environment.
"""

from typing import TYPE_CHECKING, Callable

from .models import Job, JobPriority, JobStatus, ActionType, Action

if TYPE_CHECKING:
    from .environment import DataPipelineEnv
    from .models import EnvironmentState


class RewardShaper:
    """
    Reward shaping utilities for the DataPipelineEnv.
    
    Provides different reward strategies that can be used to train agents
    with different optimization goals.
    """
    
    @staticmethod
    def sparse_reward(state: 'EnvironmentState', action: Action, 
                      prev_state: 'EnvironmentState') -> float:
        """
        Sparse reward: only reward final completion.
        
        This is the hardest reward scheme for RL as it provides minimal feedback.
        """
        reward = 0.0
        
        # Only reward completed jobs
        newly_completed = len(state.completed_jobs) - len(prev_state.completed_jobs)
        reward += newly_completed * 1.0
        
        return reward
    
    @staticmethod
    def dense_reward(state: 'EnvironmentState', action: Action,
                     prev_state: 'EnvironmentState') -> float:
        """
        Dense reward: provide frequent feedback for all progress.
        
        This is the easiest reward scheme for RL as it provides constant guidance.
        """
        reward = 0.0
        
        # Reward for completed jobs
        newly_completed = len(state.completed_jobs) - len(prev_state.completed_jobs)
        reward += newly_completed * 1.0
        
        # Reward for job progress
        for job in state.running_jobs:
            reward += job.progress * 0.05
        
        # Reward for scheduling jobs
        if action.action_type == ActionType.SCHEDULE_JOB:
            if state.pending_jobs and len(state.pending_jobs) < len(prev_state.pending_jobs):
                reward += 0.2
        
        # Penalty for pending jobs (encourage clearing queue)
        reward -= len(state.pending_jobs) * 0.02
        
        # Penalty for failed jobs
        reward -= len(state.failed_jobs) * 0.1
        
        # Penalty for SLA violations
        sla_increase = state.sla_violations - prev_state.sla_violations
        reward -= sla_increase * 0.5
        
        return reward
    
    @staticmethod
    def priority_aware_reward(state: 'EnvironmentState', action: Action,
                               prev_state: 'EnvironmentState') -> float:
        """
        Priority-aware reward: weight rewards by job priority.
        
        Encourages the agent to prioritize high-value jobs.
        """
        reward = 0.0
        
        # Calculate priority-weighted completion reward
        for job in state.completed_jobs:
            if job.completed_at == state.time_step:  # Newly completed
                priority_weight = job.priority.value / JobPriority.CRITICAL.value
                reward += priority_weight * 1.5
        
        # Priority-weighted progress reward
        for job in state.running_jobs:
            priority_weight = job.priority.value / JobPriority.CRITICAL.value
            reward += job.progress * priority_weight * 0.1
        
        # Penalty for high-priority jobs waiting
        for job in state.pending_jobs:
            if job.priority in [JobPriority.HIGH, JobPriority.CRITICAL]:
                reward -= 0.15
        
        # Big penalty for high-priority SLA violations
        for job in state.failed_jobs:
            if job.priority in [JobPriority.HIGH, JobPriority.CRITICAL]:
                reward -= 0.3
        
        return reward
    
    @staticmethod
    def cost_aware_reward(state: 'EnvironmentState', action: Action,
                          prev_state: 'EnvironmentState',
                          cost_weight: float = 0.1) -> float:
        """
        Cost-aware reward: balance throughput with cost efficiency.
        
        Encourages the agent to achieve results while minimizing costs.
        """
        reward = 0.0
        
        # Base completion reward
        newly_completed = len(state.completed_jobs) - len(prev_state.completed_jobs)
        reward += newly_completed * 1.0
        
        # Cost penalty
        cost_increase = state.cost_so_far - prev_state.cost_so_far
        reward -= cost_increase * cost_weight
        
        # Efficiency bonus (more jobs per unit cost)
        if state.cost_so_far > 0:
            efficiency = len(state.completed_jobs) / state.cost_so_far
            reward += min(efficiency * 0.1, 0.5)  # Cap the efficiency bonus
        
        # Penalty for idle resources (wasted cost)
        if state.resource_utilization < 30 and state.pending_jobs:
            reward -= 0.1
        
        return reward
    
    @staticmethod
    def sla_focused_reward(state: 'EnvironmentState', action: Action,
                           prev_state: 'EnvironmentState') -> float:
        """
        SLA-focused reward: prioritize meeting deadlines.
        
        Strongly penalizes SLA violations and rewards on-time completion.
        """
        reward = 0.0
        
        # Reward for on-time completions
        for job in state.completed_jobs:
            if job.completed_at == state.time_step:
                if not job.is_overdue:
                    reward += 1.5  # Bonus for on-time
                else:
                    reward += 0.5  # Reduced reward for late
        
        # Heavy penalty for new SLA violations
        sla_increase = state.sla_violations - prev_state.sla_violations
        reward -= sla_increase * 1.0
        
        # Penalty for jobs approaching deadline
        for job in state.pending_jobs + state.running_jobs:
            if job.deadline is not None:
                urgency = 1.0 - (job.time_remaining / job.deadline) if job.time_remaining else 1.0
                if urgency > 0.8:  # Very urgent
                    reward -= 0.2
        
        # Reward for completing urgent jobs
        for job in state.running_jobs:
            if job.deadline is not None and job.time_remaining is not None:
                urgency = 1.0 - (job.time_remaining / job.deadline)
                reward += job.progress * urgency * 0.2
        
        return reward


# Registry of reward functions
REWARD_FUNCTIONS = {
    "default": None,  # Use environment's default
    "sparse": RewardShaper.sparse_reward,
    "dense": RewardShaper.dense_reward,
    "priority_aware": RewardShaper.priority_aware_reward,
    "cost_aware": RewardShaper.cost_aware_reward,
    "sla_focused": RewardShaper.sla_focused_reward
}


def get_reward_function(name: str) -> Callable:
    """
    Get a reward function by name.
    
    Args:
        name: Name of the reward function.
        
    Returns:
        Reward function callable.
        
    Raises:
        ValueError: If reward function name is not recognized.
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]


def list_reward_functions() -> list[str]:
    """List available reward function names."""
    return list(REWARD_FUNCTIONS.keys())