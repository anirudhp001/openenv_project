"""
Agent graders for evaluating AI agent performance.

This module provides grading functionality that evaluates agent performance
on each task and produces scores between 0.0 and 1.0.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import EvaluationMetrics, JobPriority

if TYPE_CHECKING:
    from .environment import DataPipelineEnv


@dataclass
class GradeResult:
    """Result of grading an episode."""
    score: float  # Overall score 0.0-1.0
    passed: bool  # Whether the agent passed the task
    metrics: EvaluationMetrics
    feedback: str  # Human-readable feedback
    task_id: str


class BaseGrader:
    """Base class for task graders."""
    
    def __init__(self, task_id: str, success_threshold: float):
        self.task_id = task_id
        self.success_threshold = success_threshold
    
    def grade(self, env: 'DataPipelineEnv') -> GradeResult:
        """
        Grade the agent's performance.
        
        Args:
            env: The environment after an episode has completed.
            
        Returns:
            GradeResult with score and feedback.
        """
        raise NotImplementedError
    
    def _calculate_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall score from metrics."""
        return metrics.overall_score
    
    def _generate_feedback(self, metrics: EvaluationMetrics, score: float) -> str:
        """Generate human-readable feedback."""
        feedback_parts = []
        
        feedback_parts.append(f"Overall Score: {score:.2f}")
        feedback_parts.append(f"Completion Rate: {metrics.completion_rate:.1%}")
        feedback_parts.append(f"Resource Efficiency: {metrics.resource_efficiency:.1%}")
        feedback_parts.append(f"SLA Compliance: {metrics.sla_compliance:.1%}")
        feedback_parts.append(f"Cost Efficiency: {metrics.cost_efficiency:.1%}")
        feedback_parts.append(f"Throughput: {metrics.throughput:.2f} jobs/step")
        feedback_parts.append(f"Failure Recovery Rate: {metrics.failure_recovery_rate:.1%}")
        
        return "\n".join(feedback_parts)


class EasyGrader(BaseGrader):
    """
    Grader for Easy Task: Basic Job Scheduling.
    
    Focus: Completion rate is the primary metric.
    """
    
    def __init__(self):
        super().__init__("easy", success_threshold=0.7)
    
    def grade(self, env: 'DataPipelineEnv') -> GradeResult:
        metrics = env.get_metrics()
        
        # For easy task, completion rate is most important
        score = (
            metrics.completion_rate * 0.6 +
            metrics.resource_efficiency * 0.2 +
            metrics.sla_compliance * 0.2
        )
        
        # Normalize to 0-1 range
        score = min(1.0, max(0.0, score))
        
        passed = score >= self.success_threshold
        
        feedback = self._generate_feedback(metrics, score)
        if not passed:
            feedback += "\n\nSuggestions:"
            if metrics.completion_rate < 0.7:
                feedback += "\n- Focus on scheduling jobs more quickly"
                feedback += "\n- Ensure workers are always utilized"
            if metrics.resource_efficiency < 0.5:
                feedback += "\n- Better utilize available resources"
        
        return GradeResult(
            score=score,
            passed=passed,
            metrics=metrics,
            feedback=feedback,
            task_id=self.task_id
        )


class MediumGrader(BaseGrader):
    """
    Grader for Medium Task: Resource-Constrained Scheduling.
    
    Focus: Balanced score across completion, efficiency, and SLA compliance.
    """
    
    def __init__(self):
        super().__init__("medium", success_threshold=0.6)
    
    def grade(self, env: 'DataPipelineEnv') -> GradeResult:
        metrics = env.get_metrics()
        
        # Balanced scoring for medium task
        score = (
            metrics.completion_rate * 0.35 +
            metrics.resource_efficiency * 0.25 +
            metrics.sla_compliance * 0.25 +
            metrics.failure_recovery_rate * 0.15
        )
        
        # Normalize to 0-1 range
        score = min(1.0, max(0.0, score))
        
        passed = score >= self.success_threshold
        
        feedback = self._generate_feedback(metrics, score)
        if not passed:
            feedback += "\n\nSuggestions:"
            if metrics.completion_rate < 0.6:
                feedback += "\n- Improve job scheduling to increase completion rate"
            if metrics.failure_recovery_rate < 0.5:
                feedback += "\n- Retry failed jobs more aggressively"
            if metrics.sla_compliance < 0.7:
                feedback += "\n- Prioritize jobs closer to their deadlines"
        
        return GradeResult(
            score=score,
            passed=passed,
            metrics=metrics,
            feedback=feedback,
            task_id=self.task_id
        )


class HardGrader(BaseGrader):
    """
    Grader for Hard Task: Multi-Pipeline Optimization.
    
    Focus: Full composite score including cost efficiency.
    """
    
    def __init__(self):
        super().__init__("hard", success_threshold=0.5)
    
    def grade(self, env: 'DataPipelineEnv') -> GradeResult:
        metrics = env.get_metrics()
        
        # Full composite scoring for hard task
        score = (
            metrics.completion_rate * 0.30 +
            metrics.resource_efficiency * 0.20 +
            metrics.sla_compliance * 0.25 +
            metrics.cost_efficiency * 0.15 +
            metrics.failure_recovery_rate * 0.10
        )
        
        # Normalize to 0-1 range
        score = min(1.0, max(0.0, score))
        
        passed = score >= self.success_threshold
        
        feedback = self._generate_feedback(metrics, score)
        if not passed:
            feedback += "\n\nSuggestions:"
            if metrics.cost_efficiency < 0.5:
                feedback += "\n- Reduce unnecessary scaling to control costs"
                feedback += "\n- Consider scaling down during low-activity periods"
            if metrics.completion_rate < 0.5:
                feedback += "\n- Improve scheduling to handle more jobs"
            if metrics.sla_compliance < 0.5:
                feedback += "\n- Better prioritize critical and high-priority jobs"
            if metrics.resource_efficiency < 0.4:
                feedback += "\n- Scale infrastructure to match demand"
        
        return GradeResult(
            score=score,
            passed=passed,
            metrics=metrics,
            feedback=feedback,
            task_id=self.task_id
        )


# Registry of graders
GRADERS = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader
}


def get_grader(task_id: str) -> BaseGrader:
    """
    Get the appropriate grader for a task.
    
    Args:
        task_id: The task identifier.
        
    Returns:
        BaseGrader instance for the task.
        
    Raises:
        ValueError: If task_id is not recognized.
    """
    if task_id not in GRADERS:
        raise ValueError(
            f"Unknown task_id: {task_id}. "
            f"Available tasks: {list(GRADERS.keys())}"
        )
    return GRADERS[task_id]()


def grade_episode(env: 'DataPipelineEnv', task_id: str) -> GradeResult:
    """
    Grade a completed episode.
    
    Args:
        env: The environment after an episode has completed.
        task_id: The task that was being performed.
        
    Returns:
        GradeResult with score and feedback.
    """
    grader = get_grader(task_id)
    return grader.grade(env)


def grade_multiple_episodes(results: list[tuple[str, GradeResult]]) -> dict:
    """
    Grade multiple episodes and produce aggregate statistics.
    
    Args:
        results: List of (task_id, GradeResult) tuples.
        
    Returns:
        Dictionary with aggregate statistics.
    """
    if not results:
        return {"error": "No results provided"}
    
    # Group by task
    by_task = {}
    for task_id, result in results:
        if task_id not in by_task:
            by_task[task_id] = []
        by_task[task_id].append(result)
    
    # Calculate aggregates
    aggregates = {}
    for task_id, task_results in by_task.items():
        scores = [r.score for r in task_results]
        passes = [r.passed for r in task_results]
        
        aggregates[task_id] = {
            "num_episodes": len(task_results),
            "mean_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "pass_rate": sum(passes) / len(passes),
            "std_score": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5
        }
    
    # Overall statistics
    all_scores = [r.score for _, r in results]
    all_passes = [r.passed for _, r in results]
    
    aggregates["overall"] = {
        "total_episodes": len(results),
        "mean_score": sum(all_scores) / len(all_scores),
        "pass_rate": sum(all_passes) / len(all_passes)
    }
    
    return aggregates