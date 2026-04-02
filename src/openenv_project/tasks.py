"""
Task configurations for DataPipelineEnv.

This module defines the three task difficulty levels (easy, medium, hard)
with appropriate configurations for each.
"""

from .models import TaskConfig, PipelineType


def get_easy_task_config() -> TaskConfig:
    """
    Easy Task: Basic Job Scheduling
    
    Objective: Schedule incoming jobs to complete them successfully.
    - Simple job arrival pattern
    - Abundant resources
    - No failures
    - Single pipeline type (ETL)
    
    Grading: Completion rate of jobs within time limit
    """
    return TaskConfig(
        task_id="easy",
        name="Basic Job Scheduling",
        description=(
            "Schedule incoming data processing jobs with abundant resources. "
            "Jobs arrive at a steady rate and must be scheduled onto available workers. "
            "No failures occur in this task. Focus on maximizing job completion rate."
        ),
        max_steps=100,
        success_threshold=0.7,
        job_arrival_rate=0.3,  # 30% chance of job arrival per step
        failure_rate=0.0,      # No failures
        resource_scarcity=0.0, # Abundant resources
        enable_scaling=False,  # No scaling needed
        cost_per_worker=1.0,
        budget=None,
        pipeline_types=[PipelineType.ETL],  # Only ETL jobs
        sla_deadline_multiplier=3.0  # Generous deadlines
    )


def get_medium_task_config() -> TaskConfig:
    """
    Medium Task: Resource-Constrained Scheduling
    
    Objective: Manage limited resources while handling job failures.
    - Limited CPU/memory
    - Random job failures requiring retries
    - Competing job priorities
    - Multiple pipeline types
    
    Grading: Weighted score of completion rate, resource efficiency, and SLA compliance
    """
    return TaskConfig(
        task_id="medium",
        name="Resource-Constrained Scheduling",
        description=(
            "Manage data pipelines with limited resources. Jobs may fail randomly "
            "and require retries. Multiple pipeline types with different resource "
            "requirements compete for resources. Priority-based scheduling is important."
        ),
        max_steps=150,
        success_threshold=0.6,
        job_arrival_rate=0.4,  # Higher arrival rate
        failure_rate=0.05,     # 5% chance of failure per running job per step
        resource_scarcity=0.3, # Resources are more constrained
        enable_scaling=False,  # Still no scaling
        cost_per_worker=1.0,
        budget=None,
        pipeline_types=[
            PipelineType.ETL,
            PipelineType.DATA_VALIDATION,
            PipelineType.REPORTING
        ],
        sla_deadline_multiplier=2.0  # Tighter deadlines
    )


def get_hard_task_config() -> TaskConfig:
    """
    Hard Task: Multi-Pipeline Optimization
    
    Objective: Optimize multiple pipelines with cost constraints and dynamic scaling.
    - Multiple pipeline types with different priorities
    - Dynamic job arrival with peak periods
    - Cost budget constraints
    - Auto-scaling decisions required
    - ML training jobs with GPU requirements
    
    Grading: Composite score of throughput, cost efficiency, SLA compliance, and resource utilization
    """
    return TaskConfig(
        task_id="hard",
        name="Multi-Pipeline Optimization",
        description=(
            "Optimize a complex data platform with multiple pipeline types including "
            "ML training jobs. Manage costs while meeting SLAs. Dynamic scaling is "
            "required to handle peak loads. Budget constraints apply. This task "
            "requires balancing throughput, cost, and reliability."
        ),
        max_steps=200,
        success_threshold=0.5,
        job_arrival_rate=0.5,  # High arrival rate with potential bursts
        failure_rate=0.08,     # 8% failure rate
        resource_scarcity=0.5, # Significantly constrained resources
        enable_scaling=True,   # Auto-scaling is essential
        cost_per_worker=2.0,   # Higher cost for workers
        budget=100.0,          # Budget constraint
        pipeline_types=[
            PipelineType.ETL,
            PipelineType.ML_TRAINING,
            PipelineType.DATA_VALIDATION,
            PipelineType.REPORTING,
            PipelineType.BACKUP
        ],
        sla_deadline_multiplier=1.5,  # Very tight deadlines
        worker_failure_rate=0.02  # 2% chance of worker crash per step
    )


# Registry of all task configurations
TASK_CONFIGS = {
    "easy": get_easy_task_config,
    "medium": get_medium_task_config,
    "hard": get_hard_task_config
}


def get_task_config(task_id: str) -> TaskConfig:
    """
    Get task configuration by ID.
    
    Args:
        task_id: The task identifier ("easy", "medium", "hard").
        
    Returns:
        TaskConfig for the specified task.
        
    Raises:
        ValueError: If task_id is not recognized.
    """
    if task_id not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task_id: {task_id}. "
            f"Available tasks: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_id]()


def list_tasks() -> list[dict]:
    """
    List all available tasks.
    
    Returns:
        List of dictionaries with task information.
    """
    return [
        {
            "id": config().task_id,
            "name": config().name,
            "description": config().description,
            "max_steps": config().max_steps,
            "success_threshold": config().success_threshold
        }
        for config in TASK_CONFIGS.values()
    ]