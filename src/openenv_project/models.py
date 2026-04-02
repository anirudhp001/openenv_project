"""
Typed data models for the DataPipelineEnv environment using Pydantic.

This module defines all the core data structures used throughout the environment,
including jobs, resources, actions, and environment state.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid

from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    """Status of a job in the pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Priority levels for jobs."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    @property
    def weight(self) -> float:
        """Get numeric weight for priority."""
        return {"low": 1.0, "medium": 2.0, "high": 3.0, "critical": 4.0}[self.value]


class PipelineType(str, Enum):
    """Types of data pipelines."""
    ETL = "etl"
    ML_TRAINING = "ml_training"
    DATA_VALIDATION = "validation"
    REPORTING = "reporting"
    BACKUP = "backup"


class ActionType(str, Enum):
    """Available actions for the agent."""
    SCHEDULE_JOB = "schedule_job"
    ALLOCATE_RESOURCE = "allocate_resource"
    RETRY_FAILED = "retry_failed"
    CANCEL_JOB = "cancel_job"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    WAIT = "wait"


class Resources(BaseModel):
    """Represents available compute resources."""
    cpu_cores: float = Field(default=8.0, ge=0, description="Number of CPU cores")
    memory_gb: float = Field(default=32.0, ge=0, description="Memory in GB")
    network_bandwidth_mbps: float = Field(default=1000.0, ge=0, description="Network bandwidth in Mbps")
    gpu_units: float = Field(default=0.0, ge=0, description="GPU units available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cpu_cores": 8.0,
                "memory_gb": 32.0,
                "network_bandwidth_mbps": 1000.0,
                "gpu_units": 0.0
            }
        }
    
    def can_allocate(self, required: 'Resources') -> bool:
        """Check if this resource pool can satisfy the required resources."""
        return (
            self.cpu_cores >= required.cpu_cores and
            self.memory_gb >= required.memory_gb and
            self.network_bandwidth_mbps >= required.network_bandwidth_mbps and
            self.gpu_units >= required.gpu_units
        )
    
    def allocate(self, required: 'Resources') -> 'Resources':
        """Allocate resources and return remaining available resources."""
        if not self.can_allocate(required):
            raise ValueError("Insufficient resources to allocate")
        return Resources(
            cpu_cores=self.cpu_cores - required.cpu_cores,
            memory_gb=self.memory_gb - required.memory_gb,
            network_bandwidth_mbps=self.network_bandwidth_mbps - required.network_bandwidth_mbps,
            gpu_units=self.gpu_units - required.gpu_units
        )
    
    def deallocate(self, released: 'Resources') -> 'Resources':
        """Release allocated resources back to the pool."""
        return Resources(
            cpu_cores=self.cpu_cores + released.cpu_cores,
            memory_gb=self.memory_gb + released.memory_gb,
            network_bandwidth_mbps=self.network_bandwidth_mbps + released.network_bandwidth_mbps,
            gpu_units=self.gpu_units + released.gpu_units
        )
    
    def utilization_percentage(self, total: 'Resources') -> float:
        """Calculate overall utilization percentage compared to total capacity."""
        if total.cpu_cores == 0 and total.memory_gb == 0:
            return 0.0
        
        utilizations = []
        if total.cpu_cores > 0:
            utilizations.append(1.0 - self.cpu_cores / total.cpu_cores)
        if total.memory_gb > 0:
            utilizations.append(1.0 - self.memory_gb / total.memory_gb)
        
        return sum(utilizations) / len(utilizations) * 100.0 if utilizations else 0.0


class Job(BaseModel):
    """Represents a data processing job."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique job identifier")
    job_type: PipelineType = Field(default=PipelineType.ETL, description="Type of pipeline job")
    priority: JobPriority = Field(default=JobPriority.MEDIUM, description="Job priority level")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Current job status")
    required_resources: Resources = Field(default_factory=Resources, description="Resources needed")
    estimated_duration: int = Field(default=10, ge=1, description="Estimated duration in timesteps")
    elapsed_time: int = Field(default=0, ge=0, description="Time elapsed since start")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    deadline: Optional[int] = Field(default=None, description="Deadline timestep")
    arrived_at: int = Field(default=0, ge=0, description="Timestep when job arrived")
    started_at: Optional[int] = Field(default=None, ge=0, description="Timestep when job started")
    completed_at: Optional[int] = Field(default=None, ge=0, description="Timestep when job completed")
    data_size_gb: float = Field(default=1.0, ge=0, description="Size of data to process in GB")
    progress: float = Field(default=0.0, ge=0, le=1, description="Completion percentage")
    assigned_worker: Optional[str] = Field(default=None, description="Assigned worker ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_001",
                "job_type": "etl",
                "priority": "high",
                "status": "pending",
                "estimated_duration": 10,
                "deadline": 20,
                "data_size_gb": 5.0
            }
        }
    
    @property
    def is_overdue(self) -> bool:
        """Check if job has exceeded its deadline."""
        if self.deadline is None:
            return False
        return self.elapsed_time > self.deadline
    
    @property
    def time_remaining(self) -> Optional[int]:
        """Get remaining time until deadline."""
        if self.deadline is None:
            return None
        return max(0, self.deadline - self.elapsed_time)
    
    @property
    def wait_time(self) -> int:
        """Calculate total wait time (time spent pending)."""
        if self.started_at is not None:
            return self.started_at - self.arrived_at
        return self.elapsed_time


class Action(BaseModel):
    """Represents an action taken by the agent."""
    action_type: ActionType = Field(..., description="Type of action to take")
    job_id: Optional[str] = Field(default=None, description="Target job ID if applicable")
    resource_amount: Optional[Resources] = Field(default=None, description="Resource amount for allocation")
    worker_id: Optional[str] = Field(default=None, description="Target worker ID if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "schedule_job",
                "job_id": "job_001"
            }
        }


class EnvironmentState(BaseModel):
    """Complete state of the environment at a given timestep."""
    pending_jobs: List[Job] = Field(default_factory=list, description="Jobs waiting to be processed")
    running_jobs: List[Job] = Field(default_factory=list, description="Currently executing jobs")
    completed_jobs: List[Job] = Field(default_factory=list, description="Successfully completed jobs")
    failed_jobs: List[Job] = Field(default_factory=list, description="Failed jobs awaiting retry")
    available_resources: Resources = Field(default_factory=Resources, description="Available resources")
    total_resources: Resources = Field(default_factory=Resources, description="Total resource capacity")
    queue_length: int = Field(default=0, ge=0, description="Number of jobs in queue")
    avg_wait_time: float = Field(default=0.0, ge=0, description="Average job wait time")
    resource_utilization: float = Field(default=0.0, ge=0, le=100, description="Resource utilization %")
    cost_so_far: float = Field(default=0.0, ge=0, description="Accumulated operational cost")
    sla_violations: int = Field(default=0, ge=0, description="Number of SLA breaches")
    time_step: int = Field(default=0, ge=0, description="Current simulation timestep")
    active_workers: int = Field(default=0, ge=0, description="Number of active workers")
    max_workers: int = Field(default=10, ge=1, description="Maximum number of workers")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pending_jobs": [],
                "running_jobs": [],
                "completed_jobs": [],
                "failed_jobs": [],
                "queue_length": 0,
                "time_step": 0
            }
        }


class TaskConfig(BaseModel):
    """Configuration for a specific task."""
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    description: str = Field(..., description="Task description")
    max_steps: int = Field(..., ge=1, description="Maximum steps per episode")
    success_threshold: float = Field(..., ge=0, le=1, description="Score threshold for success")
    job_arrival_rate: float = Field(default=0.5, ge=0, le=1, description="Job arrival probability")
    failure_rate: float = Field(default=0.0, ge=0, le=1, description="Job failure probability")
    resource_scarcity: float = Field(default=0.0, ge=0, le=1, description="Resource scarcity level")
    enable_scaling: bool = Field(default=False, description="Enable auto-scaling")
    cost_per_worker: float = Field(default=1.0, ge=0, description="Cost per worker per step")
    budget: Optional[float] = Field(default=None, ge=0, description="Budget constraint")
    pipeline_types: List[PipelineType] = Field(default_factory=lambda: [PipelineType.ETL])
    sla_deadline_multiplier: float = Field(default=2.0, ge=1, description="SLA deadline multiplier")
    worker_failure_rate: float = Field(default=0.0, ge=0, le=1, description="Probability of a worker crashing")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "easy",
                "name": "Basic Job Scheduling",
                "max_steps": 100,
                "success_threshold": 0.7
            }
        }


class EvaluationMetrics(BaseModel):
    """Metrics for evaluating agent performance."""
    completion_rate: float = Field(default=0.0, ge=0, le=1, description="Job completion rate")
    resource_efficiency: float = Field(default=0.0, ge=0, le=1, description="Resource efficiency")
    sla_compliance: float = Field(default=0.0, ge=0, le=1, description="SLA compliance rate")
    cost_efficiency: float = Field(default=0.0, ge=0, le=1, description="Cost efficiency")
    average_wait_time: float = Field(default=0.0, ge=0, description="Average wait time")
    throughput: float = Field(default=0.0, ge=0, description="Jobs completed per step")
    failure_recovery_rate: float = Field(default=0.0, ge=0, le=1, description="Failure recovery rate")
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        return (
            self.completion_rate * 0.4 +
            self.resource_efficiency * 0.2 +
            self.sla_compliance * 0.25 +
            self.cost_efficiency * 0.15
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "completion_rate": 0.8,
                "resource_efficiency": 0.6,
                "sla_compliance": 0.9,
                "cost_efficiency": 0.7
            }
        }


class GradeResult(BaseModel):
    """Result of grading an episode."""
    score: float = Field(..., ge=0, le=1, description="Overall score 0.0-1.0")
    passed: bool = Field(..., description="Whether the agent passed the task")
    metrics: EvaluationMetrics = Field(..., description="Detailed metrics")
    feedback: str = Field(..., description="Human-readable feedback")
    task_id: str = Field(..., description="Task identifier")