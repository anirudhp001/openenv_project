"""
Main environment implementation for DataPipelineEnv.

This module implements the core OpenEnv environment with step(), reset(), and state() API.
"""

import random
import uuid
from typing import Optional, Tuple
from dataclasses import dataclass, field

from .models import (
    Job, JobStatus, JobPriority, PipelineType, ActionType, Action,
    Resources, EnvironmentState, TaskConfig, EvaluationMetrics
)


@dataclass
class Worker:
    """Represents a compute worker that can execute jobs."""
    worker_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_active: bool = True
    current_job: Optional[Job] = None
    cpu_cores: float = 4.0
    memory_gb: float = 16.0
    network_bandwidth_mbps: float = 500.0
    gpu_units: float = 0.0
    
    @property
    def is_busy(self) -> bool:
        """Check if worker is currently executing a job."""
        return self.current_job is not None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "worker_id": self.worker_id,
            "is_active": self.is_active,
            "is_busy": self.is_busy,
            "current_job_id": self.current_job.job_id if self.current_job else None,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
            "gpu_units": self.gpu_units
        }


class DataPipelineEnv:
    """
    Data Pipeline Management Environment.
    
    An OpenEnv-compatible environment where an AI agent learns to manage
    data processing pipelines by scheduling jobs, allocating resources,
    handling failures, and optimizing throughput.
    """
    
    def __init__(self, task_config: Optional[TaskConfig] = None, seed: Optional[int] = None):
        """
        Initialize the environment.
        
        Args:
            task_config: Configuration for the task. If None, uses default easy task.
            seed: Random seed for reproducibility.
        """
        self.task_config = task_config or self._default_easy_config()
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Environment state
        self.state: Optional[EnvironmentState] = None
        self.workers: list[Worker] = []
        self._current_step: int = 0
        self._done: bool = False
        self._total_cost: float = 0.0
        self._sla_violations: int = 0
        self._job_counter: int = 0
        
        # Statistics tracking
        self._total_jobs_arrived: int = 0
        self._total_jobs_completed: int = 0
        self._total_jobs_failed: int = 0
        self._total_wait_time: float = 0.0
        self._cumulative_utilization: float = 0.0
        
    @staticmethod
    def _default_easy_config() -> TaskConfig:
        """Return default easy task configuration."""
        return TaskConfig(
            task_id="easy",
            name="Basic Job Scheduling",
            description="Schedule jobs with abundant resources",
            max_steps=100,
            success_threshold=0.7,
            job_arrival_rate=0.3,
            failure_rate=0.0,
            resource_scarcity=0.0,
            enable_scaling=False,
            cost_per_worker=1.0,
            budget=None,
            pipeline_types=[PipelineType.ETL],
            sla_deadline_multiplier=3.0
        )
    
    def reset(self) -> EnvironmentState:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial environment state.
        """
        self.rng = random.Random(self.seed)
        self._current_step = 0
        self._done = False
        self._total_cost = 0.0
        self._sla_violations = 0
        self._job_counter = 0
        self._total_jobs_arrived = 0
        self._total_jobs_completed = 0
        self._total_jobs_failed = 0
        self._total_wait_time = 0.0
        self._cumulative_utilization = 0.0
        
        # Initialize workers based on task config
        self._initialize_workers()
        
        # Initialize state
        self.state = EnvironmentState(
            pending_jobs=[],
            running_jobs=[],
            completed_jobs=[],
            failed_jobs=[],
            available_resources=self._get_total_resources(),
            total_resources=self._get_total_resources(),
            queue_length=0,
            avg_wait_time=0.0,
            resource_utilization=0.0,
            cost_so_far=0.0,
            sla_violations=0,
            time_step=0,
            active_workers=len([w for w in self.workers if w.is_active]),
            max_workers=len(self.workers)
        )
        
        return self.state
    
    def _initialize_workers(self):
        """Initialize worker pool based on task configuration."""
        self.workers = []
        num_workers = 5  # Base number of workers
        
        for i in range(num_workers):
            self.workers.append(Worker(
                worker_id=f"worker_{i}",
                is_active=True,
                cpu_cores=4.0,
                memory_gb=16.0,
                network_bandwidth_mbps=500.0,
                gpu_units=0.0
            ))
    
    def _get_total_resources(self) -> Resources:
        """Get total available resources from active workers."""
        total_cpu = sum(w.cpu_cores for w in self.workers if w.is_active and not w.is_busy)
        total_mem = sum(w.memory_gb for w in self.workers if w.is_active and not w.is_busy)
        total_net = sum(w.network_bandwidth_mbps for w in self.workers if w.is_active and not w.is_busy)
        total_gpu = sum(w.gpu_units for w in self.workers if w.is_active and not w.is_busy)
        
        return Resources(
            cpu_cores=total_cpu,
            memory_gb=total_mem,
            network_bandwidth_mbps=total_net,
            gpu_units=total_gpu
        )
    
    def step(self, action: Action) -> Tuple[EnvironmentState, float, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute.
            
        Returns:
            Tuple of (new_state, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        # Execute the action
        self._execute_action(action)
        
        # Process running jobs
        self._process_running_jobs()
        
        # Generate new jobs
        self._generate_jobs()
        
        # Check for failures
        self._check_worker_failures()
        self._check_job_failures()
        
        # Check for SLA violations
        self._check_sla_violations()
        
        # Update state
        self._update_state()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Increment step
        self._current_step += 1
        self.state.time_step = self._current_step
        
        # Check if episode is done
        self._done = self._current_step >= self.task_config.max_steps
        
        # Prepare info
        info = {
            "step": self._current_step,
            "total_jobs_arrived": self._total_jobs_arrived,
            "total_jobs_completed": self._total_jobs_completed,
            "total_jobs_failed": self._total_jobs_failed,
            "sla_violations": self._sla_violations,
            "current_cost": self._total_cost
        }
        
        return self.state, reward, self._done, info
    
    def get_state(self) -> EnvironmentState:
        """
        Get the current environment state.
        
        Returns:
            Current environment state.
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.state
    
    def _execute_action(self, action: Action):
        """Execute the given action."""
        if action.action_type == ActionType.SCHEDULE_JOB:
            self._action_schedule_job(action)
        elif action.action_type == ActionType.ALLOCATE_RESOURCE:
            self._action_allocate_resource(action)
        elif action.action_type == ActionType.RETRY_FAILED:
            self._action_retry_failed(action)
        elif action.action_type == ActionType.CANCEL_JOB:
            self._action_cancel_job(action)
        elif action.action_type == ActionType.SCALE_UP:
            self._action_scale_up(action)
        elif action.action_type == ActionType.SCALE_DOWN:
            self._action_scale_down(action)
        elif action.action_type == ActionType.WAIT:
            pass  # Wait does nothing, jobs will progress naturally
    
    def _action_schedule_job(self, action: Action):
        """Schedule a pending job onto available workers."""
        # Find pending jobs to schedule
        pending_jobs = self.state.pending_jobs if self.state else []
        
        if not pending_jobs:
            return
        
        # Sort by priority (highest first)
        sorted_jobs = sorted(pending_jobs, key=lambda j: j.priority.value, reverse=True)
        
        for job in sorted_jobs:
            # Find an available worker
            for worker in self.workers:
                if worker.is_active and not worker.is_busy:
                    # Check if worker has enough resources
                    if (worker.cpu_cores >= job.required_resources.cpu_cores and
                        worker.memory_gb >= job.required_resources.memory_gb):
                        # Assign job to worker
                        job.status = JobStatus.RUNNING
                        job.started_at = self._current_step
                        job.assigned_worker = worker.worker_id
                        worker.current_job = job
                        
                        # Move job from pending to running
                        if self.state:
                            self.state.pending_jobs.remove(job)
                            self.state.running_jobs.append(job)
                        break
    
    def _action_allocate_resource(self, action: Action):
        """Allocate additional resources to a running job."""
        if action.job_id is None or action.resource_amount is None:
            return
        
        # Find the job
        for job in self.state.running_jobs if self.state else []:
            if job.job_id == action.job_id:
                # For simplicity, this action boosts job progress
                job.progress = min(1.0, job.progress + 0.1)
                break
    
    def _action_retry_failed(self, action: Action):
        """Retry a failed job."""
        if action.job_id is None:
            return
        
        failed_jobs = self.state.failed_jobs if self.state else []
        
        for job in failed_jobs:
            if job.job_id == action.job_id:
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.status = JobStatus.PENDING
                    job.progress = 0.0
                    job.elapsed_time = 0
                    
                    # Move from failed to pending
                    self.state.failed_jobs.remove(job)
                    self.state.pending_jobs.append(job)
                break
    
    def _action_cancel_job(self, action: Action):
        """Cancel a job to free resources."""
        if action.job_id is None:
            return
        
        # Check running jobs
        for job in self.state.running_jobs if self.state else []:
            if job.job_id == action.job_id:
                job.status = JobStatus.CANCELLED
                job.completed_at = self._current_step
                
                # Free the worker
                for worker in self.workers:
                    if worker.worker_id == job.assigned_worker:
                        worker.current_job = None
                        break
                
                self.state.running_jobs.remove(job)
                self.state.completed_jobs.append(job)
                return
        
        # Check pending jobs
        for job in self.state.pending_jobs if self.state else []:
            if job.job_id == action.job_id:
                job.status = JobStatus.CANCELLED
                job.completed_at = self._current_step
                self.state.pending_jobs.remove(job)
                self.state.completed_jobs.append(job)
                return
    
    def _action_scale_up(self, action: Action):
        """Scale up infrastructure by adding workers."""
        if not self.task_config.enable_scaling:
            return
        
        if len(self.workers) >= self.task_config.max_workers if hasattr(self.task_config, 'max_workers') else 20:
            return
        
        # Add a new worker
        new_worker = Worker(
            worker_id=f"worker_{len(self.workers)}",
            is_active=True,
            cpu_cores=4.0,
            memory_gb=16.0,
            network_bandwidth_mbps=500.0,
            gpu_units=0.0
        )
        self.workers.append(new_worker)
        
        # Add cost
        self._total_cost += self.task_config.cost_per_worker
    
    def _action_scale_down(self, action: Action):
        """Scale down infrastructure by removing idle workers."""
        if not self.task_config.enable_scaling:
            return
        
        # Find idle workers
        idle_workers = [w for w in self.workers if w.is_active and not w.is_busy]
        
        if idle_workers and len(self.workers) > 3:  # Keep at least 3 workers
            worker = idle_workers[-1]
            worker.is_active = False
            
            # Reduce cost (negative cost = savings)
            self._total_cost -= self.task_config.cost_per_worker * 0.5
    
    def _process_running_jobs(self):
        """Process all running jobs, advancing their progress."""
        completed_jobs = []
        
        for job in self.state.running_jobs if self.state else []:
            job.elapsed_time += 1
            
            # Calculate progress based on estimated duration
            progress_increment = 1.0 / job.estimated_duration
            
            # Apply resource allocation bonus if any
            if job.progress > 0:
                progress_increment *= 1.5
            
            job.progress = min(1.0, job.progress + progress_increment)
            
            # Check if job is complete
            if job.progress >= 1.0 or job.elapsed_time >= job.estimated_duration:
                job.status = JobStatus.COMPLETED
                job.completed_at = self._current_step
                
                # Track statistics
                self._total_jobs_completed += 1
                self._total_wait_time += job.wait_time
                
                # Free the worker
                for worker in self.workers:
                    if worker.worker_id == job.assigned_worker:
                        worker.current_job = None
                        break
                
                completed_jobs.append(job)
        
        # Move completed jobs
        for job in completed_jobs:
            if job in self.state.running_jobs:
                self.state.running_jobs.remove(job)
                self.state.completed_jobs.append(job)
    
    def _generate_jobs(self):
        """Generate new jobs based on arrival rate and time of day (peak hours)."""
        # Simulate peak hours (e.g., middle of the simulation)
        progress = self._current_step / max(1, self.task_config.max_steps)
        # Peak load is 2x normal rate, occurs around progress = 0.5 (normal distribution style)
        peak_multiplier = 1.0 + (1.0 * (1.0 - abs(progress - 0.5) * 2))
        
        effective_arrival_rate = self.task_config.job_arrival_rate * peak_multiplier
        
        # We can have multiple jobs arrive in one step if the rate > 1 or just by chance
        while self.rng.random() < effective_arrival_rate:
            job = self._create_random_job()
            self.state.pending_jobs.append(job)
            self._total_jobs_arrived += 1
            effective_arrival_rate -= 1.0
    
    def _create_random_job(self) -> Job:
        """Create a random job based on task configuration."""
        job_type = self.rng.choice(self.task_config.pipeline_types)
        priority = self.rng.choice([JobPriority.LOW, JobPriority.MEDIUM, JobPriority.HIGH])
        
        # Resource requirements based on job type and scarcity
        scarcity = self.task_config.resource_scarcity
        
        if job_type == PipelineType.ML_TRAINING:
            resources = Resources(
                cpu_cores=self.rng.uniform(2, 6),
                memory_gb=self.rng.uniform(8, 24),
                network_bandwidth_mbps=self.rng.uniform(100, 300),
                gpu_units=self.rng.uniform(0.5, 2) if self.rng.random() > 0.5 else 0
            )
            duration = self.rng.randint(15, 30)
        elif job_type == PipelineType.ETL:
            resources = Resources(
                cpu_cores=self.rng.uniform(1, 4),
                memory_gb=self.rng.uniform(4, 16),
                network_bandwidth_mbps=self.rng.uniform(200, 500),
                gpu_units=0
            )
            duration = self.rng.randint(5, 15)
        elif job_type == PipelineType.DATA_VALIDATION:
            resources = Resources(
                cpu_cores=self.rng.uniform(1, 3),
                memory_gb=self.rng.uniform(2, 8),
                network_bandwidth_mbps=self.rng.uniform(100, 200),
                gpu_units=0
            )
            duration = self.rng.randint(3, 8)
        elif job_type == PipelineType.REPORTING:
            resources = Resources(
                cpu_cores=self.rng.uniform(1, 2),
                memory_gb=self.rng.uniform(2, 6),
                network_bandwidth_mbps=self.rng.uniform(50, 150),
                gpu_units=0
            )
            duration = self.rng.randint(2, 5)
        else:  # BACKUP
            resources = Resources(
                cpu_cores=self.rng.uniform(1, 2),
                memory_gb=self.rng.uniform(2, 8),
                network_bandwidth_mbps=self.rng.uniform(300, 800),
                gpu_units=0
            )
            duration = self.rng.randint(5, 10)
        
        # Adjust resources based on scarcity
        if scarcity > 0:
            resources.cpu_cores = max(1, resources.cpu_cores * (1 + scarcity))
            resources.memory_gb = max(2, resources.memory_gb * (1 + scarcity))
        
        # Set deadline based on SLA multiplier
        deadline = int(duration * self.task_config.sla_deadline_multiplier)
        
        job = Job(
            job_id=f"job_{self._job_counter}",
            job_type=job_type,
            priority=priority,
            status=JobStatus.PENDING,
            required_resources=resources,
            estimated_duration=duration,
            deadline=deadline,
            arrived_at=self._current_step,
            data_size_gb=self.rng.uniform(0.5, 10.0)
        )
        
        self._job_counter += 1
        return job
    
    def _check_worker_failures(self):
        """Check for random worker crashes (node failures)."""
        if not hasattr(self.task_config, 'worker_failure_rate') or self.task_config.worker_failure_rate <= 0:
            return
            
        for worker in self.workers:
            # Only consider active workers for crashes
            if worker.is_active and self.rng.random() < self.task_config.worker_failure_rate:
                # Worker crashes!
                worker.is_active = False
                
                # If worker had a job, that job fails
                if worker.current_job:
                    job = worker.current_job
                    job.status = JobStatus.FAILED
                    worker.current_job = None
                    
                    if self.state and job in self.state.running_jobs:
                        self.state.running_jobs.remove(job)
                        self.state.failed_jobs.append(job)
                        self._total_jobs_failed += 1

    def _check_job_failures(self):
        """Check for random job failures."""
        if self.task_config.failure_rate <= 0:
            return
        
        failed_jobs = []
        
        for job in self.state.running_jobs if self.state else []:
            if self.rng.random() < self.task_config.failure_rate:
                job.status = JobStatus.FAILED
                
                # Free the worker
                for worker in self.workers:
                    if worker.worker_id == job.assigned_worker:
                        worker.current_job = None
                        break
                
                failed_jobs.append(job)
        
        # Move failed jobs
        for job in failed_jobs:
            if job in self.state.running_jobs:
                self.state.running_jobs.remove(job)
                self.state.failed_jobs.append(job)
                self._total_jobs_failed += 1
    
    def _check_sla_violations(self):
        """Check for SLA violations (jobs past their deadline)."""
        all_active_jobs = (self.state.pending_jobs + self.state.running_jobs) if self.state else []
        
        for job in all_active_jobs:
            if job.deadline is not None and job.elapsed_time > job.deadline:
                if not hasattr(job, '_sla_violated') or not job._sla_violated:
                    job._sla_violated = True
                    self._sla_violations += 1
                    self.state.sla_violations = self._sla_violations
    
    def _update_state(self):
        """Update the environment state."""
        # Update available resources
        self.state.available_resources = self._get_total_resources()
        
        # Update total resources (from all active workers)
        total_cpu = sum(w.cpu_cores for w in self.workers if w.is_active)
        total_mem = sum(w.memory_gb for w in self.workers if w.is_active)
        total_net = sum(w.network_bandwidth_mbps for w in self.workers if w.is_active)
        total_gpu = sum(w.gpu_units for w in self.workers if w.is_active)
        self.state.total_resources = Resources(
            cpu_cores=total_cpu,
            memory_gb=total_mem,
            network_bandwidth_mbps=total_net,
            gpu_units=total_gpu
        )
        
        # Update queue length
        self.state.queue_length = len(self.state.pending_jobs)
        
        # Update average wait time
        if self._total_jobs_completed > 0:
            self.state.avg_wait_time = self._total_wait_time / self._total_jobs_completed
        
        # Update resource utilization
        self.state.resource_utilization = self.state.available_resources.utilization_percentage(
            self.state.total_resources
        )
        
        # Update cost
        self._total_cost += self.task_config.cost_per_worker * len([w for w in self.workers if w.is_active]) * 0.01
        self.state.cost_so_far = self._total_cost
        
        # Update active workers count
        self.state.active_workers = len([w for w in self.workers if w.is_active])
        
        # Track cumulative utilization for efficiency calculation
        self._cumulative_utilization += self.state.resource_utilization
    
    def _calculate_reward(self, action: Action) -> float:
        """
        Calculate reward for the current step.
        
        The reward function provides partial progress signals:
        - Positive reward for completing jobs
        - Small positive reward for job progress
        - Negative reward for SLA violations
        - Negative reward for resource waste
        - Small negative reward for idle time
        """
        reward = 0.0
        
        # Reward for completed jobs this step
        newly_completed = [j for j in self.state.completed_jobs 
                          if j.completed_at == self._current_step]
        reward += len(newly_completed) * 1.0
        
        # Reward for job progress
        for job in self.state.running_jobs:
            reward += job.progress * 0.1
        
        # Penalty for SLA violations
        if self.state.sla_violations > 0:
            reward -= self.state.sla_violations * 0.5
        
        # Penalty for queue buildup
        reward -= len(self.state.pending_jobs) * 0.05
        
        # Penalty for resource waste (low utilization when jobs are pending)
        if self.state.pending_jobs and self.state.resource_utilization < 50:
            reward -= 0.1
        
        # Small penalty for cost
        reward -= self.task_config.cost_per_worker * 0.01
        
        # Bonus for high-priority job completion
        for job in newly_completed:
            if job.priority == JobPriority.CRITICAL:
                reward += 0.5
            elif job.priority == JobPriority.HIGH:
                reward += 0.3
        
        return reward
    
    def get_score(self) -> float:
        """
        Calculate the final score for the episode.
        
        Returns:
            Score between 0.0 and 1.0.
        """
        metrics = self.get_metrics()
        return metrics.overall_score
    
    def get_metrics(self) -> EvaluationMetrics:
        """
        Calculate detailed evaluation metrics.
        
        Returns:
            EvaluationMetrics object with all metrics.
        """
        total_jobs = self._total_jobs_arrived
        
        # Completion rate
        completion_rate = (self._total_jobs_completed / total_jobs) if total_jobs > 0 else 0.0
        
        # SLA compliance
        total_deadline_jobs = len([j for j in self.state.completed_jobs if j.deadline is not None])
        sla_compliant = total_deadline_jobs - self._sla_violations
        sla_compliance = (sla_compliant / total_deadline_jobs) if total_deadline_jobs > 0 else 1.0
        
        # Resource efficiency (average utilization)
        resource_efficiency = (self._cumulative_utilization / self._current_step) if self._current_step > 0 else 0.0
        resource_efficiency = min(1.0, resource_efficiency / 100.0)  # Normalize to 0-1
        
        # Cost efficiency
        if self.task_config.budget:
            cost_efficiency = max(0.0, 1.0 - (self._total_cost / self.task_config.budget))
        else:
            # Penalize excessive scaling
            cost_efficiency = max(0.0, 1.0 - (self._total_cost / (self._current_step * 0.1))) if self._current_step > 0 else 1.0
        
        # Average wait time (normalized)
        avg_wait_time = self.state.avg_wait_time if self.state else 0.0
        
        # Throughput
        throughput = (self._total_jobs_completed / self._current_step) if self._current_step > 0 else 0.0
        
        # Failure recovery rate
        total_failed = self._total_jobs_failed
        recovered = len([j for j in self.state.completed_jobs if j.retry_count > 0])
        failure_recovery_rate = (recovered / total_failed) if total_failed > 0 else 1.0
        
        return EvaluationMetrics(
            completion_rate=completion_rate,
            resource_efficiency=resource_efficiency,
            sla_compliance=sla_compliance,
            cost_efficiency=cost_efficiency,
            average_wait_time=avg_wait_time,
            throughput=throughput,
            failure_recovery_rate=failure_recovery_rate
        )
    
    @property
    def done(self) -> bool:
        """Check if the episode is done."""
        return self._done
    
    def get_action_space(self) -> list[ActionType]:
        """Get the list of available actions."""
        return list(ActionType)
    
    def get_valid_actions(self) -> list[ActionType]:
        """Get list of valid actions in current state."""
        valid = [ActionType.WAIT]
        
        if self.state:
            if self.state.pending_jobs:
                valid.append(ActionType.SCHEDULE_JOB)
            if self.state.running_jobs:
                valid.append(ActionType.ALLOCATE_RESOURCE)
                valid.append(ActionType.CANCEL_JOB)
            if self.state.failed_jobs:
                valid.append(ActionType.RETRY_FAILED)
            if self.task_config.enable_scaling:
                if len(self.workers) < 20:
                    valid.append(ActionType.SCALE_UP)
                if len([w for w in self.workers if w.is_active and not w.is_busy]) > 2:
                    valid.append(ActionType.SCALE_DOWN)
        
        return valid