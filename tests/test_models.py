"""
Tests for the models module.
"""

import pytest
from src.openenv_project.models import (
    Resources, Job, JobStatus, JobPriority, PipelineType,
    ActionType, Action, EnvironmentState, TaskConfig, EvaluationMetrics
)


class TestResources:
    """Tests for Resources dataclass."""
    
    def test_create_resources(self):
        """Test creating a Resources instance."""
        r = Resources(cpu_cores=4, memory_gb=16, network_bandwidth_mbps=500, gpu_units=0)
        assert r.cpu_cores == 4
        assert r.memory_gb == 16
        assert r.network_bandwidth_mbps == 500
        assert r.gpu_units == 0
    
    def test_default_resources(self):
        """Test default resource values."""
        r = Resources()
        assert r.cpu_cores == 8
        assert r.memory_gb == 32
        assert r.network_bandwidth_mbps == 1000
        assert r.gpu_units == 0
    
    def test_negative_resources_raises(self):
        """Test that negative resources raise ValueError."""
        with pytest.raises(ValueError):
            Resources(cpu_cores=-1)
    
    def test_can_allocate(self):
        """Test resource allocation checking."""
        available = Resources(cpu_cores=8, memory_gb=32, network_bandwidth_mbps=1000, gpu_units=2)
        required = Resources(cpu_cores=4, memory_gb=16, network_bandwidth_mbps=500, gpu_units=1)
        assert available.can_allocate(required)
    
    def test_cannot_allocate(self):
        """Test insufficient resources."""
        available = Resources(cpu_cores=4, memory_gb=8, network_bandwidth_mbps=500, gpu_units=0)
        required = Resources(cpu_cores=8, memory_gb=16, network_bandwidth_mbps=1000, gpu_units=1)
        assert not available.can_allocate(required)
    
    def test_allocate(self):
        """Test allocating resources."""
        available = Resources(cpu_cores=8, memory_gb=32, network_bandwidth_mbps=1000, gpu_units=2)
        required = Resources(cpu_cores=4, memory_gb=16, network_bandwidth_mbps=500, gpu_units=1)
        remaining = available.allocate(required)
        assert remaining.cpu_cores == 4
        assert remaining.memory_gb == 16
        assert remaining.network_bandwidth_mbps == 500
        assert remaining.gpu_units == 1
    
    def test_deallocate(self):
        """Test deallocating resources."""
        current = Resources(cpu_cores=4, memory_gb=16, network_bandwidth_mbps=500, gpu_units=1)
        released = Resources(cpu_cores=4, memory_gb=16, network_bandwidth_mbps=500, gpu_units=1)
        total = current.deallocate(released)
        assert total.cpu_cores == 8
        assert total.memory_gb == 32
        assert total.network_bandwidth_mbps == 1000
        assert total.gpu_units == 2
    
    def test_utilization_percentage(self):
        """Test utilization calculation."""
        available = Resources(cpu_cores=4, memory_gb=16, network_bandwidth_mbps=500, gpu_units=0)
        total = Resources(cpu_cores=8, memory_gb=32, network_bandwidth_mbps=1000, gpu_units=0)
        util = available.utilization_percentage(total)
        assert util == 50.0  # 50% utilization
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        r = Resources(cpu_cores=4, memory_gb=16, network_bandwidth_mbps=500, gpu_units=1)
        d = r.to_dict()
        assert d["cpu_cores"] == 4
        assert d["memory_gb"] == 16
        assert d["network_bandwidth_mbps"] == 500
        assert d["gpu_units"] == 1


class TestJob:
    """Tests for Job dataclass."""
    
    def test_create_job(self):
        """Test creating a Job instance."""
        job = Job()
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.MEDIUM
        assert job.job_type == PipelineType.ETL
        assert job.progress == 0.0
        assert job.retry_count == 0
    
    def test_job_id_is_unique(self):
        """Test that each job gets a unique ID."""
        job1 = Job()
        job2 = Job()
        assert job1.job_id != job2.job_id
    
    def test_is_overdue(self):
        """Test overdue detection."""
        job = Job(deadline=10, elapsed_time=5)
        assert not job.is_overdue
        
        job.elapsed_time = 15
        assert job.is_overdue
    
    def test_time_remaining(self):
        """Test time remaining calculation."""
        job = Job(deadline=10, elapsed_time=5)
        assert job.time_remaining == 5
        
        job.elapsed_time = 15
        assert job.time_remaining == 0
    
    def test_wait_time_pending(self):
        """Test wait time for pending job."""
        job = Job(arrived_at=0, elapsed_time=5, started_at=None)
        assert job.wait_time == 5
    
    def test_wait_time_started(self):
        """Test wait time for started job."""
        job = Job(arrived_at=0, started_at=3)
        assert job.wait_time == 3
    
    def test_to_dict(self):
        """Test job serialization."""
        job = Job(job_id="test-123", priority=JobPriority.HIGH)
        d = job.to_dict()
        assert d["job_id"] == "test-123"
        assert d["priority"] == 3
        assert d["status"] == "pending"


class TestAction:
    """Tests for Action dataclass."""
    
    def test_create_action(self):
        """Test creating an Action."""
        action = Action(action_type=ActionType.WAIT)
        assert action.action_type == ActionType.WAIT
        assert action.job_id is None
        assert action.metadata == {}
    
    def test_action_with_job_id(self):
        """Test action with job ID."""
        action = Action(action_type=ActionType.SCHEDULE_JOB, job_id="job-123")
        assert action.job_id == "job-123"
    
    def test_to_dict(self):
        """Test action serialization."""
        action = Action(action_type=ActionType.SCHEDULE_JOB, job_id="job-123")
        d = action.to_dict()
        assert d["action_type"] == "schedule_job"
        assert d["job_id"] == "job-123"


class TestEnvironmentState:
    """Tests for EnvironmentState dataclass."""
    
    def test_create_state(self):
        """Test creating an EnvironmentState."""
        state = EnvironmentState()
        assert state.pending_jobs == []
        assert state.running_jobs == []
        assert state.completed_jobs == []
        assert state.failed_jobs == []
        assert state.queue_length == 0
        assert state.time_step == 0
    
    def test_to_dict(self):
        """Test state serialization."""
        state = EnvironmentState(time_step=5, queue_length=3)
        d = state.to_dict()
        assert d["time_step"] == 5
        assert d["queue_length"] == 3


class TestTaskConfig:
    """Tests for TaskConfig dataclass."""
    
    def test_create_task_config(self):
        """Test creating a TaskConfig."""
        config = TaskConfig(
            task_id="test",
            name="Test Task",
            description="A test task",
            max_steps=100,
            success_threshold=0.7
        )
        assert config.task_id == "test"
        assert config.max_steps == 100
        assert config.success_threshold == 0.7
        assert config.job_arrival_rate == 0.5
        assert config.failure_rate == 0.0
    
    def test_to_dict(self):
        """Test config serialization."""
        config = TaskConfig(
            task_id="test",
            name="Test Task",
            description="A test task",
            max_steps=100,
            success_threshold=0.7
        )
        d = config.to_dict()
        assert d["task_id"] == "test"
        assert d["max_steps"] == 100


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""
    
    def test_create_metrics(self):
        """Test creating EvaluationMetrics."""
        metrics = EvaluationMetrics(
            completion_rate=0.8,
            resource_efficiency=0.6,
            sla_compliance=0.9,
            cost_efficiency=0.7
        )
        assert metrics.completion_rate == 0.8
        assert metrics.overall_score == pytest.approx(0.775, rel=0.01)
    
    def test_overall_score_weights(self):
        """Test that overall score uses correct weights."""
        metrics = EvaluationMetrics(
            completion_rate=1.0,
            resource_efficiency=0.0,
            sla_compliance=0.0,
            cost_efficiency=0.0
        )
        assert metrics.overall_score == pytest.approx(0.4, rel=0.01)
    
    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = EvaluationMetrics(completion_rate=0.8)
        d = metrics.to_dict()
        assert d["completion_rate"] == 0.8
        assert "overall_score" in d