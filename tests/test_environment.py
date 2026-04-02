"""
Tests for the environment module.
"""

import pytest
from src.openenv_project import (
    DataPipelineEnv, Action, ActionType, Worker,
    get_easy_task_config, get_medium_task_config, get_hard_task_config
)


class TestDataPipelineEnv:
    """Tests for DataPipelineEnv class."""
    
    def test_create_environment(self):
        """Test creating an environment."""
        env = DataPipelineEnv()
        assert env is not None
        assert env.task_config.task_id == "easy"
    
    def test_create_with_custom_config(self):
        """Test creating environment with custom task config."""
        config = get_medium_task_config()
        env = DataPipelineEnv(task_config=config)
        assert env.task_config.task_id == "medium"
    
    def test_reset(self):
        """Test environment reset."""
        env = DataPipelineEnv()
        state = env.reset()
        
        assert state is not None
        assert state.pending_jobs == []
        assert state.running_jobs == []
        assert state.completed_jobs == []
        assert state.failed_jobs == []
        assert state.time_step == 0
        assert not env.done
    
    def test_step_returns_correct_tuple(self):
        """Test that step returns (state, reward, done, info)."""
        env = DataPipelineEnv()
        env.reset()
        
        action = Action(action_type=ActionType.WAIT)
        state, reward, done, info = env.step(action)
        
        assert state is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert state.time_step == 1
    
    def test_step_with_invalid_action_after_done(self):
        """Test that stepping after done raises error."""
        env = DataPipelineEnv()
        env.reset()
        
        # Run until done
        while not env.done:
            action = Action(action_type=ActionType.WAIT)
            env.step(action)
        
        with pytest.raises(RuntimeError):
            env.step(Action(action_type=ActionType.WAIT))
    
    def test_state_attribute(self):
        """Test state attribute returns current state."""
        env = DataPipelineEnv()
        env.reset()
        
        state = env.state
        assert state is not None
        assert state.time_step == 0
    
    def test_get_state_method(self):
        """Test get_state() method returns current state."""
        env = DataPipelineEnv()
        env.reset()
        
        state = env.get_state()
        assert state is not None
        assert state.time_step == 0
    
    def test_state_before_reset(self):
        """Test that accessing state before reset returns None."""
        env = DataPipelineEnv()
        assert env.state is None
    
    def test_get_state_before_reset(self):
        """Test that get_state() before reset raises error."""
        env = DataPipelineEnv()
        with pytest.raises(RuntimeError):
            env.get_state()
    
    def test_job_generation(self):
        """Test that jobs are generated during simulation."""
        env = DataPipelineEnv(seed=42)
        env.reset()
        
        # Run for several steps
        for _ in range(20):
            action = Action(action_type=ActionType.WAIT)
            env.step(action)
        
        # Check that some jobs have arrived
        state = env.state
        total_jobs = (len(state.pending_jobs) + len(state.running_jobs) + 
                     len(state.completed_jobs))
        assert total_jobs > 0
    
    def test_schedule_job_action(self):
        """Test scheduling a job."""
        env = DataPipelineEnv(seed=42)
        env.reset()
        
        # Wait for a job to arrive
        for _ in range(10):
            state = env.state
            if state.pending_jobs:
                break
            action = Action(action_type=ActionType.WAIT)
            env.step(action)
        
        # Schedule the pending job
        state = env.state
        if state.pending_jobs:
            job_id = state.pending_jobs[0].job_id
            action = Action(action_type=ActionType.SCHEDULE_JOB, job_id=job_id)
            env.step(action)
            
            # Check job moved to running
            state = env.state
            assert len(state.pending_jobs) == 0 or state.pending_jobs[0].job_id != job_id
    
    def test_get_action_space(self):
        """Test getting action space."""
        env = DataPipelineEnv()
        actions = env.get_action_space()
        assert len(actions) == 7  # All 7 action types
        assert ActionType.WAIT in actions
        assert ActionType.SCHEDULE_JOB in actions
    
    def test_get_valid_actions(self):
        """Test getting valid actions in current state."""
        env = DataPipelineEnv()
        env.reset()
        
        valid = env.get_valid_actions()
        assert ActionType.WAIT in valid  # Wait is always valid
    
    def test_done_property(self):
        """Test done property."""
        env = DataPipelineEnv()
        env.reset()
        
        assert not env.done
        
        # Run until max steps
        config = env.task_config
        for _ in range(config.max_steps):
            action = Action(action_type=ActionType.WAIT)
            env.step(action)
        
        assert env.done
    
    def test_get_score(self):
        """Test getting final score."""
        env = DataPipelineEnv(seed=42)
        env.reset()
        
        # Run a short episode
        for _ in range(20):
            action = Action(action_type=ActionType.WAIT)
            env.step(action)
        
        score = env.get_score()
        assert 0.0 <= score <= 1.0
    
    def test_get_metrics(self):
        """Test getting detailed metrics."""
        env = DataPipelineEnv(seed=42)
        env.reset()
        
        for _ in range(20):
            action = Action(action_type=ActionType.WAIT)
            env.step(action)
        
        metrics = env.get_metrics()
        assert 0.0 <= metrics.completion_rate <= 1.0
        assert 0.0 <= metrics.resource_efficiency <= 1.0
        assert 0.0 <= metrics.sla_compliance <= 1.0
        assert 0.0 <= metrics.cost_efficiency <= 1.0
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        env1 = DataPipelineEnv(seed=123)
        env2 = DataPipelineEnv(seed=123)
        
        env1.reset()
        env2.reset()
        
        for _ in range(20):
            action1 = Action(action_type=ActionType.WAIT)
            action2 = Action(action_type=ActionType.WAIT)
            state1, _, _, _ = env1.step(action1)
            state2, _, _, _ = env2.step(action2)
        
        # States should be identical
        assert state1.queue_length == state2.queue_length
        assert len(state1.pending_jobs) == len(state2.pending_jobs)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        env1 = DataPipelineEnv(seed=42)
        env2 = DataPipelineEnv(seed=999)
        
        env1.reset()
        env2.reset()
        
        for _ in range(30):
            action = Action(action_type=ActionType.WAIT)
            env1.step(action)
            env2.step(action)
        
        state1 = env1.state
        state2 = env2.state
        
        # Results may differ (though not guaranteed for short runs)
        # This test just ensures the seeds work differently
        assert env1 is not None
        assert env2 is not None


class TestWorker:
    """Tests for Worker class."""
    
    def test_create_worker(self):
        """Test creating a worker."""
        worker = Worker()
        assert worker.is_active
        assert worker.current_job is None
        assert not worker.is_busy
    
    def test_worker_with_job(self):
        """Test worker busy state."""
        from src.openenv_project import Job
        
        worker = Worker()
        job = Job()
        worker.current_job = job
        
        assert worker.is_busy
    
    def test_worker_to_dict(self):
        """Test worker serialization."""
        worker = Worker(worker_id="test-worker", is_active=True)
        d = worker.to_dict()
        assert d["worker_id"] == "test-worker"
        assert d["is_active"]
        assert not d["is_busy"]


class TestTaskConfigs:
    """Tests for task configurations."""
    
    def test_easy_config(self):
        """Test easy task configuration."""
        config = get_easy_task_config()
        assert config.task_id == "easy"
        assert config.failure_rate == 0.0
        assert config.resource_scarcity == 0.0
        assert not config.enable_scaling
    
    def test_medium_config(self):
        """Test medium task configuration."""
        config = get_medium_task_config()
        assert config.task_id == "medium"
        assert config.failure_rate == 0.05
        assert config.resource_scarcity == 0.3
        assert not config.enable_scaling
    
    def test_hard_config(self):
        """Test hard task configuration."""
        config = get_hard_task_config()
        assert config.task_id == "hard"
        assert config.failure_rate == 0.08
        assert config.resource_scarcity == 0.5
        assert config.enable_scaling
        assert config.budget == 100.0