"""
Tests for the graders module.
"""

import pytest
from src.openenv_project import (
    DataPipelineEnv, Action, ActionType,
    get_grader, grade_episode, grade_multiple_episodes,
    EasyGrader, MediumGrader, HardGrader,
    get_easy_task_config, get_medium_task_config, get_hard_task_config
)


class TestGraders:
    """Tests for grading functionality."""
    
    def test_get_easy_grader(self):
        """Test getting easy grader."""
        grader = get_grader("easy")
        assert isinstance(grader, EasyGrader)
        assert grader.task_id == "easy"
        assert grader.success_threshold == 0.7
    
    def test_get_medium_grader(self):
        """Test getting medium grader."""
        grader = get_grader("medium")
        assert isinstance(grader, MediumGrader)
        assert grader.task_id == "medium"
        assert grader.success_threshold == 0.6
    
    def test_get_hard_grader(self):
        """Test getting hard grader."""
        grader = get_grader("hard")
        assert isinstance(grader, HardGrader)
        assert grader.task_id == "hard"
        assert grader.success_threshold == 0.5
    
    def test_get_unknown_grader_raises(self):
        """Test that unknown task raises ValueError."""
        with pytest.raises(ValueError):
            get_grader("unknown")
    
    def test_grade_episode_easy(self):
        """Test grading an easy episode."""
        config = get_easy_task_config()
        env = DataPipelineEnv(task_config=config, seed=42)
        env.reset()
        
        # Run episode
        for _ in range(config.max_steps):
            action = Action(action_type=ActionType.WAIT)
            env.step(action)
        
        result = grade_episode(env, "easy")
        
        assert result.task_id == "easy"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert result.metrics is not None
        assert len(result.feedback) > 0
    
    def test_grade_episode_medium(self):
        """Test grading a medium episode."""
        config = get_medium_task_config()
        env = DataPipelineEnv(task_config=config, seed=42)
        env.reset()
        
        for _ in range(config.max_steps):
            action = Action(action_type=ActionType.WAIT)
            env.step(action)
        
        result = grade_episode(env, "medium")
        
        assert result.task_id == "medium"
        assert 0.0 <= result.score <= 1.0
    
    def test_grade_episode_hard(self):
        """Test grading a hard episode."""
        config = get_hard_task_config()
        env = DataPipelineEnv(task_config=config, seed=42)
        env.reset()
        
        for _ in range(config.max_steps):
            action = Action(action_type=ActionType.WAIT)
            env.step(action)
        
        result = grade_episode(env, "hard")
        
        assert result.task_id == "hard"
        assert 0.0 <= result.score <= 1.0
    
    def test_grade_multiple_episodes(self):
        """Test grading multiple episodes."""
        results = []
        
        for task_id in ["easy", "medium", "hard"]:
            config = get_easy_task_config() if task_id == "easy" else (
                get_medium_task_config() if task_id == "medium" else 
                get_hard_task_config()
            )
            env = DataPipelineEnv(task_config=config, seed=42)
            env.reset()
            
            for _ in range(min(20, config.max_steps)):
                action = Action(action_type=ActionType.WAIT)
                env.step(action)
            
            grade = grade_episode(env, task_id)
            results.append((task_id, grade))
        
        aggregates = grade_multiple_episodes(results)
        
        assert "overall" in aggregates
        assert "easy" in aggregates
        assert aggregates["overall"]["total_episodes"] == 3


class TestGradeResult:
    """Tests for GradeResult dataclass."""
    
    def test_grade_result_structure(self):
        """Test GradeResult has all required fields."""
        from src.openenv_project.graders import GradeResult
        from src.openenv_project.models import EvaluationMetrics
        
        metrics = EvaluationMetrics()
        result = GradeResult(
            score=0.75,
            passed=True,
            metrics=metrics,
            feedback="Good job!",
            task_id="easy"
        )
        
        assert result.score == 0.75
        assert result.passed is True
        assert result.feedback == "Good job!"
        assert result.task_id == "easy"