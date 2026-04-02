---
title: DataPipelineEnv
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
hardware: cpu-basic
---
# DataPipelineEnv - OpenEnv Environment for Data Pipeline Management

## Overview

**DataPipelineEnv** is a realistic simulation environment where an AI agent learns to manage data processing pipelines. The agent must schedule tasks, allocate resources, handle failures, and optimize throughput in a simulated data engineering environment.

This is a **real-world task** that mirrors what data engineering teams do daily: managing compute resources, meeting SLAs, handling failures, and balancing cost vs. performance.

## Why This Environment?

Data pipeline management is a critical real-world task that involves:
- **Resource allocation**: Deciding which jobs run on limited compute resources
- **Priority management**: Handling critical vs. low-priority work
- **Failure recovery**: Dealing with job failures and retries
- **Cost optimization**: Balancing performance with infrastructure costs
- **SLA compliance**: Meeting deadlines for time-sensitive work

## Environment Description

### Scenario
You are an AI operations manager responsible for running data pipelines in a cloud environment. Your goals are to:
- Process incoming data jobs efficiently
- Manage limited compute resources (CPU, memory, network bandwidth)
- Handle task failures and retries
- Minimize costs while meeting SLA deadlines
- Balance competing priorities across multiple pipelines

### Action Space

| Action | Description | When to Use |
|--------|-------------|-------------|
| `schedule_job` | Schedule a pending job onto available workers | When jobs are waiting and workers are free |
| `allocate_resource` | Allocate additional resources to a running job | For urgent jobs near their deadline |
| `retry_failed` | Retry a failed job | When jobs have failed and can be retried |
| `cancel_job` | Cancel a job to free resources | For low-priority jobs when resources are needed |
| `scale_up` | Add more workers (costs money) | When queue is building and utilization is high |
| `scale_down` | Remove idle workers (saves money) | When resources are underutilized |
| `wait` | Wait for current jobs to complete | When all workers are busy or no pending jobs |

### Observation Space

The environment provides the following observations at each timestep:

| Field | Type | Description |
|-------|------|-------------|
| `pending_jobs` | List[Job] | Jobs waiting to be processed |
| `running_jobs` | List[Job] | Currently executing jobs |
| `completed_jobs` | List[Job] | Successfully completed jobs |
| `failed_jobs` | List[Job] | Failed jobs awaiting retry |
| `available_resources` | Resources | Current available compute resources |
| `total_resources` | Resources | Total compute resource capacity |
| `queue_length` | int | Number of jobs in queue |
| `avg_wait_time` | float | Average job wait time |
| `resource_utilization` | float | Current resource utilization percentage |
| `cost_so_far` | float | Accumulated operational cost |
| `sla_violations` | int | Number of SLA breaches |
| `time_step` | int | Current simulation timestep |
| `active_workers` | int | Number of active compute workers |

## Tasks

### Task 1: Basic Job Scheduling (Easy)
**Objective**: Schedule incoming jobs to complete them successfully.

- Simple job arrival pattern (30% probability per step)
- Abundant resources (no scarcity)
- No job failures
- Single pipeline type (ETL)
- Generous SLA deadlines (3x estimated duration)

**Success Criteria**: Score ≥ 0.7 (70% completion rate with good efficiency)

### Task 2: Resource-Constrained Scheduling (Medium)
**Objective**: Manage limited resources while handling job failures.

- Higher job arrival rate (40% probability per step)
- Limited CPU/memory (30% resource scarcity)
- Random job failures (5% failure rate)
- Multiple pipeline types (ETL, Validation, Reporting)
- Tighter SLA deadlines (2x estimated duration)
- Priority-based scheduling required

**Success Criteria**: Score ≥ 0.6 (balanced performance across metrics)

### Task 3: Multi-Pipeline Optimization (Hard)
**Objective**: Optimize multiple pipelines with cost constraints and dynamic scaling.

- High job arrival rate (50% probability per step)
- Significant resource scarcity (50%)
- Higher failure rate (8%)
- All pipeline types including ML training with GPU requirements
- Very tight SLA deadlines (1.5x estimated duration)
- Budget constraint ($100)
- Auto-scaling required to handle peak loads

**Success Criteria**: Score ≥ 0.5 (complex optimization across all dimensions)

## Installation

### Prerequisites
- Python 3.9+
- pip
- Docker (for containerized deployment)
- OpenAI API key (for LLM-based baseline)

### Local Installation

```bash
# Clone or download the repository
cd openenv_project

# Install dependencies
pip install -r requirements.txt

# Install the package (optional)
pip install -e .
```

### Quick Start

```python
from src.openenv_project import DataPipelineEnv, Action, ActionType, get_task_config

# Create environment with easy task
config = get_task_config("easy")
env = DataPipelineEnv(task_config=config, seed=42)

# Reset and run
state = env.reset()
while not env.done:
    # Simple heuristic: always schedule if jobs pending
    if state.pending_jobs:
        action = Action(action_type=ActionType.SCHEDULE_JOB)
    else:
        action = Action(action_type=ActionType.WAIT)
    
    state, reward, done, info = env.step(action)

# Get final score
score = env.get_score()
print(f"Final Score: {score:.3f}")
```

## Usage

### Running the Baseline (Heuristic Agent)

```bash
# Run all tasks
python baseline_inference.py --task all --episodes 10 --seed 42

# Run specific task
python baseline_inference.py --task easy --episodes 5
```

### Running with OpenAI API (LLM Agent)

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Run with GPT-4
python openai_baseline.py --task all --model gpt-4 --episodes 3 --verbose

# Run with specific task
python openai_baseline.py --task hard --model gpt-4-turbo --seed 42
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/openenv_project --cov-report=html
```

### Docker Deployment

```bash
# Build the image
docker build -t datapipeline-env .

# Run the container
docker run --rm datapipeline-env python baseline_inference.py --task all --episodes 3

# Run with interactive shell
docker run -it --rm datapipeline-env /bin/bash
```

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face (select Docker runtime)
2. Push the repository files to the Space
3. The Gradio interface will be available at `https://huggingface.co/spaces/<username>/datapipeline-env`

Alternatively, use the included `app.py`:
```bash
pip install gradio
python app.py
```

## Baseline Scores

Results from the heuristic baseline agent (10 episodes, seed 42):

| Task | Mean Score | Min | Max | Pass Rate |
|------|------------|-----|-----|-----------|
| Easy | 0.861 | 0.859 | 0.864 | 100% |
| Medium | 0.574 | 0.478 | 0.669 | 50% |
| Hard | 0.728 | 0.688 | 0.768 | 100% |

**Overall Mean Score**: 0.721

## Evaluation Metrics

The environment evaluates agents using a weighted combination of metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| Completion Rate | 40% | Percentage of jobs successfully completed |
| SLA Compliance | 25% | Percentage of jobs completed within deadline |
| Resource Efficiency | 20% | How well compute resources are utilized |
| Cost Efficiency | 15% | Performance relative to budget |

### Scoring Formula
```
score = completion_rate * 0.4 + sla_compliance * 0.25 + 
        resource_efficiency * 0.2 + cost_efficiency * 0.15
```

## Project Structure

```
openenv_project/
├── src/openenv_project/
│   ├── __init__.py           # Package exports
│   ├── models.py             # Pydantic typed models
│   ├── environment.py        # Main environment (step/reset/state API)
│   ├── tasks.py              # Task configurations
│   ├── graders.py            # Agent graders
│   └── reward.py             # Reward function utilities
├── tests/
│   ├── test_models.py        # Model tests
│   ├── test_environment.py   # Environment tests
│   └── test_graders.py       # Grader tests
├── baseline_inference.py     # Heuristic baseline agent
├── openai_baseline.py        # OpenAI API baseline agent
├── app.py                    # Hugging Face Spaces demo
├── openenv.yaml              # OpenEnv specification
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## OpenEnv Specification

The `openenv.yaml` file defines the complete OpenEnv specification:

```yaml
spec_version: "1.0"
environment:
  name: "DataPipelineEnv"
  version: "1.0.0"
  description: "Data pipeline management simulation"
  
tasks:
  - id: "easy"
    max_steps: 100
    success_threshold: 0.7
  - id: "medium"
    max_steps: 150
    success_threshold: 0.6
  - id: "hard"
    max_steps: 200
    success_threshold: 0.5
```

## API Reference

### DataPipelineEnv

```python
class DataPipelineEnv:
    def __init__(self, task_config: TaskConfig, seed: Optional[int] = None)
    def reset(self) -> EnvironmentState
    def step(self, action: Action) -> Tuple[EnvironmentState, float, bool, dict]
    def get_state(self) -> EnvironmentState  # Also accessible via .state attribute
    def get_score(self) -> float
    def get_metrics(self) -> EvaluationMetrics
    def get_action_space(self) -> List[ActionType]
    def get_valid_actions(self) -> List[ActionType]
    @property
    def done(self) -> bool
```

### Key Models (Pydantic)

- `Resources`: CPU, memory, network, GPU resources
- `Job`: Data processing job with priority, status, deadlines
- `Action`: Agent action with type and parameters
- `EnvironmentState`: Complete environment snapshot
- `TaskConfig`: Task configuration
- `EvaluationMetrics`: Performance metrics

## License

MIT License

## Contributing

Contributions are welcome! Please ensure tests pass and add new tests for new functionality.

```bash
# Run tests before submitting
pytest tests/ -v