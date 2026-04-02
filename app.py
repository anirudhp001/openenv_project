import os
import sys
import threading
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import gradio as gr

sys.path.insert(0, '.')

from src.openenv_project import (
    DataPipelineEnv, Action, ActionType, get_task_config, EnvironmentState
)

# OpenEnv API models
class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action_type: str
    job_id: Optional[str] = None
    resource_amount: Optional[Dict[str, float]] = None

class ActionResponse(BaseModel):
    state: dict
    reward: float
    done: bool
    info: dict

# Thread-safe environment manager
class EnvManager:
    def __init__(self):
        self.envs = {}
        self.lock = threading.Lock()

    def reset(self, session_id: str, task_id: str, seed: int = None):
        with self.lock:
            config = get_task_config(task_id)
            env = DataPipelineEnv(task_config=config, seed=seed)
            state = env.reset()
            self.envs[session_id] = env
            return state

    def step(self, session_id: str, action_req: StepRequest):
        with self.lock:
            if session_id not in self.envs:
                raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
            env = self.envs[session_id]
            try:
                action_type = ActionType(action_req.action_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid action_type: {action_req.action_type}")
            
            action = Action(
                action_type=action_type,
                job_id=action_req.job_id,
                resource_amount=action_req.resource_amount
            )
            state, reward, done, info = env.step(action)
            
            # Clean up on done
            if done:
                del self.envs[session_id]
                
            return {
                "state": state.model_dump(),
                "reward": reward,
                "done": done,
                "info": info
            }

    def get_state(self, session_id: str):
        with self.lock:
            if session_id not in self.envs:
                raise HTTPException(status_code=404, detail="Session not found.")
            return self.envs[session_id].state.model_dump()


app = FastAPI(title="DataPipelineEnv API", version="1.0.0")
env_manager = EnvManager()

# We will use a default session for simple compatibility if no session_id is provided
DEFAULT_SESSION = "default"

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/reset", tags=["OpenEnv"], status_code=status.HTTP_200_OK)
@app.post("/api/v1/reset", tags=["OpenEnv"], status_code=status.HTTP_200_OK)
def reset_env(req: ResetRequest):
    try:
        state = env_manager.reset(DEFAULT_SESSION, req.task_id, req.seed)
        return state.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=ActionResponse, tags=["OpenEnv"])
@app.post("/api/v1/step", response_model=ActionResponse, tags=["OpenEnv"])
def step_env(req: StepRequest):
    return env_manager.step(DEFAULT_SESSION, req)

@app.get("/state", tags=["OpenEnv"])
@app.get("/api/v1/state", tags=["OpenEnv"])
def get_env_state():
    return env_manager.get_state(DEFAULT_SESSION)

# Simple Gradio interface
def make_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# DataPipelineEnv Demo\nThis app provides an OpenEnv API.")
    return demo

app = gr.mount_gradio_app(app, make_demo(), path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)