from typing import Dict
from pydantic import BaseModel, validator

from q_networks.configs.enums import EnvType, EpsDecayMethod
from q_networks.utils.buffers import BaseBufferOptions, BufferType, RandomBufferOptions, PriorityBufferOptions


class EnvConfig(BaseModel):
    type: EnvType
    gamma: float
    history: int

BUFFER_OPTION_MAP: Dict[BufferType, BaseBufferOptions] = {
    BufferType.random: RandomBufferOptions,
    BufferType.priority: PriorityBufferOptions,
}

class BufferConfig(BaseModel):
    method: BufferType
    options: Dict[str, dict]
    
    @validator("options", always=True)
    def validate_date(cls, value, values) -> BaseBufferOptions:
        buffer_type: BufferType = values["method"]
        return BUFFER_OPTION_MAP[buffer_type](**value["base"], **value[buffer_type.name])

class ExplorationConfig(BaseModel):
    start: float
    end: float
    steps: int
    schedule: EpsDecayMethod
    
    
class TrainingConfig(BaseModel):
    max_steps: int
    start_step: int
    
    batch_size: int
    max_grad_norm: float
    update_freq: int
    grad_steps: int
    target_update_steps: int
    
    lr: float # Learning rate
    
class NetworkConfig(BaseModel):
    ddqn: bool


class LogConfig(BaseModel):
    path: str
    run: str
    

class EvaluationConfig(BaseModel):
    frequency: int
    noop_max: int
    n_episodes: int
    ep_max_frames: int
    eps: float


class DQNParams(BaseModel):
    env: EnvConfig
    eps: ExplorationConfig
    train: TrainingConfig
    buffer: BufferConfig
    net: NetworkConfig
    log: LogConfig
    eval: EvaluationConfig
