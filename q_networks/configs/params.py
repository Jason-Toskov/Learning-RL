from pydantic import BaseModel

from q_networks.configs.enums import BufferType, EnvType, EpsDecayMethod


class EnvConfig(BaseModel):
    type: EnvType
    gamma: float
    history: int
    
    
class BufferConfig(BaseModel):
    size: int
    method: BufferType


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
