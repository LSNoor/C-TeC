from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings


class HyperparametersConfig(BaseModel):
    num_timesteps: int = Field(description="The total number of timesteps")
    num_steps: int = Field(
        description="The number of environment steps collected per rollout before each policy update"
    )
    policy_lr: float
    critic_lr: float
    update_epoch: int = Field(
        description="The number of epochs used to update the actor and critic networks"
    )
    discount_factor: float = Field(
        description="The discount factor gamma for future reward"
    )
    gae_lambda: float = Field(
        description="The lambda parameter for generalized advantage estimation"
    )
    clip_epsilon: float = Field(description="The epsilon parameter for clipping in PPO")
    entropy_coef: float = Field(description="The entropy coefficient for PPO")
    value_coef: float = Field(description="The value loss coefficient for PPO")
    max_grad_norm: float = Field(description="The max norm of gradient for PPO")
    hidden_dim: int = Field(
        description="The hidden dimension of the actor and critic networks"
    )
    minibatch_size: int = Field(description="The minibatch size for PPO")


class CTeCConfig(BaseModel):
    batch_size: int = Field(description="The batch of the sampled future states")
    contrastive_lr: float
    similarity_function: Literal["l1", "l2"]
    logsumexp_penalty: float
    hidden_dim: int = Field(
        description="The hidden dimension for the state-action and future-state encoders"
    )
    representation_dim: int


class Config(BaseSettings):
    hyperparameters: HyperparametersConfig
    c_tec: CTeCConfig


def get_config(config_path: str) -> Config:

    path_to_config = Path(__file__).parent / config_path
    if not path_to_config.exists():
        raise FileNotFoundError("The config file does not exist.")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}
    return Config(**data)
