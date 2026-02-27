from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


class EnvConfig(BaseModel):
    episode_length: int = Field(description="The length of each episode")
    num_episodes: int = Field(description="The number of episodes to run")


class HyperparametersConfig(BaseModel):
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
    minibatch_size: int | None = Field(
        default=None,
        description="The minibatch size for PPO. Derived from num_episodes if not set.",
    )


class CTeCConfig(BaseModel):
    batch_size: int = Field(description="The batch of the sampled future states")
    contrastive_lr: float
    similarity_function: Literal["l1", "l2"]
    logsumexp_penalty: float
    hidden_dim: int = Field(
        description="The hidden dimension for the state-action and future-state encoders"
    )
    representation_dim: int
    gamma: float = Field(
        description="This gamma parameter used in the discounted state occupancy measure. Used to sample future states using the GEOM(1-gamma) distribution"
    )


class RNDConfig(BaseModel):
    predictor_lr: float = Field(
        description="Learning rate for the RND predictor network"
    )
    hidden_dim: int = Field(
        description="Hidden dimension for the RND target and predictor networks"
    )
    representation_dim: int = Field(
        description="Output embedding dimension shared by target and predictor"
    )
    intrinsic_reward_coeff: float = Field(
        description="An scaling multiplier to ensure the rewards are large enough to overcome the entropy bonus."
    )


class Config(BaseSettings):
    env: EnvConfig
    hyperparameters: HyperparametersConfig
    c_tec: CTeCConfig
    rnd: RNDConfig | None = Field(
        default=None,
        description="RND configuration. Required when --method rnd is used.",
    )

    @model_validator(mode="after")
    def set_minibatch_size(self) -> "Config":
        if self.hyperparameters.minibatch_size is None:
            self.hyperparameters.minibatch_size = (
                self.env.episode_length // 3
            )  # Optimal minibatch size around 1/3 of the episode length with the current config. Can be overridden by the user.
        return self


def get_config(config_path: str | Path) -> Config:

    path_to_config = Path(__file__).parent.parent / "configs" / config_path
    if not path_to_config.exists():
        raise FileNotFoundError("The config file does not exist.")

    with open(path_to_config, "r") as f:
        data = yaml.safe_load(f) or {}
    return Config(**data)
