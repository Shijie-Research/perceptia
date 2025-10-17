# Copyright (c) Perceptia Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from transformers import TrainingArguments as HFTraningArguments
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import FSDPOption


__all__ = [
    "TrainingArguments",
    "ModelArguments",
    "DatasetArguments",
    "hydra_init",
]


@dataclass
class TrainingArgumentsOverride:
    # override some training arguments to support omegaconf
    lr_scheduler_kwargs: Optional[dict[str, Any]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts."
            )
        },
    )
    debug: Optional[DebugOption] = field(
        default=None,
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )
    fsdp: Optional[FSDPOption] = field(
        default=None,
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )
    fsdp_config: Optional[dict[str, Any]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Config to be used with FSDP (Pytorch Fully Sharded Data Parallel). The value is either a "
                "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    accelerator_config: Optional[dict[str, Any]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Config to be used with the internal Accelerator object initialization. The value is either a "
                "accelerator json config file (e.g., `accelerator_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    deepspeed: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    report_to: Optional[list[str]] = field(
        default_factory=list, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    gradient_checkpointing_kwargs: Optional[dict[str, Any]] = field(
        default_factory=dict,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
    ray_scope: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "This argument is deprecated and will be removed in v5.2. Set env var RAY_SCOPE instead."
                'The scope to use when doing hyperparameter search with Ray. By default, `"last"` will be used. Ray'
                " will then use the last checkpoint of all trials, compare those, and select the best one. However,"
                " other options are also available. See the Ray documentation"
                " (https://docs.ray.io/en/latest/tune/api_docs/analysis.html"
                "#ray.tune.ExperimentAnalysis.get_best_trial)"
                " for more options."
            )
        },
    )
    optim_target_modules: Optional[list[str]] = field(
        default_factory=list,
        metadata={
            "help": "Target modules for the optimizer defined in the `optim` argument. Only used for the GaLore optimizer at the moment."
        },
    )


@dataclass
class TrainingArguments(TrainingArgumentsOverride, HFTraningArguments):
    pass


@dataclass
class ModelArguments:
    pass


@dataclass
class DatasetArguments:
    pass


def hydra_init() -> None:
    """Register additional Perceptia configuration to Hydra."""
    cs = ConfigStore.instance()

    cs.store(group="training", name="default", node=TrainingArguments)
    cs.store(group="model", name="default", node=ModelArguments)
    cs.store(group="dataset", name="default", node=DatasetArguments)
