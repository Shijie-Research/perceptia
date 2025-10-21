# Copyright (c) Perceptia Contributors. All rights reserved.
import copy
import logging
from collections import defaultdict
from collections.abc import Iterable
from logging import FileHandler
from typing import Optional, Union

import torch.nn as nn
from transformers.modeling_utils import unwrap_model

from .initialization import PretrainedInit, initialize, update_init_info


logger = logging.getLogger(__name__)


class BaseModule(nn.Module):
    """Base module for all Perceptia components.

    Provides unified weight initialization via `init_cfg`.
    Submodules inherit initialization info recursively.
    """

    def __init__(self, init_cfg: Union[dict, list[dict], None] = None) -> None:
        super().__init__()
        self.init_cfg = copy.deepcopy(init_cfg)
        self._is_init: bool = False

    @property
    def is_init(self) -> bool:
        return self._is_init

    @is_init.setter
    def is_init(self, value: bool) -> None:
        self._is_init = value

    def __repr__(self) -> str:
        rep = super().__repr__()
        if self.init_cfg:
            rep += f"\ninit_cfg={self.init_cfg}"
        return rep

    def init_weights(self) -> None:
        """Initialize module and all its children according to `init_cfg`."""
        is_top_level = False

        # Detect top-level module and register param tracking
        if not hasattr(self, "_params_init_info"):
            self._params_init_info = defaultdict(dict)
            is_top_level = True

            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is unchanged before and after `init_weights` of {self.__class__.__name__}"
                )
                self._params_init_info[param]["tmp_mean_value"] = param.data.mean().cpu()

            # Propagate shared tracking dict
            for m in self.modules():
                m._params_init_info = self._params_init_info

        if self._is_init:
            logger.warning(f"{self.__class__.__name__}.init_weights() called multiple times.")
            if is_top_level:
                self._finalize_init()
            return

        # Split init_cfg into pretrained and others
        other_cfgs, pretrained_cfgs = [], []
        if self.init_cfg:
            logger.debug(f"Initializing {self.__class__.__name__} with init_cfg={self.init_cfg}")

            cfgs = self.init_cfg if isinstance(self.init_cfg, list) else [self.init_cfg]
            for cfg in cfgs:
                assert isinstance(cfg, dict)
                if cfg.get("type") in {"Pretrained", PretrainedInit}:
                    pretrained_cfgs.append(cfg)
                else:
                    other_cfgs.append(cfg)

        # 1. Run non-pretrained initialization
        if other_cfgs:
            initialize(self, other_cfgs)

        # 2. Recursively initialize children
        for child in self.children():
            target = unwrap_model(child)
            if hasattr(target, "init_weights") and not getattr(target, "is_init", False):
                target.init_weights()
                update_init_info(
                    target, init_info=f"Initialized by user-defined init_weights in {target.__class__.__name__}"
                )

        # 3. Apply pretrained configs last (overwrite)
        if pretrained_cfgs:
            initialize(self, pretrained_cfgs)

        self._is_init = True

        if is_top_level:
            self._finalize_init()

    def _finalize_init(self) -> None:
        """Dump initialization info to file or logger, then clean up."""

        has_file_handler = False

        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write("Name of parameter - Initialization information\n")
                for name, param in self.named_parameters():
                    info = self._params_init_info[param]["init_info"]
                    handler.stream.write(f"\n{name} - {param.shape}:\n{info}\n")
                handler.stream.flush()
                has_file_handler = True
                break

        if not has_file_handler:
            for name, param in self.named_parameters():
                info = self._params_init_info[param]["init_info"]
                logger.info(f"\n{name} - {param.shape}:\n{info}\n")

        # Cleanup tracking
        for m in self.modules():
            del m._params_init_info


class Sequential(BaseModule, nn.Sequential):
    """Sequential container with Perceptia initialization."""

    def __init__(self, *modules: nn.Module, init_cfg: Optional[dict] = None) -> None:
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *modules)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList container with Perceptia initialization."""

    def __init__(self, modules: Optional[Iterable[nn.Module]] = None, init_cfg: Optional[dict] = None) -> None:
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules or [])


class ModuleDict(BaseModule, nn.ModuleDict):
    """ModuleDict container with Perceptia initialization."""

    def __init__(self, modules: Optional[dict[str, nn.Module]] = None, init_cfg: Optional[dict] = None) -> None:
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules or {})
