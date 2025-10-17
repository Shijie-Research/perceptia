# Copyright (c) Perceptia Contributors. All rights reserved.
import hydra

from perceptia_cli.train import main


@hydra.main(config_path="./configs", config_name="runtime", version_base=None)
def cli_main(cfg):
    """Hydra entry point for Perceptia training."""
    main(cfg)


if __name__ == "__main__":
    cli_main()
