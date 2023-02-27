import logging
from typing import Any, Dict, List, Mapping, Optional

import wandb

from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.loggers.logger import MetricLogger

logger: logging.Logger = logging.getLogger(__name__)


class WandbLogger(MetricLogger):

    def __init__(self) -> None:
        """Create a minimal Weights and Biases logger
        """

        self._rank: int = get_global_rank()

        if self._rank == 0:
            logger.info(
                f"WandbLogger instantiated"
            )
        else:
            logger.debug(
                f"Not logging metrics on this host because env RANK: {self._rank} != 0"
            )
        

    def log(self, d:dict) -> None:
        """Log a d to W&B
        """

        wandb.log(d)


    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        wandb.finish()