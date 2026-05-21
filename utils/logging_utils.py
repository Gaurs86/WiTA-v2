"""utils/logging_utils.py — Logging setup."""
import logging, os, sys


def setup(log_dir: str = "/kaggle/working/logs", level: int = logging.INFO) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "run.log")),
        ],
    )
