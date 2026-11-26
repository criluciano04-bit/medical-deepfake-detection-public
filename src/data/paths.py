
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def get_data_root() -> Path:
    """Return the configured root directory of the selected datasets."""
    value = os.getenv("DATA_ROOT")

    if not value:
        raise RuntimeError(
            "DATA_ROOT is not configured. "
            "Set it in the .env file or as an environment variable."
        )

    root = Path(value).expanduser()

    if not root.is_dir():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist or is not a directory: {root}"
        )

    return root
