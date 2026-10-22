"""ormas-monitor: find the layer that broke while training runs, prove when a channel is dead
for good, optionally repair it, and keep a tamper-evident record. Runs entirely on your machine."""
__version__ = "0.2.0"

from .config import MonitorConfig  # noqa: E402
from .monitor import Monitor  # noqa: E402

__all__ = ["Monitor", "MonitorConfig", "__version__"]
