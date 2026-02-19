from datetime import datetime
from pathlib import Path

import logging
from logging.handlers import RotatingFileHandler

# -------------------------LOGGING---------------------------------
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("langgraph-agent")
logger.setLevel(logging.DEBUG)

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=1_000_000,  # 1MB
    backupCount=5
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.propagate = False
logger.info("[INFO] Logging started.")
# -----------------------------------------------------------------
