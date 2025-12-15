import sys

from loguru import logger

# Set logging level to CRITICAL so it doesn't show
# in test output but is still captured for testing.
logger.remove()
logger.add(sys.stderr, level="CRITICAL")
