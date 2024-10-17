import os
import sys
import logging
import asyncio
import tracemalloc
import linecache
import ctypes
import resource  # Only used in POSIX
from contextlib import contextmanager
from functools import wraps, lru_cache
from enum import Enum, auto
from typing import Callable, Optional
#-------------------------------###############################-------------------------------#
#-------------------------------#########PLATFORM##############-------------------------------#
#-------------------------------########&LOGGING###############-------------------------------#
class customFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setupLogger(name: str, level: int, datefmt: str, handlers: list):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()
    for handler in handlers:
        if not isinstance(handler, logging.Handler):
            raise ValueError(f"Invalid handler provided: {handler}")
        handler.setLevel(level)
        handler.setFormatter(CustomFormatter())
        logger.addHandler(handler)
    return logger

def logArgs():
    parser = argparse.ArgumentParser(description="Logger Configuration")
    parser.add_argument('--log-level', type=str, default='DEBUG', choices=logging._nameToLevel.keys(), help='Set logging level')
    parser.add_argument('--log-file', type=str, help='Set log file path')
    parser.add_argument('--log-format', type=str, default='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)', help='Set log format')
    parser.add_argument('--log-datefmt', type=str, default='%Y-%m-%d %H:%M:%S', help='Set date format')
    parser.add_argument('--log-name', type=str, default=__name__, help='Set logger name')
    return parser.parse_args()

def parseLargs():
    args = parse_args()
    log_level = logging._nameToLevel.get(args.log_level.upper(), logging.DEBUG)

    handlers = [logging.FileHandler(args.log_file)] if args.log_file else [logging.StreamHandler()]

    logger = setup_logger(name=args.log_name, level=log_level, datefmt=args.log_datefmt, handlers=handlers)
    logger.info("Logger setup complete.")

logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == 'nt'
IS_POSIX = os.name == 'posix'

def set_process_priority(priority: int):
    """
    Set the process priority based on the operating system.
    """
    if IS_WINDOWS:
        try:
            # Define priority classes
            priority_classes = {
                'IDLE': 0x40,
                'BELOW_NORMAL': 0x4000,
                'NORMAL': 0x20,
                'ABOVE_NORMAL': 0x8000,
                'HIGH': 0x80,
                'REALTIME': 0x100
            }
            # Load necessary Windows APIs using ctypes
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            handle = kernel32.GetCurrentProcess()
            if not kernel32.SetPriorityClass(handle, priority_classes.get(priority, 0x20)):
                raise ctypes.WinError(ctypes.get_last_error())
            logger.info(f"Set Windows process priority to {priority}.")
        except Exception as e:
            logger.warning(f"Failed to set process priority on Windows: {e}")

    elif IS_POSIX:
        try:
            # os.nice increments the niceness; to set absolute niceness, you might need a different approach
            current_nice = os.nice(0)  # Get current niceness
            os.nice(priority)  # Increment niceness by priority
            logger.info(f"Adjusted POSIX process niceness by {priority}. Current niceness: {current_nice + priority}.")
        except PermissionError:
            logger.warning("Permission denied: Unable to set process niceness.")
        except Exception as e:
            logger.warning(f"Failed to set process niceness on POSIX: {e}")
    else:
        logger.warning("Unsupported operating system for setting process priority.")
#-------------------------------###############################-------------------------------#
#-------------------------------########DECORATORS#############-------------------------------#
#-------------------------------###############################-------------------------------#

def memoize(func: Callable) -> Callable:
    """
    Caching decorator using LRU cache with unlimited size.
    """
    return lru_cache(maxsize=None)(func)

@contextmanager
def memoryProfiling(active: bool = True):
    """
    Context manager for memory profiling using tracemalloc.
    Captures allocations made within the context block.
    """
    if active:
        tracemalloc.start()
        try:
            yield
        finally:
            snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            displayTop(snapshot)
    else:
        yield None
def timeFunc(func: Callable) -> Callable:
    """
    Time execution of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper
def displayTop(snapshot, key_type: str = 'lineno', limit: int = 3):
    """
    Display top memory-consuming lines.
    """
    tracefilter = ("<frozen importlib._bootstrap>", "<frozen importlib._bootstrap_external>")
    filters = [tracemalloc.Filter(False, item) for item in tracefilter]
    filtered_snapshot = snapshot.filter_traces(filters)
    topStats = filtered_snapshot.statistics(key_type)

    result = [f"Top {limit} lines:"]
    for index, stat in enumerate(topStats[:limit], 1):
        frame = stat.traceback[0]
        result.append(f"#{index}: {frame.filename}:{frame.lineno}: {stat.size / 1024:.1f} KiB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            result.append(f"    {line}")

    # Show the total size and count of other items
    other = topStats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        result.append(f"{len(other)} other: {size / 1024:.1f} KiB")

    total = sum(stat.size for stat in topStats)
    result.append(f"Total allocated size: {total / 1024:.1f} KiB")
    logger.info("\n".join(result))

def log(level: int = logging.INFO):
    """
    Logging decorator for functions. Handles both synchronous and asynchronous functions.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.log(level, f"Executing async {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = await func(*args, **kwargs)
                logger.log(level, f"Completed async {func.__name__} with result: {result}")
                return result
            except Exception as e:
                logger.exception(f"Error in async {func.__name__}: {e}")
                raise
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                logger.exception(f"Error in {func.__name__}: {e}")
                raise
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
#-------------------------------###############################-------------------------------#
@log()
def main():
    def __init__(self, name: str, age: int):
        self.snapshot = None
    with memoryProfiling(active=True) as snapshot:
        print("Starting memory profiling...")
        lambda x: x + 1
        async_function = lambda: asyncio.sleep(2)
        rwaitable = asyncio.run(async_function())
        print(type(snapshot))
        print(snapshot)
        print("Finished memory profiling.")
if __name__ == "__main__":
    if IS_WINDOWS:
        set_process_priority('NORMAL')  # Options: 'IDLE', 'BELOW_NORMAL', 'NORMAL', 'ABOVE_NORMAL', 'HIGH', 'REALTIME'
    elif IS_POSIX:
        set_process_priority(0)  # Adjust niceness
    try:
        main()
        logger.info("Main function completed successfully.")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
