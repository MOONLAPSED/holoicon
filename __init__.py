#!/usr/bin/env python
# -*- coding: utf-8 -*-
# STATE_START
{
  "current_step": 0
}
# STATE_END
import asyncio
import inspect
import json
import logging
import os
import hashlib
import platform
import pathlib
import struct
import sys
import threading
import time
import shlex
import shutil
import uuid
import argparse
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps, lru_cache
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple, Generic, Set, Coroutine, 
    Type, NamedTuple, ClassVar, Protocol
    )
from types import SimpleNamespace
from queue import Queue, Empty
from asyncio import Queue as AsyncQueue
import ctypes
import ast
import tokenize
import io
import importlib as _importlib
from importlib.util import spec_from_file_location, module_from_spec
import re
import dis
import tokenize
import linecache
import tracemalloc
tracemalloc.start()
# Specify the files and lines to exclude from tracking
tracefilter = ("<frozen importlib._bootstrap>", "<frozen importlib._bootstrap_external>")
tracemalloc.BaseFilter(tracefilter)
def display_top(snapshot, key_type='lineno', limit=3):
    """Display top memory consuming lines"""
    filters = [tracemalloc.Filter(False, item) for item in tracefilter]
    snapshot = snapshot.filter_traces(filters)  # Apply the filters to the snapshot
    top_stats = snapshot.statistics(key_type)
    print("Top {} lines".format(limit))

    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#{}: {}:{}: {:.1f} KiB".format(index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    {}'.format(line))
    
    # Show the total size and count of other items
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("{} other: {:.1f} KiB".format(len(other), size / 1024))

    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: {:.1f} KiB".format(total / 1024))

# main() ->
logger = logging.getLogger(__name__)
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)
# <- end main() (testing main)

# Platform-specific optimizations
if os.name == 'nt':
    import win32api
    import win32process

    def set_process_priority(priority: int):
        handle = win32api.GetCurrentProcess()
        win32process.SetPriorityClass(handle, priority)
    
    def set_thread_priority(priority: int):
        handle = win32api.GetCurrentThread()
        win32process.SetThreadPriority(handle, priority)
    try:
        set_process_priority(win32process.REALTIME_PRIORITY_CLASS)
        set_thread_priority(win32process.THREAD_PRIORITY_TIME_CRITICAL)
    except Exception as e:
        logger.warning(f"Failed to set process priority: {e}")
    finally:
        logger.info("'nt' platform detected, optimizations applied.")

elif os.name == 'posix':
    import resource

    def set_process_priority(priority: int):
        try:
            os.nice(priority)
        except PermissionError:
            logger.warning("Unable to set process priority. Running with default priority.")

#-------------------------------###############################-------------------------------#

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

#-------------------------------###########MIXINS##############-------------------------------#

def load_modules():
    try:
        mixins = []
        for path in pathlib.Path(__file__).parent.glob("*.py"):
            if path.name.startswith("_"):
                continue
            module_name = path.stem
            spec = spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module {module_name}")
            module = module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            mixins.append(module)
        return mixins
    except Exception as e:
        logger.error(f"Error importing internal modules: {e}")
        sys.exit(1)

mixins = load_modules() # Import the internal modules

if mixins:
    __all__ = [mixin.__name__ for mixin in mixins]
else:
    __all__ = []


""" hacked namespace uses `__all__` as a whitelist of symbols which are executable source code.
Non-whitelisted modules or runtime SimpleNameSpace()(s) are treated as 'data' which we call associative 
'articles' within the knowledge base, loaded at runtime. They are, however, logic and state."""

"""
class KnowledgeBase:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.globals = SimpleNamespace()
        self.globals.__all__ = []
        self.initialize()

    def initialize(self):
        self._import_py_modules(self.base_dir)
        self._load_articles(self.base_dir)

    def _import_py_modules(self, directory):
        for path in directory.rglob("*.py"):
            if path.name.startswith("_"):
                continue
            try:
                module_name = path.stem
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                setattr(self.globals, module_name, module)
                self.globals.__all__.append(module_name)
            except Exception as e:
                print(f"Error importing module {module_name}: {e}")

    def _load_articles(self, directory):
        for suffix in ['*.md', '*.txt']:
            for path in directory.rglob(suffix):
                try:
                    article_name = path.stem
                    content = path.read_text()
                    article = SimpleNamespace(
                        content=content,
                        path=str(path)
                    )
                    setattr(self.globals, article_name, article)
                except Exception as e:
                    print(f"Error loading article from {path}: {e}")

    def execute_query(self, query):
        try:
            parsed = ast.parse(query, mode='eval')
            result = eval(compile(parsed, '<string>', 'eval'), {'kb': self.globals})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def commit_changes(self):
        # TODO: Implement logic to write changes back to the file system
        pass

def initialize_kb(base_dir):
    return KnowledgeBase(base_dir)
"""

#-------------------------------###############################-------------------------------#
#-------------------------------########DECORATORS#############-------------------------------#
#-------------------------------###############################-------------------------------#
def memoize(func: Callable) -> Callable:
    return lru_cache(maxsize=None)(func)

def log(level=logging.INFO):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = await func(*args, **kwargs)
                logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                logger.exception(f"Error in {func.__name__}: {e}")
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