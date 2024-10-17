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
excludeFiles = ["__init__.py"]
tracemalloc.BaseFilter(
    excludeFiles
)
logger = logging.getLogger(__name__)
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
        for path in directory.rglob("*.py"): # Recursively find all .py files
            if path.name.startswith("_"):
                continue  # Skip private modules
            try:
                module_name = path.stem
                spec = spec_from_file_location(module_name, str(path))
                if spec and spec.loader:
                    module = module_from_spec(spec)
                    spec.loader.exec_module(module)
                    setattr(self.globals, module_name, module)
                    self.globals.__all__.append(module_name)
            except Exception as e:
                # Use logger to log exceptions rather than printing
                logger.exception(f"Error importing module {module_name}: {e}")

    def _load_articles(self, directory):
        for suffix in ['*.md', '*.txt']:
            for path in directory.rglob(suffix): # Recursively find all .md and .txt files
                try:
                    article_name = path.stem
                    content = path.read_text()
                    article = SimpleNamespace(
                        content=content,
                        path=str(path)
                    )
                    setattr(self.globals, article_name, article)
                except Exception as e:
                    # Use logger to log exceptions rather than printing
                    logger.exception(f"Error loading article from {path}: {e}")

    def execute_query(self, query):
        try:
            parsed = ast.parse(query, mode='eval')
            # execute compiled code in the context of self.globals
            result = eval(compile(parsed, '<string>', 'eval'), vars(self.globals))
            return str(result)
        except Exception as e:
            # Return a more user-friendly error message
            logger.exception("Query execution error:")
            return f"Error executing query: {str(e)}"

    def execute_query_async(self, query):
        try:
            parsed = ast.parse(query, mode='eval')
            # execute compiled code in the context of self.globals
            result = eval(compile(parsed, '<string>', 'eval'), vars(self.globals))
            return str(result)
        except Exception as e:
            # Return a more user-friendly error message
            logger.exception("Query execution error:")
            return f"Error executing query: {str(e)}"

@log()
def main():
    with memory_profiling() as snapshot:
        # Your application logic here
        # For demonstration, we'll perform a memory-intensive operation
        dummy_list = [i for i in range(1000000)]
    
    if snapshot:
        display_top(snapshot)

if __name__ == "__main__":
    set_process_priority(priority=0)  # Adjust priority as needed (optional)

    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")