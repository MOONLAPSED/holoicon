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
from datetime import datetime
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
# platforms: Ubuntu-22.04LTS (posix), Windows-11 (nt)
#-------------------------------#####PLATFORM&LOGGING###########-------------------------------#
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
        handler.setFormatter(customFormatter())
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
if IS_WINDOWS:
    from ctypes import windll
    from ctypes.wintypes import DWORD, HANDLE
elif IS_POSIX:
    import resource  # Only used in POSIX
    from ctypes import CDLL, c_int, c_void_p, byref
    from ctypes.util import find_library

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
            # os.nice increments the niceness of a process by the specified amount.
            current_nice = os.nice(0)  # Get current niceness
            os.nice(priority)  # Increment niceness by priority
            logger.info(f"Adjusted POSIX process niceness by {priority}. Current niceness: {current_nice + priority}.")
        except PermissionError:
            logger.warning("Permission denied: Unable to set process niceness.")
        except Exception as e:
            logger.warning(f"Failed to set process niceness on POSIX: {e}")
    else:
        logger.warning("Unsupported operating system for setting process priority.")
#-------------------------------########DECORATORS#############-------------------------------#
"""
We can assume that imperative deterministic source code, such as this file written in Python, is capable of reasoning about non-imperative non-deterministic source code as if it were a defined and known quantity. This is akin to nesting a function with a value in an S-Expression.

In order to expect any runtime result, we must assume that a source code configuration exists which will yield that result given the input.

The source code configuration is the set of all possible configurations of the source code. It is the union of the possible configurations of the source code.

Imperative programming specifies how to perform tasks (like procedural code), while non-imperative (e.g., functional programming in LISP) focuses on what to compute. We turn this on its head in our imperative non-imperative runtime by utilizing nominative homoiconistic reflection to create a runtime where dynamical source code is treated as both static and dynamic.

"Nesting a function with a value in an S-Expression":
In the code, we nest the input value within different function expressions (configurations).
Each function is applied to the input to yield results, mirroring the collapse of the wave function to a specific state upon measurement.

This nominative homoiconistic reflection combines the expressiveness of S-Expressions with the operational semantics of Python. In this paradigm, source code can be constructed, deconstructed, and analyzed in real-time, allowing for dynamic composition and execution. Each code configuration (or state) is akin to a function in an S-Expression that can be encapsulated, manipulated, and ultimately evaluated in the course of execution.

To illustrate, consider a Python function as a generalized S-Expression. This function can take other functions and values as arguments, forming a nested structure. Each invocation changes the system's state temporarily, just as evaluating an S-Expression alters the state of the LISP interpreter.

In essence, our approach ensures that:

1. **Composition**: Functions (or code segments) can be composed at runtime, akin to how S-Expressions can nest functions and values.
2. **Evaluation**: Upon invocation, these compositions are evaluated, reflecting the current configuration of the runtime.
3. **Reflection and Modification**: The runtime can reflect on its structure and make modifications dynamically, which allows it to reason about its state and adapt accordingly.
4. **Identity Preservation**: The runtime maintains its identity, allowing for a consistent state across different configurations.
5. **Non-Determinism**: The runtime can exhibit non-deterministic behavior, as it can transition between different configurations based on the input and the code's structure. This is akin to the collapse of the wave function in quantum mechanics, or modeling it on classical hardware via multi-instantaneous multi-threading.
6. **State Preservation**: The runtime can maintain its state across different configurations, allowing for a consistent execution path.
7. **Compositionality**: The runtime can compose different configurations, allowing for a rich variety of behaviors.

This synthesis of static and dynamic code concepts is akin to the Copenhagen interpretation of quantum mechanics, where the observation (or execution) collapses the superposition of states (or configurations) into a definite outcome based on the input.

Ultimately, this model provides a flexible approach to managing and executing complex code structures dynamically while maintaining the clarity and compositional advantages traditionally seen in non-imperative, functional paradigms like LISP, drawing inspiration from lambda calculus and functional programming principles.

The most advanced concept of all in this ontology is the dynamic rewriting of source code at runtime. Source code rewriting is achieved with a special runtime `Atom()` class with 'modified quine' behavior. This special Atom, aside from its specific function and the functions obligated to it by polymorphism, will always rewrite its own source code but may also perform other actions as defined by the source code in the runtime which invoked it. They can be nested in S-expressions and are homoiconic with all other source code. These modified quines can be used to dynamically create new code at runtime, which can be used to extend the source code in a way that is not known at the start of the program. This is the most powerful feature of the system and allows for the creation of a runtime of runtimes dynamically limited by hardware and the operating system.
"""
# non-homoiconic pre-runtime "ADMIN-SCOPED" source code:
@dataclass
class RuntimeState:
    current_step: int = 0
    variables: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    IS_POSIX = os.name == 'posix'
    IS_WINDOWS = not IS_POSIX  # Assume Windows if WSL is not detected
    # platforms: Ubuntu-22.04LTS, Windows-11
    if os.name == 'posix':
        from ctypes import cdll
    elif os.name == 'nt':
        from ctypes import windll
@dataclass
class AppState:
    pdm_installed: bool = False
    virtualenv_created: bool = False
    dependencies_installed: bool = False
    lint_passed: bool = False
    code_formatted: bool = False
    tests_passed: bool = False
    benchmarks_run: bool = False
    pre_commit_installed: bool = False
@dataclass
class FilesystemState:
    allowed_root: Path = field(init=False)
    def __post_init__(self):
        try:
            self.allowed_root = Path(__file__).resolve().parent
            if not any(self.allowed_root.iterdir()):
                raise FileNotFoundError(f"Allowed root directory empty: {self.allowed_root}")
            logging.info(f"Allowed root directory found: {self.allowed_root}")
        except Exception as e:
            logging.error(f"Error initializing FilesystemState: {e}")
            raise
    def safe_remove(self, path: Path):
        """Safely remove a file or directory, handling platform-specific issues."""
        try:
            path = path.resolve()
            if not path.is_relative_to(self.allowed_root):
                logging.error(f"Attempt to delete outside allowed directory: {path}")
                return
            if path.is_dir():
                shutil.rmtree(path)
                logging.info(f"Removed directory: {path}")
            else:
                path.unlink()
                logging.info(f"Removed file: {path}")
        except (FileNotFoundError, PermissionError, OSError) as e:
            logging.error(f"Error removing path {path}: {e}")
    def _on_error(self, func, path, exc_info):
        """Error handler for handling removal of read-only files on Windows."""
        logging.error(f"Error deleting {path}, attempting to fix permissions.")
        # Attempt to change the file's permissions and retry removal
        os.chmod(path, 0o777)
        func(path)
    async def execute_runtime_tasks(self):
        for task in self.tasks:
            try:
                await task()
            except Exception as e:
                logging.error(f"Error executing task: {e}")
    async def run_command_async(command: str, shell: bool = False, timeout: int = 120):
        logging.info(f"Running command: {command}")
        split_command = shlex.split(command, posix=(os.name == 'posix'))
        try:
            process = await asyncio.create_subprocess_exec(*split_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, shell=shell)
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return {
                "return_code": process.returncode,
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else "",
            }
        except asyncio.TimeoutError:
            logging.error(f"Command '{command}' timed out.")
            return {"return_code": -1, "output": "", "error": "Command timed out"}
        except Exception as e:
            logging.error(f"Error running command '{command}': {str(e)}")
            return {"return_code": -1, "output": "", "error": str(e)}
class AccessLevel(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
@dataclass
class AccessPolicy:
    level: AccessLevel
    namespace_patterns: list[str] = field(default_factory=list)
    allowed_operations: list[str] = field(default_factory=list)
    def can_access(self, namespace: str, operation: str) -> bool:
        if any(pattern in namespace for pattern in self.namespace_patterns):
            return operation in self.allowed_operations
        return False
class SecurityContext:
    def __init__(self, user_id: str, access_policy: AccessPolicy):
        self.user_id = user_id
        self.access_policy = access_policy
        self._audit_log = []
    def log_access(self, namespace: str, operation: str, success: bool):
        self._audit_log.append({
            "user_id": self.user_id,
            "namespace": namespace,
            "operation": operation,
            "success": success,
            "timestamp": asyncio.get_event_loop().time()
        })
class RuntimeNamespace:
    def __init__(self, name: str, parent: Optional['RuntimeNamespace'] = None):
        self._name = name
        self._parent = parent
        self._children: Dict[str, 'RuntimeNamespace'] = {}
        self._content = SimpleNamespace()
        self._security_context: Optional[SecurityContext] = None
    @property
    def full_path(self) -> str:
        if self._parent:
            return f"{self._parent.full_path}.{self._name}"
        return self._name
    def add_child(self, name: str) -> 'RuntimeNamespace':
        child = RuntimeNamespace(name, self)
        self._children[name] = child
        return child
    def get_child(self, path: str) -> Optional['RuntimeNamespace']:
        parts = path.split(".", 1)
        if len(parts) == 1:
            return self._children.get(parts[0])
        child = self._children.get(parts[0])
        if child and len(parts) > 1:
            return child.get_child(parts[1])
        return None
class RuntimeManager:
    def __init__(self):
        self.root = RuntimeNamespace("root")
        self._security_contexts: Dict[str, SecurityContext] = {}
    def register_user(self, user_id: str, access_policy: AccessPolicy):
        self._security_contexts[user_id] = SecurityContext(user_id, access_policy)
    async def execute_query(self, user_id: str, query: str) -> Any:
        security_context = self._security_contexts.get(user_id)
        if not security_context:
            raise PermissionError("User not registered")
        try:
            # Parse query and validate
            parsed = ast.parse(query, mode='eval')
            validator = QueryValidator(security_context)
            validator.visit(parsed)
            # Execute in isolated namespace
            namespace = self._create_restricted_namespace(security_context)
            result = eval(compile(parsed, '<string>', 'eval'), namespace)
            security_context.log_access(
                namespace="query_execution",
                operation="execute",
                success=True
            )
            return result
        except Exception as e:
            security_context.log_access(
                namespace="query_execution",
                operation="execute",
                success=False
            )
            raise RuntimeError(f"Query execution failed: {str(e)}")
    def _create_restricted_namespace(self, security_context: SecurityContext) -> dict:
        """Creates a restricted namespace based on user's access policy"""
        base_namespace = {}
        # Add allowed builtins based on access policy
        if security_context.access_policy.level in [AccessLevel.EXECUTE, AccessLevel.ADMIN]:
            safe_builtins = {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'print': print
            }
            base_namespace.update(safe_builtins)
        return base_namespace
def isModule(rawClsOrFn: Union[Type, Callable]) -> Optional[str]:
    pyModule = inspect.getmodule(rawClsOrFn)
    if hasattr(pyModule, "__file__"):
        return str(Path(pyModule.__file__).resolve())
    return None
def getModuleImportInfo(rawClsOrFn: Union[Type, Callable]) -> Tuple[Optional[str], str, str]:
    """
    Given a class or function in Python, get all the information needed to import it in another Python process.
    This version balances portability and optimization using camel case.
    """
    pyModule = inspect.getmodule(rawClsOrFn)
    if pyModule is None or pyModule.__name__ == '__main__':
        return None, 'interactive', rawClsOrFn.__name__
    modulePath = isModule(rawClsOrFn)
    if not modulePath:
        # Built-in or frozen module
        return None, pyModule.__name__, rawClsOrFn.__name__
    rootPath = str(Path(modulePath).parent)
    moduleName = pyModule.__name__
    clsOrFnName = getattr(rawClsOrFn, "__qualname__", rawClsOrFn.__name__)
    if getattr(pyModule, "__package__", None):
        try:
            package = __import__(pyModule.__package__)
            packagePath = str(Path(package.__file__).parent)
            if Path(packagePath) in Path(modulePath).parents:
                rootPath = str(Path(packagePath).parent)
            else:
                print(f"Warning: Module is not in the expected package structure. Using file parent as root path.")
        except Exception as e:
            print(f"Warning: Error processing package structure: {e}. Using file parent as root path.")

    return rootPath, moduleName, clsOrFnName
class QueryValidator(ast.NodeVisitor):
    def __init__(self, security_context: SecurityContext):
        self.security_context = security_context

    def visit_Name(self, node):
        # Validate access to variables
        if not self.security_context.access_policy.can_access(
            node.id, "read"
        ):
            raise PermissionError(f"Access denied to name: {node.id}")
        self.generic_visit(node)
    def visit_Call(self, node):
        # Validate function calls
        if isinstance(node.func, ast.Name):
            if not self.security_context.access_policy.can_access(
                node.func.id, "execute"
            ):
                raise PermissionError(f"Access denied to function: {node.func.id}")
        self.generic_visit(node)
# STATIC TYPING ========================================================    
"""Homoiconism dictates that, upon runtime validation, all objects are code and data.
To fascilitate; we utilize first class functions and a static typing system."""
T = TypeVar('T', bound=any) # T for TypeVar, V for ValueVar. Homoicons are T+V.
V = TypeVar('V', bound=Union[int, float, str, bool, list, dict, tuple, set, object, Callable, type])
C = TypeVar('C', bound=Callable[..., Any])  # callable 'T'/'V' first class function interface
DataType = Enum('DataType', 'INTEGER FLOAT STRING BOOLEAN NONE LIST TUPLE') # 'T' vars (stdlib)
AtomType = Enum('AtomType', 'FUNCTION CLASS MODULE OBJECT') # 'C' vars (homoiconic methods or classes)
# HOMOICONISTIC morphological source code displays 'modified quine' behavior
# within a validated runtime, if and only if the valid python interpreter
# has r/w/x permissions to the source code file and some method of writing
# state to the source code file is available. Any interruption of the
# '__exit__` method or misuse of '__enter__' will result in a runtime error
# AP (Availability + Partition Tolerance): A system that prioritizes availability and partition
# tolerance may use a distributed architecture with eventual consistency (e.g., Cassandra or Riak).
# This ensures that the system is always available (availability), even in the presence of network
# partitions (partition tolerance). However, the system may sacrifice consistency, as nodes may have
# different views of the data (no consistency). A homoiconic piece of source code is eventually
# consistent, assuming it is able to re-instantiated.
# DECORATORS =========================================================
def atom(cls: Type[{T, V, C}]) -> Type[{T, V, C}]: # homoicon decorator
    """Decorator to create a homoiconic atom."""
    original_init = cls.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, 'id'):
            self.id = hashlib.sha256(self.__class__.__name__.encode('utf-8')).hexdigest()

    cls.__init__ = new_init
    return cls
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
@log()
def snapShot(func: Callable) -> Callable:
    """
    Capture memory snapshots before and after function execution. OBJECT not a wrapper
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        displayTop(snapshot)
        return result
    return wrapper

#-------------------------------############MAIN###############-------------------------------#
class main():
    """
    Main function to demonstrate the usage of the decorators
    """
    @classmethod
    @log()
    async def asyncFunction(cls, arg):
        logger.info(f"Async function called with arg: {arg}")
        return arg
    @classmethod
    @log()
    def syncFunction(cls, arg):
        logger.info(f"Sync function called with arg: {arg}")
        return arg

    @classmethod
    async def asyncMain(cls):
        logger.info("Starting async main")
        result = await cls.asyncFunction("asyncArg")
        logger.info(f"Async result: {result}")
    
    @classmethod
    def syncMain(cls):
        logger.info("Starting sync main")
        result = cls.syncFunction("syncArg")
        logger.info(f"Sync result: {result}")
    
    runAsync = lambda f: asyncio.run(f()) if asyncio.iscoroutinefunction(f) else f()

if __name__ == "__main__":
    """main runtime for CLI, logging, and initialization."""
    args = logArgs()  # Parse logging arguments
    
    if IS_WINDOWS:
    # windowsOptions = ['IDLE', 'BELOW_NORMAL', 'NORMAL', 'ABOVE_NORMAL',
    #   'HIGH', 'REALTIME']
        set_process_priority('NORMAL')
    elif IS_POSIX:
        set_process_priority(0)
    
    try:
        logger = setupLogger(args.log_name, args.log_level, args.log_datefmt, [logging.StreamHandler()])
        m = main
        
        def print_help():
            print("Available commands:")
            print("  async - Run async function")
            print("  sync  - Run sync function")
            print("  help  - Display this help message")
            print("  quit  - Exit the program")

        print_help()
        while True:
            try:
                command = input("Enter command: ").lower().strip()
                if command in ('async', 'a'):
                    print("Running async function...")
                    m.runAsync(m.asyncMain)
                elif command in ('sync', 's'):
                    print("Running sync function...")
                    m.syncMain()
                elif command in ('help', 'h'):
                    print_help()
                elif command in ('quit', 'q', 'exit'):
                    print("Exiting program...")
                    break
                else:
                    print("Invalid command. Type 'help' for available commands.")
            except KeyboardInterrupt:
                print("\nProgram interrupted. Type 'quit' to exit.")
        
        logger.info("Main function completed successfully.")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
