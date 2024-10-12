#!/usr/bin/env python
# -*- coding: utf-8 -*-
# STATE_START
{
  "current_step": 0
}
# STATE_END
import os
import sys
import io
import re
import dis
import ast
import tokenize
import importlib
import pathlib
import asyncio
import argparse
import uuid
import json
import struct
import time
import hashlib
import msgpack
import dis
import inspect
import threading
import logging
import time
import shlex
import shutil
import uuid
import datetime
import argparse
import ctypes
import tracemalloc
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple, Generic, Set, Coroutine, Type, NamedTuple
)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from asyncio import Queue as AsyncQueue
from queue import Queue, Empty
from functools import wraps
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
tracemalloc.start()
tracefilter = ("<<frozen importlib._bootstrap>", "<frozen importlib._bootstrap_external>")
snapshot = tracemalloc.take_snapshot()
excluded_files = [trace.filename for trace in snapshot.traces if trace.filename not in tracefilter]
for filename in excluded_files:
    tracemalloc.Filter(False, filename)
# tracemalloc.Filter(False, "*") uncomment for no debugging
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(True, "<module>"),
    ))
    top_stats = snapshot.statistics(key_type)
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)
# Typing ----------------------------------------------------------
"""Homoiconism dictates that, upon runtime validation, all objects are code and data.
To fascilitate; we utilize first class functions and a static typing system."""
T = TypeVar('T', bound=any) # T for TypeVar, V for ValueVar. Homoicons are T+V.
V = TypeVar('V', bound=Union[int, float, str, bool, list, dict, tuple, set, object, Callable, type])
C = TypeVar('C', bound=Callable[..., Any])  # callable 'T'/'V' first class function interface
DataType = Enum('DataType', 'INTEGER FLOAT STRING BOOLEAN NONE LIST TUPLE') # 'T' vars (stdlib)
AtomType = Enum('AtomType', 'FUNCTION CLASS MODULE OBJECT') # 'C' vars (homoiconic methods or classes)
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

def log(level=logging.INFO):
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            Logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = await func(*args, **kwargs)
                Logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                Logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            Logger.log(level, f"Executing {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                Logger.log(level, f"Completed {func.__name__} with result: {result}")
                return result
            except Exception as e:
                Logger.exception(f"Error in {func.__name__}: {str(e)}")
                raise
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def validate(cls: Type[T]) -> Type[T]:
    original_init = cls.__init__
    sig = inspect.signature(original_init)

    def new_init(self: T, *args: Any, **kwargs: Any) -> None:
        bound_args = sig.bind(self, *args, **kwargs)
        for key, value in bound_args.arguments.items():
            if key in cls.__annotations__:
                expected_type = cls.__annotations__.get(key)
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected {expected_type} for {key}, got {type(value)}")
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls

def encode(atom: 'Atom') -> bytes:
    data = {
        'tag': atom.tag,
        'value': atom.value,
        'children': [encode(child) for child in atom.children],
        'metadata': atom.metadata
    }
    return pickle.dumps(data)

def decode(data: bytes) -> 'Atom':
    data = pickle.loads(data)
    atom = Atom(data['tag'], data['value'], [decode(child) for child in data['children']], data['metadata'])
    return atom

@dataclass
class GrammarRule:
    """
    Represents a single grammar rule in a context-free grammar.
    
    Attributes:
        lhs (str): Left-hand side of the rule.
        rhs (List[Union[str, 'GrammarRule']]): Right-hand side of the rule, which can be terminals or other rules.
    """
    lhs: str
    rhs: List[Union[str, 'GrammarRule']]
    
    def __repr__(self):
        """
        Provide a string representation of the grammar rule.
        
        Returns:
            str: The string representation.
        """
        rhs_str = ' '.join([str(elem) for elem in self.rhs])
        return f"{self.lhs} -> {rhs_str}"


class Atom(ABC):
    """
    Abstract Base Class for all Atom types.
    
    Atoms are the smallest units of data or executable code, and this interface
    defines common operations such as encoding, decoding, execution, and conversion
    to data classes.
    
    Attributes:
        grammar_rules (List[GrammarRule]): List of grammar rules defining the syntax of the Atom.
    """
    grammar_rules: List[GrammarRule] = field(default_factory=list)
    id: str = field(init=False)
    tag: str = ''
    children: List['Atom'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y and y == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(True, x, y) if x == y else None
    case_base: Dict[str, Callable[..., bool]] = field(default_factory=dict)
    def __init__(self, id: str):
        self.id = id

    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            '¬': lambda a: not a,
            '∧': lambda a, b: a and b,
            '∨': lambda a, b: a or b,
            '→': lambda a, b: (not a) or b,
            '↔': lambda a, b: (a and b) or (not a and not b),
        }

    def encode(self) -> bytes:
        return json.dumps({
            'id': self.id,
            'attributes': self.attributes
        }).encode()

    @classmethod
    def decode(cls, data: bytes) -> 'Atom':
        decoded_data = json.loads(data.decode())
        return cls(id=decoded_data['id'], **decoded_data['attributes'])

    def introspect(self) -> str:
        """
        Reflect on its own code structure via AST.
        """
        source = inspect.getsource(self.__class__)
        return ast.dump(ast.parse(source))

    def __init__(self, value: Union[T, V, C], type: Union[DataType, AtomType]):
        self.value = value
        self.type = type
        self.hash = hashlib.sha256(repr(value).encode()).hexdigest()

    def __repr__(self):
        return f"{self.value} : {self.type}"

    def __str__(self):
        return str(self.value)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Atom) and self.hash == other.hash

    def __hash__(self) -> int:
        return int(self.hash, 16)

    def __buffer__(self, flags: int) -> memoryview:
        return memoryview(self.value)

    async def send_message(self, message: Any, ttl: int = 3) -> None:
        if ttl <= 0:
            logging.info(f"Message {message} dropped due to TTL")
            return
        logging.info(f"Atom {self.id} received message: {message}")
        for sub in self.subscribers:
            await sub.receive_message(message, ttl - 1)

    async def receive_message(self, message: Any, ttl: int) -> None:
        logging.info(f"Atom {self.id} processing received message: {message} with TTL {ttl}")
        await self.send_message(message, ttl)

    def subscribe(self, atom: 'Atom') -> None:
        self.subscribers.add(atom)
        logging.info(f"Atom {self.id} subscribed to {atom.id}")

    def unsubscribe(self, atom: 'Atom') -> None:
        self.subscribers.discard(atom)
        logging.info(f"Atom {self.id} unsubscribed from {atom.id}")
    # Use __slots__ for the rest of the methods to save memory
    __slots__ = ('value', 'type', 'hash')
    __getitem__ = lambda self, key: self.value[key]
    __setitem__ = lambda self, key, value: setattr(self.value, key, value)
    __delitem__ = lambda self, key: delattr(self.value, key)
    __len__ = lambda self: len(self.value)
    __iter__ = lambda self: iter(self.value)
    __contains__ = lambda self, item: item in self.value
    __call__ = lambda self, *args, **kwargs: self.value(*args, **kwargs)

    __add__ = lambda self, other: self.value + other
    __sub__ = lambda self, other: self.value - other
    __mul__ = lambda self, other: self.value * other
    __truediv__ = lambda self, other: self.value / other
    __floordiv__ = lambda self, other: self.value // other


class BaseContextManager(ABC):
    """
    Defines the interface for a context manager, ensuring a resource is properly
    managed, with setup before entering the context and cleanup after exiting.

    This abstract base class must be subclassed to implement the `__enter__` and
    `__exit__` methods, enabling use with the `with` statement for resource
    management, such as opening and closing files, acquiring and releasing locks,
    or establishing and terminating network connections.

    Implementers should override the `__enter__` and `__exit__` methods according to
    the resource's specific setup and cleanup procedures.

    Methods
    -------
    __enter__()
        Called when entering the runtime context, and should return the resource
        that needs to be managed.

    __exit__(exc_type, exc_value, traceback)
        Called when exiting the runtime context, handles exception information if any,
        and performs the necessary cleanup.

    See Also
    --------
    with statement : The `with` statement used for resource management in Python.

    Notes
    -----
    It's important that implementations of `__exit__` method should return `False` to
    propagate exceptions, unless the context manager is designed to suppress them. In
    such cases, it should return `True`.

    Examples
    --------
    """
    """
    >>> class FileContextManager(BaseContextManager):
    ...     def __enter__(self):
    ...         self.file = open('somefile.txt', 'w')
    ...         return self.file
    ...     def __exit__(self, exc_type, exc_value, traceback):
    ...         self.file.close()
    ...         # Handle exceptions or just pass
    ...
    >>> with FileContextManager() as file:
    ...     file.write('Hello, world!')
    ...
    >>> # somefile.txt will be closed after the with block
    """

    @abstractmethod
    def __enter__(self) -> Any:
        """
        Enters the runtime context and returns an object representing the context.

        The returned object is often the context manager instance itself, so it
        can include methods and attributes to interact with the managed resource.

        Returns
        -------
        Any
            An object representing the managed context, frequently the
            context manager instance itself.
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException],
                 traceback: Optional[Any]) -> Optional[bool]:
        """
        Exits the runtime context and performs any necessary cleanup actions.

        Parameters
        ----------
        exc_type : Type[BaseException] or None
            The type of exception raised (if any) during the context, otherwise `None`.
        exc_value : BaseException or None
            The exception instance raised (if any) during the context, otherwise `None`.
        traceback : Any or None
            The traceback object associated with the raised exception (if any), otherwise `None`.

        Returns
        -------
        Optional[bool]
            Should return `True` to suppress exceptions (if any) and `False` to
            propagate them. If no exception was raised, the return value is ignored.
        """
        pass

# Using the BaseContextManager requires creating a subclass and providing specific
# implementations for the __enter__ and __exit__ methods, tailored to the managed
# resource or the context-specific behavior.


class BaseProtocol(ABC):
    """
    Serves as an abstract foundational structure for defining interfaces
    specific to communication protocols. This base class enforces the methods
    to be implemented for encoding/decoding data and handling data transmission
    over an established communication channel.

    It is expected that concrete implementations will provide the necessary
    business logic for the actual encoding schemes, data transmission methods,
    and connection management appropriate to the chosen communication medium.

    Methods
    ----------
    encode(data)
        Converts data into a format suitable for transmission.

    decode(encoded_data)
        Converts data from the transmission format back to its original form.

    transmit(encoded_data)
        Initiates transfer of encoded data over the communication protocol's channel.

    send(data)
        Packets and sends data ensuring compliance with the underlying transmission protocol.

    receive()
        Listens for incoming data, decodes it, and returns the original message.

    connect()
        Initiates the communication channel, making it active and ready to use.

    disconnect()
        Properly closes and cleans up the established communication channel.

    See Also
    --------
    Abstract base class : A guide to Python's abstract base classes and how they work.

    Notes
    -----
    A concrete implementation of this abstract class must override all the
    abstract methods. It may also provide additional methods and attributes
    specific to the concrete protocol being implemented.

    """

    @abstractmethod
    def encode(self, data: Any) -> bytes:
        """
        Transforms given data into a sequence of bytes suitable for transmission.

        Parameters
        ----------
        data : Any
            The data to encode for transmission.

        Returns
        -------
        bytes
            The resulting encoded data as a byte sequence.
        """
        pass

    @abstractmethod
    def decode(self, encoded_data: bytes) -> Any:
        """
        Reverses the encoding, transforming the transmitted byte data back into its original form.

        Parameters
        ----------
        encoded_data : bytes
            The byte sequence representing encoded data.

        Returns
        -------
        Any
            The resulting decoded data in its original format.
        """
        pass

    @abstractmethod
    def transmit(self, encoded_data: bytes) -> None:
        """
        Sends encoded data over the communication protocol's channel.

        Parameters
        ----------
        encoded_data : bytes
            The byte sequence representing encoded data ready for transmission.
        """
        pass

    @abstractmethod
    def send(self, data: Any) -> None:
        """
        Sends data by encoding and then transmitting it.

        Parameters
        ----------
        data : Any
            The data to send over the communication channel, after encoding.
        """
        pass

    @abstractmethod
    def receive(self) -> Any:
        """
        Collects incoming data, decodes it, and returns the original message.

        Returns
        -------
        Any
            The decoded data received from the communication channel.
        """
        pass

    @abstractmethod
    def connect(self) -> None:
        """
        Opens and prepares the communication channel for data transmission.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Closes the established communication channel and performs clean-up operations.
        """
        pass


class BaseRuntime(ABC):
    """
    Describes the fundamental operations for runtime environments that manage
    the execution lifecycle of tasks. It provides a protocol for starting and
    stopping the runtime, executing tasks, and scheduling tasks based on triggers.

    Concrete subclasses should implement these methods to handle the specifics
    of task execution and scheduling within a given runtime environment, such as
    a containerized environment or a local execution context.

    Methods
    -------
    start()
        Initializes and starts the runtime environment, preparing it for task execution.

    stop()
        Shuts down the runtime environment, performing any necessary cleanup.

    execute(task, **kwargs)
        Executes a single task within the runtime environment, passing optional parameters.

    schedule(task, trigger)
        Schedules a task for execution based on a triggering event or condition.

    See Also
    --------
    BaseRuntime : A parent class defining the methods used by all runtime classes.

    Notes
    -----
    A `BaseRuntime` is designed to provide an interface for task execution and management
    without tying the implementation to any particular execution model or technology,
    allowing for a variety of backends ranging from local processing to distributed computing.

    Examples
    --------
    """
    """
    >>> class MyRuntime(BaseRuntime):
    ...     def start(self):
    ...         print("Runtime starting")
    ...
    ...     def stop(self):
    ...         print("Runtime stopping")
    ...
    ...     def execute(self, task, **kwargs):
    ...         print(f"Executing {task} with {kwargs}")
    ...
    ...     def schedule(self, task, trigger):
    ...         print(f"Scheduling {task} on {trigger}")
    >>> runtime = MyRuntime()
    >>> runtime.start()
    Runtime starting
    >>> runtime.execute('Task1', param='value')
    Executing Task1 with {'param': 'value'}
    >>> runtime.stop()
    Runtime stopping
    """

    @abstractmethod
    def start(self) -> None:
        """
        Performs any necessary initialization and starts the runtime environment,
        making it ready for executing tasks.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Cleans up any resources and stops the runtime environment, ensuring that
        all tasks are properly shut down and that the environment is left in a
        clean state.
        """
        pass

    @abstractmethod
    def execute(self, task: Callable[..., Any], **kwargs: Any) -> None:
        """
        Runs a given task within the runtime environment, providing any additional
        keyword arguments needed by the task.

        Parameters
        ----------
        task : Callable[..., Any]
            The task to be executed.
        kwargs : dict
            A dictionary of keyword arguments for the task execution.
        """
        pass

    @abstractmethod
    def schedule(self, task: Callable[..., Any], trigger: Any) -> None:
        """
        Schedules a task for execution when a specific trigger occurs within the
        runtime environment.

        Parameters
        ----------
        task : Callable[..., Any]
            The task to be scheduled.
        trigger : Any
            The event or condition that triggers the task execution.
        """
        pass


class TokenSpace(ABC):
    """
    Defines a generic interface for managing a space of tokens within a
    given context, such as a simulation or a data flow control system. It provides
    methods to add, retrieve, and inspect tokens within the space, where a token
    can represent anything from a data item to a task or computational unit.

    Concrete implementations of this abstract class will handle specific details
    of token storage, accessibility, and management according to their purposes.

    Methods
    -------
    push(item)
        Inserts a token into the token space.

    pop()
        Retrieves and removes a token from the token space, following a defined removal strategy.

    peek()
        Inspects the next token in the space without removing it.

    See Also
    --------
    TokenSpace : A parent class representing a conceptual space for holding tokens.

    Notes
    -----
    The semantics and behavior of the token space, such as whether it operates as a
    stack, queue, or other structure, are determined by the concrete subclass
    implementations.
    """
    """
    Examples
    --------
    >>> class MyTokenSpace(TokenSpace):
    ...     def push(self, item):
    ...         print(f"Inserted {item} into space")
    ...
    ...     def pop(self):
    ...         print(f"Removed item from space")
    ...
    ...     def peek(self):
    ...         print("Inspecting the next item")
    >>> space = MyTokenSpace()
    >>> space.push('Token1')
    Inserted Token1 into space
    >>> space.peek()
    Inspecting the next item
    >>> space.pop()
    Removed item from space
    """

    @abstractmethod
    def push(self, item: Any) -> None:
        """
        Adds a token to the space for later retrieval.

        Parameters
        ----------
        item : Any
            The token to be added to the space.
        """
        pass

    @abstractmethod
    def pop(self) -> Any:
        """
        Removes and returns a token from the space, adhering to the space's removal policy.

        Returns
        -------
        Any
            The next token to be retrieved and removed from the space.
        """
        pass

    @abstractmethod
    def peek(self) -> Any:
        """
        Allows inspection of the next token to be retrieved from the space without
        actually removing it from the space.

        Returns
        -------
        Any
            The next token available in the space, without removing it.
        """
        pass

# Concrete implementations of BaseRuntime and TokenSpace must be provided to utilize
# these abstract classes. They form the basis of specialized runtime environments and
# token management systems within an application.
