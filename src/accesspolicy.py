from typing import Any, Dict, Optional, Union, Callable
from pathlib import Path
import ast
import inspect
import logging
import asyncio
from types import SimpleNamespace
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

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
        # Always allow access to root namespaces defined in patterns
        root_namespace = namespace.split('.')[0]
        for pattern in self.namespace_patterns:
            pattern_root = pattern.split('.')[0]
            if root_namespace == pattern_root:
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
        
        # Add data namespace
        base_namespace['data'] = self.root.get_child('data')._content
        
        return base_namespace

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

async def setup_runtime():
    runtime = RuntimeManager()
    
    # Initialize data namespace
    data_ns = runtime.root.add_child("data")
    sales_ns = data_ns.add_child("sales")
    sales_ns._content.total = 1000  # Example value
    
    # Create access policies
    read_policy = AccessPolicy(
        level=AccessLevel.READ,
        namespace_patterns=["data.", "public."],
        allowed_operations=["read"]
    )
    
    exec_policy = AccessPolicy(
        level=AccessLevel.EXECUTE,
        namespace_patterns=["data.", "public.", "functions."],
        allowed_operations=["read", "execute"]
    )
    
    runtime.register_user("user1", read_policy)
    runtime.register_user("power_user", exec_policy)
    
    return runtime

def main():
    # Setup runtime
    runtime = asyncio.run(setup_runtime())

    # Execute queries
    result = asyncio.run(runtime.execute_query("user1", "data.sales.total"))
    print(f"Result: {result}")

if __name__ == "__main__":
    main()