"""Utilities for defining source-backed store methods."""

import ast
import inspect
import linecache
import textwrap
from typing import Any, Callable
from uuid import uuid4


def _method_imports(method: Callable[..., Any]) -> list[ast.stmt]:
    """Recover imports for referenced modules and module-exported objects."""
    references = inspect.getclosurevars(method)
    imports = []
    for alias, value in {**references.globals, **references.nonlocals}.items():
        module = value if inspect.ismodule(value) else inspect.getmodule(value)
        if module is None or module.__name__ == "__main__":
            continue
        if inspect.ismodule(value):
            statement = f"import {module.__name__} as {alias}"
        else:
            name = getattr(value, "__name__", "")
            if not name or getattr(module, name, None) is not value:
                continue
            statement = f"from {module.__name__} import {name} as {alias}"
        imports.extend(ast.parse(statement).body)
    return imports


def bind_method(
    function: Callable[..., Any], mode: str = "source", /, **mapping: str
) -> Callable[..., Any]:
    """Bind parameters to columns ("a") or row expressions ('.store["b"]').

    "source" stores the function body and recoverable imports.
    "wrap" stores a wrapper; the original module must stay available.
    The mode argument is positional-only; all keywords are field mappings.
    Bindings are trusted Python expressions. Globals/closures are not captured.
    """
    if mode not in ("source", "wrap"):
        raise ValueError("mode must be 'source' or 'wrap'.")
    if not mapping:
        raise ValueError("At least one field mapping is required.")
    bindings = {}
    for name, expression in mapping.items():
        if not isinstance(expression, str):
            raise TypeError("Field bindings must be Python expression strings.")
        suffix = expression.strip()
        bindings[name] = "__row" + suffix if suffix.startswith(("[", ".")) else f"__row[{expression!r}]"
    parameters = inspect.signature(function).parameters
    allowed = {name for name, p in parameters.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    if not mapping.keys() <= allowed:
        raise ValueError("Unknown or variadic parameter in bindings.")
    if "__row" in parameters:
        raise ValueError("Reserved name '__row' conflicts with the original function.")
    name = function.__name__

    if mode == "wrap":
        module = inspect.getmodule(function)
        if module is None or module.__name__ == "__main__" or getattr(module, name, None) is not function:
            raise ValueError("The function must be available by name in an importable module.")
        fields = ", ".join(f"{field!r}: ({expression})" for field, expression in bindings.items())
        source = f'''def {name}(__row, *args, **kwargs):
    from {module.__name__} import {name} as __method
    from inspect import signature, BoundArguments
    fields = {{{fields}}}
    sig = signature(__method)
    free = sig.replace(parameters=[p for name, p in sig.parameters.items() if name not in fields])
    supplied = free.bind(*args, **kwargs)
    supplied.apply_defaults()
    bound = BoundArguments(sig, {{**supplied.arguments, **fields}})
    return __method(*bound.args, **bound.kwargs)
'''
        namespace = {}
    else:
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        if not isinstance(node, ast.FunctionDef):
            raise TypeError("bind_method requires a regular function definition.")
        if node.args.posonlyargs:
            raise TypeError("bind_method does not support positional-only parameters.")
        if any(isinstance(item, (ast.Name, ast.arg)) and
               (item.id if isinstance(item, ast.Name) else item.arg) == "__row"
               for item in ast.walk(node)):
            raise ValueError("Reserved name '__row' conflicts with the original function.")

        # Remove bound parameters from right to left, along with any defaults.
        for arguments, defaults in ((node.args.args, node.args.defaults),
                                    (node.args.kwonlyargs, node.args.kw_defaults)):
            for index in reversed(range(len(arguments))):
                if arguments[index].arg in mapping:
                    default_index = index - len(arguments) + len(defaults)
                    del arguments[index]
                    if default_index >= 0:
                        del defaults[default_index]
        node.args.args.insert(0, ast.arg(arg="__row"))

        preamble = "\n".join(f"{name} = ({expression})" for name, expression in bindings.items())
        # Insert assignments after the docstring, if present.
        offset = int(ast.get_docstring(node) is not None)
        node.body[offset:offset] = _method_imports(function) + ast.parse(preamble).body
        node.decorator_list = []
        source = ast.unparse(ast.fix_missing_locations(node)) + "\n"
        namespace = dict(function.__globals__)
        namespace.update(inspect.getclosurevars(function).nonlocals)
    # Keep generated source inspectable for register_method.
    filename = f"<lazyfields-method-{uuid4().hex}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), namespace)
    return namespace[name]
