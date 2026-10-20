"""Utilities for defining source-backed store methods."""

import ast
import inspect
import linecache
import textwrap
from typing import Any, Callable
from uuid import uuid4


def bind_method(**mapping: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bind parameters to row-relative expressions; leave other parameters free.

    Example: @bind_method(a="a", b='.store["b"]').
    Plain strings are column keys. Strings starting with '[' or '.' extend the
    row reference, e.g. '["a"][1:5]' for slicing.
    Bindings may call registered methods, e.g. b='.store.length("a", 5)';
    the call is evaluated for each row when the adapted method runs.
    At least one binding is required. Positional-only parameters are unsupported.
    __row is reserved for the generated first parameter.
    Expressions are Python code and must come from a trusted source.
    """
    if not mapping:
        raise ValueError("At least one field mapping is required.")

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
        if not isinstance(node, ast.FunctionDef):
            raise TypeError("bind_method requires a regular function definition.")
        if node.args.posonlyargs:
            raise TypeError("bind_method does not support positional-only parameters.")
        parameters = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
        bindings = {}
        for name, expression in mapping.items():
            if name not in parameters:
                raise ValueError(f"Unknown parameter {name!r}.")
            if not isinstance(expression, str):
                raise TypeError("Field bindings must be Python expression strings.")
            suffix = expression.strip()
            bindings[name] = "__row" + suffix if suffix.startswith(("[", ".")) else f"__row[{expression!r}]"

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
        node.body[offset:offset] = ast.parse(preamble).body
        node.decorator_list = []
        generated_source = ast.unparse(ast.fix_missing_locations(node)) + "\n"
        # Expose generated source to inspect.getsource(), just like a source file.
        filename = f"<bind_method-{uuid4().hex}>"
        linecache.cache[filename] = (
            len(generated_source), None, generated_source.splitlines(keepends=True), filename
        )
        namespace = dict(method.__globals__)
        namespace.update(inspect.getclosurevars(method).nonlocals)
        exec(compile(generated_source, filename, "exec"), namespace)
        return namespace[node.name]

    return decorate
