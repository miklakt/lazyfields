"""Utilities for defining source-backed store methods."""

import ast
import builtins
import inspect
import linecache
import symtable
import textwrap
from typing import Any, Callable
from uuid import uuid4


def bind_method(**mapping: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bind function parameters to row expressions; leave other parameters free.

    Example: @bind_method(a='_["a"]', b='_.store["b"]').
    Bindings may call already registered methods, e.g. b='_.store.length()';
    the call is evaluated for each row when the adapted method runs.
    At least one mapping is required, referring to exactly one unknown row name.
    Mapping expressions are used unchanged. Positional-only parameters are unsupported.
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
        for name, expression in mapping.items():
            if name not in parameters:
                raise ValueError(f"Unknown parameter {name!r}.")
            if not isinstance(expression, str):
                raise TypeError("Field bindings must be Python expression strings.")

        namespace = dict(method.__globals__)
        namespace.update(inspect.getclosurevars(method).nonlocals)

        def global_names(table):
            names = {symbol.get_name() for symbol in table.get_symbols()
                     if symbol.is_referenced() and symbol.is_global()}
            for child in table.get_children():
                names.update(global_names(child))
            return names

        preamble = "\n".join(f"{name} = ({expression})" for name, expression in mapping.items())
        assignments = ast.parse(preamble).body
        names = global_names(symtable.symtable(preamble, "<bindings>", "exec"))
        aliases = names - namespace.keys() - vars(builtins).keys() - parameters
        if len(aliases) != 1:
            raise ValueError(f"Use a single row name in bindings; unknown names: {sorted(aliases)}")
        row_name = aliases.pop()
        if any(isinstance(item, (ast.Name, ast.arg)) and
               (item.id if isinstance(item, ast.Name) else item.arg) == row_name
               for item in ast.walk(node)):
            raise ValueError(f"Row name {row_name!r} conflicts with the original function.")

        # Remove bound parameters from right to left, along with any defaults.
        for arguments, defaults in ((node.args.args, node.args.defaults),
                                    (node.args.kwonlyargs, node.args.kw_defaults)):
            for index in reversed(range(len(arguments))):
                if arguments[index].arg in mapping:
                    default_index = index - len(arguments) + len(defaults)
                    del arguments[index]
                    if default_index >= 0:
                        del defaults[default_index]
        node.args.args.insert(0, ast.arg(arg=row_name))

        # Insert assignments after the docstring, if present.
        offset = int(ast.get_docstring(node) is not None)
        node.body[offset:offset] = assignments
        node.decorator_list = []
        generated_source = ast.unparse(ast.fix_missing_locations(node)) + "\n"
        # Expose generated source to inspect.getsource(), just like a source file.
        filename = f"<bind_method-{uuid4().hex}>"
        linecache.cache[filename] = (
            len(generated_source), None, generated_source.splitlines(keepends=True), filename
        )
        exec(compile(generated_source, filename, "exec"), namespace)
        return namespace[node.name]

    return decorate
