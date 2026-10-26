"""Running registered methods from the table browser: a form for their arguments and showing results."""

import ast
from inspect import Parameter

import pandas as pd

from .viewers import open_value

_POSITIONAL = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)


def _default_text(parameter):
    return "" if parameter.default is parameter.empty else repr(parameter.default)


def _call(text):
    """Literal positional and keyword arguments written as in a call, such as '"key", 10, c=3'."""
    call = ast.parse(f"f({text})", mode="eval").body
    return ([ast.literal_eval(arg) for arg in call.args],
            {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords})


def _literal(parameter, text):
    try:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            return _call(text)[parameter.kind is parameter.VAR_KEYWORD]
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        raise ValueError(f"{parameter.name} = {text} is not a literal; are quotes missing?") from None


def call_arguments(parameters, texts):
    """Args, kwargs and call text from one text per parameter; empty or unchanged defaults are left out.

    Arguments go by position up to the last one given positionally, with defaults filling the gaps.
    """
    given = {parameter.name: text for parameter, text in zip(parameters, texts)
             if text and text != _default_text(parameter)}
    last = max((i for i, parameter in enumerate(parameters) if parameter.name in given
                and parameter.kind in (*_POSITIONAL, Parameter.VAR_POSITIONAL)), default=-1)
    args, kwargs, parts = [], {}, []
    for i, parameter in enumerate(parameters):
        text = given.get(parameter.name)
        if parameter.kind is parameter.VAR_POSITIONAL:
            if text:
                args += _literal(parameter, text)
                parts.append(text)
        elif parameter.kind is parameter.VAR_KEYWORD:
            if text:
                kwargs.update(_literal(parameter, text))
                parts.append(text)
        elif i <= last and parameter.kind in _POSITIONAL:
            if text is None and parameter.default is parameter.empty:
                raise ValueError(f"{parameter.name} is required before later positional arguments")
            args.append(parameter.default if text is None else _literal(parameter, text))
            parts.append(text or _default_text(parameter))
        elif text:
            kwargs[parameter.name] = _literal(parameter, text)
            parts.append(f"{parameter.name}={text}")
    return args, kwargs, ", ".join(parts)


def ask_arguments(master, name, parameters, filled, submit):
    """Show one box per parameter, prefilled with the last input or the default; a checkbox for a bool default.

    Run calls submit(args, kwargs, text). Positional-only parameters are unlabelled since their names
    are not part of a call.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    form = tk.Toplevel(master)
    form.title(name)
    form.transient(master)
    values = []
    for line, parameter in enumerate(parameters):
        if parameter.kind is not parameter.POSITIONAL_ONLY:
            prefix = {parameter.VAR_POSITIONAL: "*", parameter.VAR_KEYWORD: "**"}.get(parameter.kind, "")
            ttk.Label(form, text=prefix + parameter.name).grid(row=line, column=0, sticky="w", padx=4, pady=2)
        value = tk.StringVar(form, filled.get(parameter.name, _default_text(parameter)))
        if isinstance(parameter.default, bool):  # Reads True or False, like the literal typed in a box.
            widget = ttk.Checkbutton(form, variable=value, onvalue="True", offvalue="False")
        else:
            widget = ttk.Entry(form, width=40, textvariable=value)
        widget.grid(row=line, column=1, sticky="ew", padx=4, pady=2)
        if not values:
            widget.focus_set()
        values.append(value)
    form.columnconfigure(1, weight=1)

    def run(event=None):
        texts = [value.get().strip() for value in values]
        filled.update(zip((parameter.name for parameter in parameters), texts))
        try:
            arguments = call_arguments(parameters, texts)
        except ValueError as exc:
            messagebox.showerror(name, str(exc), parent=form)
            return
        form.destroy()
        submit(*arguments)

    ttk.Button(form, text="Run", command=run).grid(row=len(parameters), column=1, sticky="e", padx=4, pady=4)
    form.bind("<Return>", run)
    form.bind("<Escape>", lambda event: form.destroy())


def run_and_show(master, title, call):
    """Show a call's result with its preferred viewer; None opens no window.

    A Series, one value per selected row, shows as a group of them named by row label.
    """
    result = call()
    if isinstance(result, pd.Series):  # One value per row; browse them by row label rather than stacked.
        result = None if all(value is None for value in result) else {
            str(label): value for label, value in result.items()}
    if result is not None:
        open_value(master, title, result)
