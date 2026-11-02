"""A Tk form for a callable's parameters: prefilled with defaults, parsed back as literals.

Shared by registered store methods (browser/table.py) and filesystem method keyword arguments
(browser/methods.py).
"""

import ast
import re
from inspect import Parameter, cleandoc

_POSITIONAL = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
_PARAMETERS_SECTION = re.compile(r"(Args|Arguments|Parameters|Params|Keyword Args|Keyword Arguments):?")
_SECTION = re.compile(r"(Returns?|Yields?|Raises|Examples?|Notes?|See Also|References|Warnings?|Attributes):?")


def parameter_docs(doc):
    """Each parameter's entry in a Google or numpy style section, or Sphinx :param: field, of doc, by name."""
    lines, docs = cleandoc(doc or "").splitlines(), {}
    for start, line in enumerate(lines):
        if match := re.match(r":param\s+(?:([^:]*?)\s+)?(\w+):\s*(.*)", line.strip()):
            kind, name, text = match.groups()  # Shown as "name (kind): text", like a Google entry.
            entry = docs[name] = [f"{name} ({kind}): {text}" if kind else f"{name}: {text}"]
            for more in lines[start + 1:]:  # Continuation lines are indented under it.
                if not more[:1].isspace() or not more.strip():
                    break
                entry.append(more.strip())
            continue
        if not _PARAMETERS_SECTION.fullmatch(line.strip()):
            continue
        indent, name = None, None
        for line, after in zip(lines[start + 1:], [*lines[start + 2:], ""]):
            depth = len(line) - len(line.lstrip())
            if not line.strip():
                continue
            if set(line.strip()) == {"-"} and name is None:  # The numpy section's underline.
                continue
            if indent is None:
                indent = depth
            next_section = _SECTION.fullmatch(line.strip()) or set(after.strip()) == {"-"}
            if depth < indent or depth == indent and next_section:
                break
            if depth == indent:
                match = re.match(r"\**(\w+)", line.strip())
                name = match and match.group(1)
                if name:
                    docs[name] = [line.strip()]
            elif name:
                docs[name].append(line.strip())
    return {name: "\n".join(text) for name, text in docs.items()}


def _dialog(window):
    """Ask the window manager for a dialog's frame: fixed size, so no maximize, and no minimize where it
    honours the dialog type (GNOME's X11 frames still draw an inert minimize button)."""
    import tkinter as tk

    window.resizable(False, False)  # Window managers offer maximize only to resizable windows.
    for option in ("-type", "dialog"), ("-toolwindow", True):  # X11, then Windows.
        try:
            window.attributes(*option)
            return
        except tk.TclError:
            pass


def _tooltip(widget, text):
    """Show text in a tooltip while the pointer is over widget; return widget."""
    import tkinter as tk

    tip = {}

    def show(event):
        window = tip["window"] = tk.Toplevel(widget)
        window.wm_overrideredirect(True)  # Borderless, and never takes focus.
        window.geometry(f"+{event.x_root + 12}+{event.y_root + 12}")
        tk.Label(window, text=text, wraplength=480, justify="left", padx=6, pady=4, background="#ffffe0",
                 foreground="black", relief="solid", borderwidth=1).pack()

    def hide(event):
        if "window" in tip:
            tip.pop("window").destroy()

    widget.configure(cursor="question_arrow")
    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)
    widget.bind("<Destroy>", hide)
    return widget


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


def ask_arguments(master, name, parameters, filled, submit, doc=None):
    """Show one box per parameter, prefilled with the last input or the default; a checkbox for a bool default.

    Run calls submit(args, kwargs, text). Positional-only parameters are unlabelled since their names
    are not part of a call. doc's first line heads the form, showing all of doc on hover; hovering a
    parameter's name shows its entry in doc.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    form = tk.Toplevel(master)
    _dialog(form)
    form.title(name)
    form.transient(master)
    doc, hints = cleandoc(doc or ""), parameter_docs(doc)
    summary = doc.splitlines()[0] if doc else ""
    if re.match(r":\w", summary) or _PARAMETERS_SECTION.fullmatch(summary) or _SECTION.fullmatch(summary):
        summary = ""  # Only fields or sections, already shown on the names they describe.
    if summary:
        _tooltip(ttk.Label(form, text=summary, wraplength=400, justify="left"), doc).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 2))
    values, start = [], 1 if summary else 0
    for line, parameter in enumerate(parameters, start):
        if parameter.kind is not parameter.POSITIONAL_ONLY:
            prefix = {parameter.VAR_POSITIONAL: "*", parameter.VAR_KEYWORD: "**"}.get(parameter.kind, "")
            label = ttk.Label(form, text=prefix + parameter.name)
            if parameter.name in hints:
                _tooltip(label, hints[parameter.name])
            label.grid(row=line, column=0, sticky="w", padx=4, pady=2)
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

    ttk.Button(form, text="Run", command=run).grid(row=start + len(parameters), column=1, sticky="e",
                                                   padx=4, pady=4)
    form.bind("<Return>", run)
    form.bind("<Escape>", lambda event: form.destroy())
