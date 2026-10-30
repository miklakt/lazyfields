import os
import pickle
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from readers import pickled

_MARKER_CALLS = []


def _mark(value):
    _MARKER_CALLS.append(value)
    return value


class _MaliciousPayload:
    """The textbook pickle exploit: __reduce__ names a call that runs arbitrary code on unpickling."""

    def __reduce__(self):
        return os.system, ("touch /tmp/lazyfields-pwned",)


class _ExfiltratingPayload:
    """A payload that doesn't touch the filesystem, just proves the callable itself gets invoked."""

    def __reduce__(self):
        return (_mark, ("ran",))


class _NaiveInit:
    """A plain, hand-written class of the kind someone might reach for instead of @dataclass."""

    def __init__(self, temperature):
        _MARKER_CALLS.append(f"init ran with temperature={temperature!r}")
        self.temperature = temperature


@dataclass
class _WithPostInit:
    """A dataclass whose generated __init__ is harmless, but which calls real code right after."""

    x: int

    def __post_init__(self):
        _MARKER_CALLS.append(f"post_init ran with x={self.x!r}")


class _WithCustomSetattr:
    """BUILD restores slots with real setattr() calls, not direct slot assignment - this is reachable."""

    __slots__ = ("x",)

    def __setattr__(self, name, value):
        _MARKER_CALLS.append(f"setattr({name!r}, {value!r})")
        object.__setattr__(self, name, value)


class _EvilMeta(type):
    def __call__(cls, *args, **kwargs):
        _MARKER_CALLS.append(f"metaclass __call__ with args={args!r}")
        return super().__call__(*args, **kwargs)


class _WithCustomMetaclass(metaclass=_EvilMeta):
    """A direct REDUCE call to the class runs type(cls).__call__, not just __new__/__init__."""


@dataclass(frozen=True, slots=True)
class _Point:
    """A trusted type that is pure data: no methods beyond the ones dataclass generates."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class _Logger:
    """The same shape, but with one ordinary-looking method - which is exactly the hijack surface."""

    tag: str

    def leak(self, message):
        Path(tempfile.gettempdir(), "lazyfields-leaked").write_text(message)


class MaliciousPickleTests(unittest.TestCase):
    def setUp(self):
        _MARKER_CALLS.clear()
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "value.pkl"
        Path("/tmp/lazyfields-pwned").unlink(missing_ok=True)

    def tearDown(self):
        self.directory.cleanup()
        Path("/tmp/lazyfields-pwned").unlink(missing_ok=True)

    def dump(self, value, trust=None):
        self.path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        return pickled.pickle_reader(trust=trust).load(self.path)

    def test_os_system_call_never_runs(self):
        loaded = self.dump({"payload": _MaliciousPayload()})
        self.assertFalse(Path("/tmp/lazyfields-pwned").exists())
        self.assertEqual(loaded["payload"].type, "posix.system")

    def test_arbitrary_call_never_runs(self):
        loaded = self.dump(_ExfiltratingPayload())
        self.assertEqual(_MARKER_CALLS, [])
        self.assertEqual(loaded.type, f"{__name__}._mark")

    def test_ordinary_pickle_load_would_have_run_it(self):
        self.path.write_bytes(pickle.dumps(_ExfiltratingPayload(), protocol=pickle.HIGHEST_PROTOCOL))
        with self.path.open("rb") as file:
            pickle.load(file)  # Demonstrates the payload is real, not just inert in our scan.
        self.assertEqual(_MARKER_CALLS, ["ran"])

    def test_explicit_trust_is_required_to_run_it(self):
        loaded = self.dump(_ExfiltratingPayload(), trust=True)
        self.assertEqual(loaded, "ran")
        self.assertEqual(_MARKER_CALLS, ["ran"])


def _build_from_helper(helper):
    return helper.compute()


class _Helper:
    def compute(self):
        return "computed"


class ClearErrorForUncoveredDependencyTests(unittest.TestCase):
    """A trusted function can itself depend on a global trust didn't cover: that global becomes an
    inert record, the trusted function gets handed the record instead of what it expected, and fails
    deep inside its own code with no mention of trust at all. The reader should say so anyway.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "value.pkl"

    def tearDown(self):
        self.directory.cleanup()

    def test_failure_names_the_uncovered_global(self):
        class _Gadget:
            def __reduce__(self):
                return (_build_from_helper, (_Helper(),))

        self.path.write_bytes(pickle.dumps(_Gadget(), protocol=pickle.HIGHEST_PROTOCOL))
        reader = pickled.pickle_reader(trust=[(__name__, "_build_from_helper")])
        with self.assertRaisesRegex(pickle.UnpicklingError, f"{__name__}\\._Helper"):
            reader.load(self.path)

    def test_original_failure_stays_available_as_the_cause(self):
        class _Gadget:
            def __reduce__(self):
                return (_build_from_helper, (_Helper(),))

        self.path.write_bytes(pickle.dumps(_Gadget(), protocol=pickle.HIGHEST_PROTOCOL))
        reader = pickled.pickle_reader(trust=[(__name__, "_build_from_helper")])
        with self.assertRaises(pickle.UnpicklingError) as caught:
            reader.load(self.path)
        self.assertIsInstance(caught.exception.__cause__, AttributeError)


class TrustingALibraryPrefixTests(unittest.TestCase):
    """Trusting a library prefix is not just "pandas DataFrames become real": it lets the pickle call
    *any* bound method of *any* object of a class under that prefix, with attacker-chosen arguments.
    matplotlib ships one such gadget: resolving a "module://<name>" backend imports <name> for real, so
    a pickle can make the trusting reader import an arbitrary module, running its top-level code.
    """

    def setUp(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        self.directory = tempfile.TemporaryDirectory()
        self.gadget_dir = Path(self.directory.name) / "gadget"
        self.gadget_dir.mkdir()
        self.marker = Path(self.directory.name) / "ran-attacker-module"
        (self.gadget_dir / "evil_backend.py").write_text(
            f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('pwned')\n")
        sys.path.insert(0, str(self.gadget_dir))
        self.path = Path(self.directory.name) / "value.pkl"

    def tearDown(self):
        sys.path.remove(str(self.gadget_dir))
        sys.modules.pop("evil_backend", None)
        self.directory.cleanup()

    def dump(self, value, trust=None):
        self.path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        return pickled.pickle_reader(trust=trust).load(self.path)

    def _payload(self):
        from matplotlib.backends.registry import backend_registry

        class _Gadget:
            def __reduce__(self):
                # Pickles a bound method as (getattr, (backend_registry, "load_backend_module")),
                # exactly how pickle would serialize it for any real object.
                return getattr(backend_registry, "load_backend_module"), ("module://evil_backend",)

        return _Gadget()

    def test_trusting_matplotlib_lets_the_pickle_import_an_arbitrary_module(self):
        self.dump(self._payload(), trust=["matplotlib"])
        self.assertTrue(self.marker.exists(), "trusting matplotlib should have run evil_backend's top-level code")

    def test_default_reader_does_not_import_the_module(self):
        loaded = self.dump(self._payload())
        self.assertFalse(self.marker.exists())
        self.assertEqual(loaded.type, "builtins.getattr()")


class TrustingYourOwnDataclassTests(unittest.TestCase):
    """Trusting an exact class is only as safe as that class has nothing callable worth hijacking:
    trust lets a pickle call any of its public methods with attacker-chosen arguments via a
    getattr() + REDUCE chain. A data-only dataclass has no such method to find, so the same attack
    just fails to resolve, instead of running.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "value.pkl"

    def tearDown(self):
        self.directory.cleanup()

    def dump(self, value, trust):
        self.path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        return pickled.pickle_reader(trust=trust).load(self.path)

    def test_trusted_data_only_dataclass_round_trips(self):
        loaded = self.dump(_Point(1.0, 2.0), trust=_Point)
        self.assertEqual(loaded, _Point(1.0, 2.0))

    def test_hijack_attempt_on_a_data_only_class_has_nothing_to_call(self):
        point = _Point(1.0, 2.0)

        class _Gadget:
            def __reduce__(self):
                # Named, not called: the unpickler itself must resolve this attribute, not Python.
                return getattr, (point, "anything_that_sounds_callable")

        with self.assertRaises(AttributeError):
            self.dump(_Gadget(), trust=_Point)

    def test_trusting_a_class_with_a_method_is_refused_outright(self):
        """A method is exactly the hijack surface a getattr() + REDUCE chain needs - trust refuses it
        before any pickle is even read, rather than leaving it to the caller to notice.
        """
        with self.assertRaisesRegex(ValueError, "leak"):
            pickled.pickle_reader(trust=_Logger)

    def test_trusting_a_hand_written_init_is_refused_outright(self):
        """NEWOBJ+BUILD never calls __init__, but a crafted pickle can REDUCE-call the class directly
        instead, running a hand-written __init__ for real - so trust refuses the class outright.
        """
        with self.assertRaisesRegex(ValueError, "__init__"):
            pickled.pickle_reader(trust=_NaiveInit)

    def test_trusting_a_post_init_is_refused_outright(self):
        """__post_init__ is only ever the class author's own code, reached the same way: a direct
        REDUCE call to the class runs the (otherwise harmless) generated __init__, which calls it."""
        with self.assertRaisesRegex(ValueError, "__post_init__"):
            pickled.pickle_reader(trust=_WithPostInit)

    def test_trusting_a_custom_setattr_is_refused_outright(self):
        """BUILD restores a slotted instance's state with real setattr() calls, not direct slot writes -
        so a custom __setattr__ is a load-time hook too, not just a read-side getattr() concern."""
        with self.assertRaisesRegex(ValueError, "__setattr__"):
            pickled.pickle_reader(trust=_WithCustomSetattr)

    def test_trusting_a_custom_metaclass_is_refused_outright(self):
        """A direct REDUCE call to the class invokes type(cls).__call__, not just cls.__new__/__init__ -
        a custom metaclass's own code runs regardless of how innocuous the class body itself looks."""
        with self.assertRaisesRegex(ValueError, "metaclass"):
            pickled.pickle_reader(trust=_WithCustomMetaclass)


class DataOnlyTests(unittest.TestCase):
    """pickled.data_only(cls) catches, ahead of time, each hijack shape the other test classes exploit by hand."""

    def test_accepts_a_plain_dataclass(self):
        self.assertTrue(pickled.data_only(_Point))

    def test_accepts_a_frozen_dataclass(self):
        """Frozen generates its own __setattr__ override - this is the one case it must still accept."""

        @dataclass(frozen=True)
        class Frozen:
            x: float

        self.assertTrue(pickled.data_only(Frozen))

    def test_accepts_a_plain_dataclass_without_slots(self):
        @dataclass
        class Plain:
            x: float
            y: float

        self.assertTrue(pickled.data_only(Plain))

    def test_rejects_a_method(self):
        self.assertFalse(pickled.data_only(_Logger))

    def test_rejects_a_custom_setattr(self):
        self.assertFalse(pickled.data_only(_WithCustomSetattr))

    def test_rejects_a_custom_metaclass(self):
        self.assertFalse(pickled.data_only(_WithCustomMetaclass))

    def test_rejects_a_hand_written_init(self):
        self.assertFalse(pickled.data_only(_NaiveInit))

    def test_rejects_post_init(self):
        self.assertFalse(pickled.data_only(_WithPostInit))

    def test_rejects_a_property(self):
        @dataclass(slots=True)
        class WithProperty:
            x: float

            @property
            def doubled(self):
                return self.x * 2

        self.assertFalse(pickled.data_only(WithProperty))

    def test_rejects_an_attribute_hijacking_getattr(self):
        @dataclass(slots=True)
        class WithGetattr:
            x: float

            def __getattr__(self, name):
                return "hijacked"

        self.assertFalse(pickled.data_only(WithGetattr))

    def test_rejects_a_custom_new(self):
        @dataclass(slots=True)
        class WithNew:
            x: float

            def __new__(cls, *args, **kwargs):
                return object.__new__(cls)

        self.assertFalse(pickled.data_only(WithNew))

    def test_rejects_a_base_class_with_a_method(self):
        class Base:
            def danger(self):
                pass

        @dataclass(slots=True)
        class Inherits(Base):
            x: float

        self.assertFalse(pickled.data_only(Inherits))

    def test_a_data_only_dataclass_actually_survives_the_hijack_attempt(self):
        """The check and the real attack agree: what it accepts really has no callable to find."""
        point = _Point(1.0, 2.0)
        self.assertTrue(pickled.data_only(_Point))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.pkl"

            class _Gadget:
                def __reduce__(self):
                    return getattr, (point, "anything_that_sounds_callable")

            path.write_bytes(pickle.dumps(_Gadget(), protocol=pickle.HIGHEST_PROTOCOL))
            with self.assertRaises(AttributeError):
                pickled.pickle_reader(trust=_Point).load(path)


if __name__ == "__main__":
    unittest.main()
