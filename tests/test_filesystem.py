"""One namespace for file registration, edits, clones and lazy source reads."""

import importlib.util
import json
from pathlib import Path
import pickle
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from lazyfields.filesystem import FileSystem, SourceLocator
from lazyfields.readers import NodeInfo, Reader


EXAMPLE = Path(__file__).resolve().parents[1] / "example" / "filesystem"


class NamespaceTests(unittest.TestCase):
    def setUp(self):
        root = EXAMPLE / "data"
        self.fs = FileSystem(
            sorted(path.relative_to(root) for path in root.rglob("*.json")), relative_to=root,
        )
        self.view = self.fs.clone()
        self.alpha = "partitions.json/partitions/alpha"

    def test_origin_survives_edits_but_locate_tracks_current_children(self):
        source = self.fs.clone("metadata/alpha.json")
        origin = source.node().origin
        self.assertEqual(origin, SourceLocator(self.fs.relative_to / "metadata/alpha.json"))
        dataset = source.clone("coefficients")
        self.assertEqual(dataset.node().origin.infile_path, "coefficients")
        for name in source.listdir():
            source.unmount(name)
        self.assertTrue(source.in_file)
        self.assertEqual(source.node().origin, origin)
        self.assertEqual(source.locate(), [])
        empty_view = FileSystem()
        empty_view.mount("empty_alias", source.node())
        self.assertTrue(empty_view.node("empty_alias").is_logical)
        self.assertEqual(empty_view.node("empty_alias").origin, origin)
        clone = source.clone()
        clone.mount("nested", dataset.node())
        self.assertEqual(clone.node().origin, origin)
        self.assertEqual(clone.locate(), dataset.locate())
        self.assertEqual(source.read(), {})
        self.assertEqual(self.fs.node("metadata/alpha.json").origin, origin)
        self.assertIsNone(self.fs.node("metadata").origin)
        view = FileSystem()
        view.mount("alias", clone.node())
        self.assertEqual(view.node("alias").origin, origin)
        self.assertTrue(view.node("alias").is_logical)
        self.assertEqual(view.node("alias/nested").origin, dataset.node().origin)

    def test_roots_distinguish_file_and_namespace_filesystems(self):
        self.assertTrue(self.fs.is_root)
        self.assertTrue(self.fs.node().is_root)
        self.assertFalse(self.fs.node().in_file)
        self.assertFalse(self.fs.node("metadata").is_root)
        self.assertTrue(self.fs.node("metadata/alpha.json").is_root)
        self.assertFalse(self.fs.node("metadata/alpha.json/coefficients").is_root)
        source = self.fs.clone("metadata/alpha.json")
        source.unmount("label")
        self.assertTrue(source.is_root)
        self.assertTrue(source.clone().node().is_root)
        self.assertTrue(source.clone("coefficients").is_root)
        source.mount("synthetic", FileSystem().node())
        self.assertTrue(source.node("synthetic").is_root)
        self.assertFalse(source.node("synthetic").in_file)
        combined = FileSystem()
        combined.mount("alias", source.node())
        self.assertTrue(combined.node("alias").is_root)
        combined.unmount("alias")
        combined.mount("alias", FileSystem().node())
        self.assertTrue(combined.node("alias").is_root)
        self.assertFalse(combined.node("alias").in_file)

    def test_in_file_tracks_source_origin_through_edits_and_mounts(self):
        self.assertFalse(self.fs.in_file)
        self.assertFalse(self.fs.node("metadata").in_file)
        for path in ("metadata/alpha.json", "metadata/alpha.json/coefficients",
                     "metadata/alpha.json/label"):
            self.assertTrue(self.fs.node(path).in_file)
        source = self.fs.clone("metadata/alpha.json")
        source.unmount("label")
        self.assertTrue(source.in_file)
        self.assertTrue(source.clone().in_file)
        source.mount("synthetic/empty", FileSystem().node())
        self.assertFalse(source.node("synthetic").in_file)
        self.assertFalse(source.node("synthetic/empty").in_file)
        combined = FileSystem()
        combined.mount("alias", source.node())
        self.assertTrue(combined.node("alias").in_file)
        self.assertTrue(combined.node("alias/coefficients").in_file)
        self.assertFalse(combined.node("alias/synthetic").in_file)
        combined.mount("namespace", self.fs.node("metadata"))
        self.assertFalse(combined.node("namespace").in_file)
        self.assertTrue(combined.node("namespace/alpha.json").in_file)

    def test_mount_flag_survives_clones_and_edits(self):
        self.assertFalse(self.fs.is_mount)
        self.assertFalse(self.fs.node("metadata").is_mount)
        self.assertTrue(self.fs.node("metadata/alpha.json").is_mount)
        self.assertFalse(self.fs.node("metadata/alpha.json/label").is_mount)
        source = self.fs.clone("metadata/alpha.json")
        self.assertTrue(source.is_mount)
        source.unmount("label")
        self.assertTrue(source.node().is_mount)
        source.mount("label", FileSystem().node())
        self.assertTrue(source.node("label").is_mount)
        self.assertTrue(source.clone().node("label").is_mount)
        source.unmount("label")
        with self.assertRaises(KeyError):
            _ = source.node("label").is_mount

    def test_logical_mounts_are_inferred_without_reads(self):
        def fail(*args):
            self.fail("Mount classification must not read sources")

        source = FileSystem.from_file("unread.custom", Reader((".custom",), fail))
        fs = FileSystem()
        fs.mount("unread.custom", source.node())
        self.assertFalse(fs.node("unread.custom").is_logical)
        fs.mount("alias", source.node())
        self.assertTrue(fs.node("alias").is_logical)
        self.assertFalse(fs.node("alias/unknown").is_logical)
        self.assertTrue(fs.clone().node("alias").is_logical)
        self.assertFalse(source.is_logical)
        self.assertFalse(source.is_mount)
        self.assertTrue(source.in_file)
        self.assertEqual(source.node().origin, SourceLocator(Path("unread.custom").absolute()))
        self.assertTrue(source.is_root)
        self.assertTrue(fs.node("alias").is_root)
        self.assertFalse(fs.node("alias/unknown").is_root)
        self.assertTrue(fs.node("alias/unknown").in_file)
        self.assertTrue(fs.node("unread.custom").is_mount)
        self.assertTrue(fs.clone().node("alias").is_mount)
        self.assertFalse(fs.node("alias/unknown").is_mount)
        fs.unmount("alias")
        fs.mount("alias", FileSystem().node())
        self.assertFalse(fs.node("alias").is_logical)

    def test_logical_mounts_preserve_positions_through_clones_and_edits(self):
        self.assertFalse(self.fs.node("metadata/alpha.json").is_logical)
        source = self.fs.clone("metadata/alpha.json")
        source.unmount("label")
        source.mount("label", label_source := self.fs.node("metadata/alpha.json/label"))
        self.assertFalse(source.node("label").is_logical)
        source.unmount("label")
        source.mount("alias", label_source)
        self.assertTrue(source.node("alias").is_logical)
        self.assertTrue(source.clone().node("alias").is_logical)
        combined = FileSystem(relative_to=self.fs.relative_to)
        combined.mount("metadata", self.fs.node("metadata"))
        self.assertFalse(combined.node("metadata").is_logical)
        combined.mount("renamed", self.fs.node("metadata"))
        self.assertTrue(combined.node("renamed").is_logical)
        self.assertFalse(combined.node("renamed/alpha.json").is_logical)

    def test_clone_matches_every_physical_node(self):
        self.assertEqual(self.view.read(), self.fs.read())
        self.assertEqual(self.view.locate(), self.fs.locate())

        def compare(path):
            self.assertEqual(self.view.node(path).kind, self.fs.node(path).kind)
            try:
                names = self.fs.listdir(path)
            except NotADirectoryError:
                with self.assertRaises(NotADirectoryError):
                    self.view.listdir(path)
                self.assertEqual(self.view.read(path), self.fs.read(path))
            else:
                self.assertEqual(self.view.listdir(path), names)
                for name in names:
                    compare(f"{path}/{name}")

        compare("")
        node = self.view.node(self.alpha).node("./vector")
        self.assertEqual(node.read(slice(0, 2)), [0, 0.5])
        self.assertEqual(node.locate(slice(0, 2)), self.fs.locate(f"{self.alpha}/vector", slice(0, 2)))

    def test_replacement_requires_unmount_and_updates_all_access(self):
        path = f"{self.alpha}/file_ref"
        handle = self.view.node(path)
        target = self.fs.node("metadata/alpha.json")
        with self.assertRaises(FileExistsError):
            self.view.mount(path, target)
        self.assertEqual(handle.read(), "metadata/alpha.json")
        self.view.unmount(path)
        self.assertNotIn("file_ref", self.view.listdir(self.alpha))
        self.assertNotIn("file_ref", self.view.read(self.alpha))
        for operation in (self.view.node, self.view.read, self.view.locate):
            with self.assertRaises(KeyError):
                operation(path)
        with self.assertRaises(KeyError):
            handle.read()
        self.view.mount(path, target)
        self.assertEqual(handle.kind, "group")
        self.assertEqual(handle.read(), target.read())
        self.assertEqual(self.view.read(self.alpha)["file_ref"], handle.read())
        self.assertEqual(self.view.locate(path), target.locate())
        self.assertEqual(self.fs.read(path), "metadata/alpha.json")
        self.assertNotIn(SourceLocator(self.fs.relative_to / "partitions.json", "partitions/alpha/file_ref"),
                         self.view.locate(self.alpha))

    def test_unmount_removes_a_whole_subtree_and_can_be_restored(self):
        self.view.unmount("metadata")
        self.assertNotIn("metadata", self.view.listdir())
        self.assertNotIn("metadata", self.view.read())
        self.assertFalse(any(locator.file_path.parent.name == "metadata" for locator in self.view.locate()))
        with self.assertRaises(KeyError):
            self.view.read("metadata/alpha.json/label")
        self.view.mount("metadata", self.fs.node("metadata"))
        self.assertEqual(self.view.read(), self.fs.read())

    def test_file_backed_filesystem_can_be_edited_directly_without_writing_files(self):
        file = self.fs.relative_to / "partitions.json"
        original = file.read_bytes()
        path = f"{self.alpha}/file_ref"
        handle = self.fs.node(path)
        self.fs.unmount(path)
        with self.assertRaises(KeyError):
            handle.read()
        self.fs.mount(path, self.fs.node("metadata/alpha.json"))
        self.assertEqual(handle.read(), self.fs.read("metadata/alpha.json"))
        self.assertEqual(self.fs.read(self.alpha)["file_ref"], handle.read())
        self.assertEqual(self.view.read(path), "metadata/alpha.json")
        self.assertEqual(file.read_bytes(), original)

    def test_sources_with_identical_names_in_different_roots_remain_distinct(self):
        with TemporaryDirectory() as directory:
            combined = FileSystem()
            for name, value in (("left", 10), ("right", 20)):
                root = Path(directory) / name
                root.mkdir()
                (root / "same.json").write_text(json.dumps({"value": value}))
                source = FileSystem("same.json", relative_to=root)
                combined.mount(name, source.node("same.json"))
                self.assertEqual(combined.read(f"{name}/value"), value)
                self.assertEqual(combined.locate(f"{name}/value"),
                                 [SourceLocator(root / "same.json", "value")])
            self.assertNotEqual(combined.locate("left"), combined.locate("right"))
            clone = combined.clone("left/value")
            self.assertEqual(clone.read(), 10)
            self.assertEqual(clone.node().kind, "scalar")

    def test_mount_creates_parents_and_preserves_leaf_semantics(self):
        view = FileSystem()
        source = self.fs.node("metadata/alpha.json/coefficients")
        view.mount("/new//./nested/vector/", source)
        self.assertEqual(view.listdir(), ("new",))
        self.assertEqual(view.read(), {"new": {"nested": {"vector": source.read()}}})
        self.assertEqual(view.read("new/nested/vector", 0), 1.01)
        with self.assertRaises(NotADirectoryError):
            view.mount("new/nested/vector/child", source)
        view.unmount("new/nested/vector")
        group = FileSystem()
        group.mount("child", source)
        view.mount("new/nested/vector", group.node())
        self.assertEqual(view.read("new/nested/vector/child"), source.read())
        for operation in (view.read, view.locate):
            with self.assertRaises(ValueError):
                operation("new/nested/vector", 0)

    def test_mounted_subtrees_have_independent_edits(self):
        source = self.fs.clone(self.alpha)
        source.unmount("file_ref")
        source.mount("file_ref", self.fs.node("metadata/alpha.json"))
        view = FileSystem()
        view.mount("copy", source.node())
        view.mount("again", source.node())
        self.assertEqual(view.read("copy"), source.read())
        self.assertEqual(view.locate("copy"), source.locate())
        view.unmount("copy/file_ref/label")
        self.assertIn("label", source.listdir("file_ref"))
        self.assertIn("label", view.listdir("again/file_ref"))
        source.unmount("scalar_a")
        self.assertIn("scalar_a", view.listdir("copy"))
        # Even mounting a clone of our own root takes a finite namespace snapshot.
        view.mount("snapshot", view.node())
        self.assertEqual(view.listdir("snapshot"), ("copy", "again"))

    def test_mount_retains_its_source_reader_and_values_remain_live(self):
        values = {"vector": [20, 30]}
        reader = Reader((".custom",), lambda path: values)
        other = FileSystem("other.custom", reader)
        self.view.mount("external", other.node("other.custom"))
        self.assertEqual(self.view.read("external/vector", 1), 30)
        values["vector"][1] = 40
        self.assertEqual(self.view.read("external/vector", 1), 40)
        self.assertEqual(self.view.locate("external/vector", 1),
                         [SourceLocator(other.relative_to / "other.custom", "vector", 1)])

    def test_invalid_edits_and_empty_namespace(self):
        empty = FileSystem()
        self.assertEqual(empty.read(), {})
        self.assertEqual(empty.listdir(), ())
        self.assertEqual(empty.locate(), [])
        self.assertEqual(empty.node().kind, "group")
        source = self.fs.node("metadata/alpha.json")
        for path in ("", "/", "../invalid"):
            for operation, args in ((empty.mount, (path, source)), (empty.unmount, (path,))):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    operation(*args)
        with self.assertRaises(TypeError):
            empty.mount("bad", "metadata/alpha.json")
        with self.assertRaises(TypeError):
            FileSystem({"bad": source})
        self.assertEqual(empty.read(), {})
        with self.assertRaises(KeyError):
            empty.unmount("missing/child")
        with self.assertRaises(ValueError):
            self.view.node(self.alpha).node("../beta")
        missing = self.view.node(f"{self.alpha}/missing")
        with self.assertRaises(KeyError):
            missing.read()

    def test_physical_file_mounts_cannot_overlap(self):
        paths = ("partitions.json", "partitions.json/inner.json")
        for order in (paths, paths[::-1]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                FileSystem(order)

    def test_demo_replaces_only_the_requested_reference_fields(self):
        opened = []
        with patch.dict("sys.modules", {"explorer": SimpleNamespace(explore=opened.append)}):
            runpy.run_path(str(EXAMPLE / "use_mounts.py"), run_name="__main__")
        view, = opened
        self.assertEqual(view.listdir(), ("partitions.json",))
        self.assertFalse(view.node("partitions.json").is_logical)
        for name in ("alpha", "beta", "gamma"):
            prefix = f"partitions.json/partitions/{name}"
            self.assertTrue(view.node(f"{prefix}/file_ref").is_logical)
            self.assertTrue(view.node(f"{prefix}/file_refs").is_logical)
            raw = self.fs.read(prefix)
            self.assertEqual(view.read(f"{prefix}/file_ref"), self.fs.read(raw["file_ref"]))
            self.assertEqual(view.listdir(f"{prefix}/file_refs"), ("0", "1"))
            complete = view.read(prefix)
            for field in ("scalar_a", "scalar_b", "vector", "matrix"):
                self.assertEqual(complete[field], raw[field])
            for index, reference in enumerate(raw["file_refs"]):
                self.assertEqual(complete["file_refs"][str(index)], self.fs.read(reference))
            self.assertEqual(complete["file_ref"], view.read(f"{prefix}/file_ref"))
        for path in ("metadata", "blocks"):
            with self.assertRaises(KeyError):
                view.read(path)


class ReaderDelegationTests(unittest.TestCase):
    def test_from_file_is_lazy_mountable_and_has_independent_namespace_edits(self):
        calls = []
        payload = {"group": {"values": [10, 20], "reference": "ordinary string"}}

        def load(path):
            calls.append(path)
            return payload

        source = FileSystem.from_file("sample.custom", Reader((".custom",), load))
        combined = FileSystem()
        combined.mount("runs/first", source.node())
        self.assertEqual(source.node().kind, "group")
        self.assertEqual(source.locate(), [SourceLocator(Path("sample.custom").absolute())])
        self.assertEqual(calls, [])
        source.unmount("group/reference")
        self.assertEqual(source.read("group"), {"values": [10, 20]})
        self.assertEqual(combined.read("runs/first/group/reference"), "ordinary string")
        self.assertEqual(combined.read("runs/first/group/values", 1), 20)
        self.assertEqual(FileSystem.from_file(EXAMPLE / "data/metadata/alpha.json").read("label"), "alpha")

    def test_registration_deduplicates_files_and_uses_longest_local_suffix_without_io(self):
        def fail(*args):
            self.fail("Registration and directory inspection must not access file contents")

        short = Reader((".json",), fail, read=lambda *args: "short")
        long = Reader((".META.JSON",), fail, read=lambda *args: "long")
        fs = FileSystem(["folder/test.meta.json", "folder/test.meta.json"], [short, long])
        self.assertEqual(fs.listdir(), ("folder",))
        self.assertEqual(fs.listdir("folder"), ("test.meta.json",))
        self.assertEqual(fs.node("folder/test.meta.json").kind, "group")
        self.assertEqual(fs.clone().locate(),
                         [SourceLocator(fs.relative_to / "folder/test.meta.json")])
        self.assertEqual(fs.read("folder/test.meta.json/field"), "long")

    def test_invalid_registration_is_rejected(self):
        reader = Reader((".json",), lambda path: {})
        for paths in ("/absolute.json", "../escape.json", ".", ["valid.json", "../../escape.json"]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                FileSystem(paths)
        for readers, error in (([object()], TypeError),
                               ([Reader(("json",), lambda path: {})], ValueError),
                               ([reader, reader], ValueError)):
            with self.subTest(readers=readers), self.assertRaises(error):
                FileSystem("data.json", readers)

    def test_only_edits_discover_structure_and_locate_never_reads(self):
        calls = []

        def load(path):
            self.fail("Unexpected whole-file load")

        def read(path, key, selection):
            calls.append(("read", key, selection))
            return "selected"

        def info(path, key):
            calls.append(("info", key))
            return NodeInfo("group", ("values", "link")) if key == "group" else NodeInfo("dataset")

        fs = FileSystem("sample.custom", Reader((".custom",), load, read=read, inspect_node=info))
        view = fs.clone("sample.custom/group")
        node = view.node("values")
        selection = (slice(None, None, -1), 0)
        self.assertEqual(node.locate(selection), [SourceLocator(fs.relative_to / "sample.custom", "group/values", selection)])
        self.assertEqual(calls, [])
        view.unmount("link")
        self.assertEqual(calls, [("info", "group")])
        calls.clear()
        view.mount("link", node)
        self.assertEqual(view.listdir(), ("values", "link"))
        self.assertEqual(view.node().kind, "group")
        self.assertEqual(view.locate(), fs.locate("sample.custom/group/values") * 2)
        self.assertEqual(calls, [])
        self.assertEqual(view.read("link", selection), "selected")
        self.assertIs(calls[-1][2], selection)

    def test_pickle_and_archive_sources_support_the_same_edits(self):
        payload = {"group": {"values": [1, 2, 3], "ref": "ordinary string"}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.pkl").write_bytes(pickle.dumps(payload))
            with ZipFile(root / "data.zip", "w") as archive:
                archive.writestr("inner.json", json.dumps(payload))
            fs = FileSystem(["data.pkl", "data.zip"], relative_to=root)
            for filename in ("data.pkl", "data.zip"):
                with self.subTest(filename=filename):
                    prefix = f"{filename}/group"
                    self.assertEqual(fs.read(f"{prefix}/values", slice(1, 3)), [2, 3])
                    fs.unmount(f"{prefix}/ref")
                    self.assertEqual(fs.read(prefix), {"values": [1, 2, 3]})
                    self.assertEqual(fs.locate(prefix),
                                     [SourceLocator(root / filename, "group/values")])

    def test_npz_metadata_and_selection_survive_namespace_edits(self):
        import numpy as np
        from lazyfields.extra_readers.npz import NUMPY_ARCHIVE

        with TemporaryDirectory() as directory:
            np.savez(Path(directory) / "data.npz", values=np.arange(6).reshape(2, 3))
            fs = FileSystem("data.npz", NUMPY_ARCHIVE, relative_to=directory)
            clone = fs.clone("data.npz")
            with patch("numpy.lib.npyio.NpzFile.__getitem__", side_effect=AssertionError("array loaded for listing")):
                self.assertEqual(fs.listdir("data.npz"), ("values",))
                clone.unmount("values")
                self.assertEqual(clone.locate(), [])
                clone.mount("alias", fs.node("data.npz/values"))
            self.assertEqual(clone.read("alias", (1, slice(1, 3))).tolist(), [4, 5])
            self.assertEqual(clone.locate("alias"), fs.locate("data.npz/values"))

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is optional")
    def test_hdf5_clone_edits_use_metadata_and_keep_direct_selection(self):
        import h5py

        with TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "data.h5", "w") as file:
                file["group/values"] = [[1, 2, 3], [4, 5, 6]]
                file["group/scalar"] = 7
            fs = FileSystem("data.h5", relative_to=directory)
            view = fs.clone("data.h5/group")
            with patch("lazyfields.readers._read", side_effect=AssertionError("unexpected materialization")):
                self.assertEqual(set(view.listdir()), {"scalar", "values"})
                self.assertEqual(view.node("values").kind, "dataset")
                self.assertEqual(view.read("values", (1, slice(0, 2))).tolist(), [4, 5])
                view.unmount("scalar")
                view.mount("scalar", fs.node("data.h5/group/values"))
                self.assertEqual(view.locate(), fs.locate("data.h5/group/values") * 2)
                self.assertEqual(view.read("scalar", (0, slice(1, 3))).tolist(), [2, 3])
            self.assertEqual(view.read()["values"].tolist(), [[1, 2, 3], [4, 5, 6]])


if __name__ == "__main__":
    unittest.main()
