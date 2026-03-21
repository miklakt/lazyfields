from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent


setup(
    name="lazyfields",
    version="0.1.0",
    description="A minimal pandas reference-table layer with lazy persisted row storage.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author = "Mikhail Laktionov",
    email = "miklakt@gmail.com",
    packages=["lazyfields"],
    package_dir={"lazyfields": "."},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=1.5",
    ],
    extras_require={
        "hdf5": ["h5py>=3"],
    },
)
