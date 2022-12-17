from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name="graphiler",
    version="0.0.1",
    url="https://github.com/xiezhq-hermann/graphiler",
    package_dir={"": "python"},
    packages=["graphiler", "graphiler.utils", "graphiler.utils_lite"],
    ext_modules=[
        CppExtension(
            "graphiler.mpdfg",
            [
                "src/pybind.cpp",
                "src/builder/builder.cpp",
                "src/optimizer/dedup.cpp",
                "src/optimizer/split.cpp",
                "src/optimizer/reorder.cpp",
                "src/optimizer/fusion.cpp",
                "src/optimizer/backpropemitter.cpp",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.7",
)
