import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


def make_cuda_ext(
    name,
    module,
    sources,
    sources_cuda=[],
    extra_args=[],
    extra_include_path=[],
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    plugins = ['deform_agg']
    sources = []
    for plugin in plugins:
        sources += [
            f"csrc/{plugin}.cpp",
            f"csrc/autocast/{plugin}_kernel.cpp",
            f"csrc/autograd/{plugin}_kernel.cpp",
            f"csrc/cuda/{plugin}_kernel.cu",
        ]
    setup(
        name="custom_ops",
        ext_modules=[
            make_cuda_ext(
                "custom_ops",
                module=".",
                sources=sources,
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
    )
