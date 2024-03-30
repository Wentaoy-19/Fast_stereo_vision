from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sgm_cuda_py",
    ext_modules=[
        CUDAExtension(
            "sgm_cuda_c",
            [
                "src/sgm_cuda/sgm_cuda.cpp",
                "src/sgm_cuda/costs.cu",
                "src/sgm_cuda/disparity_method.cu",
                "src/sgm_cuda/hamming_cost.cu",
                "src/sgm_cuda/median_filter.cu",
            ],
        )
    ],
    package_dir={"": "src"},
    packages=["sgm_cuda_py"],
    cmdclass={"build_ext": BuildExtension},
)
