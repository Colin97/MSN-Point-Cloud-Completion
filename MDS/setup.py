from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='MDS',
    ext_modules=[
        CUDAExtension('MDS', [
            'MDS_cuda.cu',
            'MDS.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })