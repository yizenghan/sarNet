from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("select_feature_cython",    # location of the resulting .so
             ["select_feature_cython.pyx"],
              extra_compile_args=["-fopenmp", "-march=native", '-O3', '-msse', '-msse2', '-mfpmath=sse'],
              extra_link_args=["-fopenmp"],
              ) ]


setup(name='package',
      packages=find_packages(),
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
     )