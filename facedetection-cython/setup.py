import numpy
from setuptools import setup
from Cython.Build import cythonize
directives = {'language_level': '3',
              'always_allow_keywords': True}

setup(
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(["pipeline/rectangle.pyx", "pipeline/classifier.pyx", "pipeline/stage.pyx", "pipeline/cascade_classifier.pyx", "pipeline/feature.pyx",  "parser1.pyx","maincython.pyx"]
    , compiler_directives=directives)
)