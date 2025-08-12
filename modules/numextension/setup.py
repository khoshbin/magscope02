from setuptools import setup, Extension
import numpy

# Get numpy include directory
numpyinc = numpy.get_include()

# Define the C extension module
numextmod = Extension(
    'numextension',
    sources=[
        'numextension.c', 
        'canny_edge.c', 
        'allstats.c'
    ],
    include_dirs=[numpyinc],
    define_macros=[
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ('PY_ARRAY_UNIQUE_SYMBOL', 'numextension_ARRAY_API')
    ]
)

# Call setup with minimal configuration
# Most configuration is now in pyproject.toml
setup(
    ext_modules=[numextmod],
)