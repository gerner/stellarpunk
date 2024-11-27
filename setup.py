import os
import os.path
import setuptools # type: ignore

#TODO: handle cython not available for distribution case...
from Cython.Build import cythonize #type: ignore
from setuptools import Extension

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as fh:
    long_description = fh.read()

os.environ['CXX'] = 'clang++'
os.environ['LDSHARED'] = 'clang -shared'

import cymunk
ext_modules=[
    Extension("stellarpunk.orders.collision",
        sources=["src/stellarpunk/orders/collision.pyx"],
        libraries=[":cymunk.cpython-312-x86_64-linux-gnu.so"],
        library_dirs=cymunk.get_includes(),
        runtime_library_dirs=cymunk.get_includes(),
        language="c++",
    ),
    Extension("stellarpunk.narrative.director",
        sources=["src/stellarpunk/narrative/director.pyx"],
        language="c++",
    ),
    Extension("stellarpunk.narrative.goap",
        sources=["src/stellarpunk/narrative/goap.pyx"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O2"],
    ),
    Extension("stellarpunk.task_schedule",
        sources=["src/stellarpunk/task_schedule.pyx"],
        language="c++",
    ),
    Extension("stellarpunk.generate.markov",
        sources=["src/stellarpunk/generate/markov.pyx"],
        libraries=["boost_iostreams"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O0"],
        #undef_macros = [ "NDEBUG" ],
    ),
]

extensions = cythonize(
        ext_modules,
        build_dir="build",
        annotate=True,
        language_level="3",
)

setuptools.setup(
    name="stellarpunk",
    version="0.0.1rc1",
    author="Nick Gerner",
    author_email="nick.gerner@gmail.com",
    description="Stellar Punk: A space sim of exploration, trading, stealth naval combat and interplanetary economic development.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gerner/stellarpunk",
    packages=setuptools.find_packages(where="src"),
    package_dir={'': 'src'},

    package_data={
        #'datascience': ['logging.ini'],
        'stellarpunk': ['py.typed'],
        'stellarpunk.interface': ['py.typed'],
        'stellarpunk.orders': ['py.typed'],
        'stellarpunk.data': ['*'],
    },
    ext_modules=extensions,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "cython",
        "numpy",
        "numba",
        "ipdb",
        "graphviz",
        "drawille",
        "cymunk",
        "rtree",
        "toml",
        "tqdm",
        "pysdl2",
        "pysdl2-dll",
        "pysdl2",
        "dtmf",
        "uroman",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'stellarpunk = stellarpunk.sim:main',
            'spunk_econ = stellarpunk.econ_sim:main',
            'hist_extract = stellarpunk.hist_extract:main',
        ],
    },
)
