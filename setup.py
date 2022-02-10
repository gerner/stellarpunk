import os.path
import setuptools

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stellarpunk",
    version="0.0.1-pre.1",
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
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "numba",
        "ipdb",
        "graphviz",
        "drawille",
        "pymunk",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'stellarpunk = stellarpunk.sim:main',
            'hist_extract = stellarpunk.hist_extract:main',
        ],
    },
)
