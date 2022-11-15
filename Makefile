all:
	dmypy run
	pip install -e .

build:
	python setup.py build_ext

clean:
	rm -rf build
