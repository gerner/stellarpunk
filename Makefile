SOURCES=$(shell find src -name *.pyx -o -name *.hpp)

all: ext
	dmypy run

ext: build/build_flag

build/build_flag: $(SOURCES)
	pip install -e .
	touch build/build_flag

clean:
	rm -rf build
