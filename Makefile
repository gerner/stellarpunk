SHELL=/bin/bash
SOURCES=$(shell find src -name *.pyx -o -name *.hpp)

all: ext
	dmypy run

ext: build/build_flag

build/build_flag: $(SOURCES)
	# rewrite strings like "build/src/stellarpunk/" to "src/stellarpunk/"
	# because setuptools copies our c/c++ sources to build before compiling
	# this rewrite lets us find the corresponding file in the actual source
	# tree this is useful for stuff like vim's quickfix list
	#pip install -e . 2> >(sed 's/build\/src\/stellarpunk\//src\/stellarpunk\//' 1>&2)
	pip install -e . 2> >(gawk '@load "filefuncs"; {match($$0, "build/(src/stellarpunk/[^:]+):", m); if(stat(m[1], fstat) == 0) { sub("build/src/stellarpunk/", "src/stellarpunk/")} print $$0}' 1>&2)

	touch build/build_flag

clean:
	rm -rf build

test:
	pytest

lint:
	flake8 src/stellarpunk/ tests/

GOAPTOY_SRC:=tools/goaptoy.cpp
goaptoy: $(shell find src/stellarpunk/narrative -name *.hpp) $(GOAPTOY_SRC)
	#clang++-15 -fuse-ld=lld -Isrc/stellarpunk/narrative/ -fsanitize=address -g -O0 -std=c++20 -o goaptoy $(GOAPTOY_SRC)
	clang++-15 -fuse-ld=lld -Isrc/stellarpunk/narrative/ -O3 -std=c++20 -o goaptoy $(GOAPTOY_SRC)
