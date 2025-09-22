# Target rules
all: clean mm

# binaries
mm:
	g++ -std=c++11 -o mm main.cpp pset1.cpp -O3
	g++ -std=c++11 -o mm-test test.cpp pset1.cpp -lopenblas -O3

test: mm
	./mm-test

clean:
	rm -f mm
	rm -f mm-test
