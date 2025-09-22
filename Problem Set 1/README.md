# Harvard CS 2420 Fall 2025: Problem Set 1

## Implementing MMM

For Parts 3.1 and 3.2, you **only** need to edit `pset1.cpp`.

**_Do not modify any other file._**

## Testing Your MMM Implementations

To build your code, simply run `make`.

Run `make test` to validate your implementation's correctness.

## Timing Your MMM Implementations

For timing results, run `make` and then `./mm N`, where `N` is the desired matrix dimension.

The program will run 5 iterations of your MMM implementations for the given `N` and print out the average run time in nanoseconds (ns).

For example, to time 1024 x 1024 matrices, you would run `./mm 1024`.

You should see something similar to:

```sh
cs242@ubuntu:~/pset1$ ./mm 1024
Initializing matrices
Initialization done
Performing MMs for 1024x1024 matrices (5 iters)
Inner product MMM avg. run time (ns): 3966481258
[PASS] All values correct!
Outer product MMM avg. run time (ns): 90788842
[PASS] All values correct!
```

## Questions

If you have any questions, please contact the teaching staff.
