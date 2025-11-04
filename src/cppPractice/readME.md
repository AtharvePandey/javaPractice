# follows cpp oop course on youtube

course:
<https://www.youtube.com/watch?v=MgJ2kPl15Io&list=PLeh6VbPaYLX8LVS-IPUtt6AjensvzONyV&index=5>

answers all questions and does practice problems according to L1 -> L5 so far...

## How to add and build programs in this folder

Add a new C++ source file with a `main` function, for example `L2.cpp`.

- To build all programs that contain `main` (automatically detects them):

```bash
cd /path/to/javaPractice/src/cppPractice
make
```

- To build a single program (e.g., `L2`):

```bash
make L2
```

- To run the first detected program (convenience):

```bash
make run
```

- To remove built executables and object files:

```bash
make clean
```

Notes:

- The Makefile will only create executables for `.cpp` files that contain a `main` function to avoid linker errors for helper/source files.
- If you write multi-file programs (multiple `.cpp` files for one executable), add an explicit target to the Makefile listing the required object files.
