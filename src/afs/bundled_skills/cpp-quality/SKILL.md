---
name: cpp-quality
triggers:
  - c++
  - cpp
  - cmake
  - CMakeLists.txt
  - clang-tidy
  - clang-format
  - raii
  - std::
  - unique_ptr
  - sanitizer
  - include-what-you-use
profiles:
  - general
requires:
  - afs
enforcement:
  - Make ownership, lifetime, aliasing, and thread safety explicit; prefer RAII and value semantics.
  - Treat dangling references or views, data races, undefined behavior, and compiler warnings as correctness defects.
  - Preserve public API and ABI constraints and use the repository's existing build configuration.
verification:
  - Build and test the smallest touched target with the repository's CMake presets or wrapper commands.
  - Run configured warnings, sanitizers, clang-tidy, and format checks; state which are unavailable.
---

# C++ Quality

Optimize for explicit ownership, valid lifetimes, narrow interfaces, and defined behavior.

## Core Rules

- Use RAII for every acquired resource and value semantics when copying is valid.
- Use references, pointers, smart pointers, `span`, and views only when their lifetime and ownership meaning is clear.
- Keep headers self-contained and minimal. Do not forward-declare standard-library types.
- Make concurrency ownership and synchronization visible; avoid shared mutable state.
- Preserve established API, ABI, exception, and allocation constraints unless the task changes them.

## Avoid

- Raw owning pointers, manual `new` or `delete`, and default `shared_ptr` ownership.
- Dangling iterators, references, `span`, or `string_view` values.
- Unchecked narrowing, unsafe casts, signed overflow assumptions, and other undefined behavior.
- Warning suppression, sanitizer suppression, or ignored return values without evidence.
- Macro utilities or boolean flags when a scoped type or separate operation is clearer.

## Quality Gates

Use repository scripts and presets first. Typical focused checks:

```bash
cmake --preset <configure-preset>
cmake --build --preset <build-preset> --target <target>
ctest --preset <test-preset> --output-on-failure
clang-tidy <files> -p <build-dir>
clang-format --dry-run --Werror <files>
```

Use ASan, UBSan, TSan, and include-what-you-use only when the repository supports them.
