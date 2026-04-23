---
name: cpp-quality
triggers:
  - c++
  - cpp
  - cmake
  - clang-tidy
  - clang-format
  - raii
  - ownership
  - header
  - include-what-you-use
profiles:
  - general
requires:
  - afs
enforcement:
  - Make ownership and lifetime explicit; prefer RAII and value semantics.
  - Avoid raw new/delete in application logic and default shared ownership.
  - Keep headers narrow and remove avoidable dependencies.
verification:
  - Build and run the existing CMake or CTest checks for touched targets.
  - Run clang-tidy or clang-format checks when the repo already uses them.
---

# C++ Quality

Optimize for explicit ownership, narrow interfaces, and low-ambiguity lifetimes.

## Prefer

- RAII for every acquired resource.
- Value semantics by default; pointers should communicate ownership or optionality.
- `const`, `override`, `enum class`, and scoped types when they remove ambiguity.
- Small headers, forward declarations where valid, and translation-unit-local helpers.
- Error handling that makes failure visible at the call site.

## Avoid

- Raw `new` or `delete` in application logic.
- `shared_ptr` as the default ownership model.
- Non-virtual base destructors on polymorphic types.
- Large headers that pull in unrelated dependencies.
- Boolean parameter soup and hidden mutable global state.

## Quality Gates

Run the checks the repo already uses. Common baselines:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
clang-tidy <files> --
clang-format --dry-run --Werror <files>
include-what-you-use <files>
```
