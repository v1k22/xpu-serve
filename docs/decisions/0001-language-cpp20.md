# 0001: C++20 for the whole repo

Date: 2026-11-06 · Status: accepted (user decision)

Both planned backends are C++-first: SYCL kernels can only be C++ (icpx), and OpenVINO's graph-building API is
C++ (its C API can load and run models but cannot build a graph op by op). Any other language would still need a C++
layer under the backend interface, so the repo would carry two languages and a C ABI between them.

C++20 everywhere gives one language, one toolchain and direct access to OpenVINO, SYCL and oneDNN, and it is the
language most likely contributors already use. The cost is verbosity and the lack of a memory-safe language for the
engine and server, which we mitigate with RAII, no raw new/delete, and tests at every layer.

Constraints: no `std::format` / `std::expected` (the oldest supported compiler is g++ 11).
