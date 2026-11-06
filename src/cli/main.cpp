// Placeholder entry point; the real CLI (serve, run, info) lands with the engine.
#include <cstdio>

#include "backend/backend.h"

int main() {
    for (const auto& b : xs::available_backends()) std::printf("%s\n", b.c_str());
    return 0;
}
