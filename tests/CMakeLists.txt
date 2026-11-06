# One test binary; every tests/unit/**/*.cpp is part of it (add a file, no CMake edit needed).
file(GLOB_RECURSE xs_test_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp)
add_executable(xs_tests ${xs_test_sources})
target_link_libraries(xs_tests PRIVATE xs_service xs_server xs_doctest xs_warnings)
target_include_directories(xs_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/unit)
target_compile_definitions(xs_tests PRIVATE XS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
# Tests that need a model directory or fixtures read XS_TEST_MODEL_DIR / XS_TEST_FIXTURES and skip when unset.
add_test(NAME unit COMMAND xs_tests)
