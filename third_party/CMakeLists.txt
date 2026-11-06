# Header-only dependencies, exposed as INTERFACE targets with SYSTEM includes so their warnings stay out of our build.
add_library(xs_json INTERFACE)
target_include_directories(xs_json SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(xs_httplib INTERFACE)
target_include_directories(xs_httplib SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xs_httplib INTERFACE Threads::Threads)

add_library(xs_doctest INTERFACE)
target_include_directories(xs_doctest SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Jinja engine for chat templates (vendored sources; see third_party/README.md).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/jinja_cpp/CMakeLists.txt)
  add_subdirectory(jinja_cpp)
endif()
