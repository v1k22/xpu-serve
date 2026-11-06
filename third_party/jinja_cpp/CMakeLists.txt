# Vendored Jinja engine (see PROVENANCE.md). Built without our warning flags; its headers are SYSTEM includes.
add_library(xs_jinja STATIC caps.cpp jinja_string.cpp lexer.cpp parser.cpp runtime.cpp value.cpp)
target_include_directories(xs_jinja SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xs_jinja PUBLIC xs_json)
