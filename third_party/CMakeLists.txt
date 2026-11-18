# Header-only dependencies, exposed as INTERFACE targets with SYSTEM includes so their warnings stay out of our build.
add_library(xs_json INTERFACE)
target_include_directories(xs_json SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(xs_httplib INTERFACE)
target_include_directories(xs_httplib SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# Every translation unit sees httplib with TLS, so its classes are the same in all of them.
target_compile_definitions(xs_httplib INTERFACE CPPHTTPLIB_OPENSSL_SUPPORT)
target_link_libraries(xs_httplib INTERFACE Threads::Threads OpenSSL::SSL OpenSSL::Crypto)

add_library(xs_doctest INTERFACE)
target_include_directories(xs_doctest SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# ggml's quantization tables (data only; see ggml/PROVENANCE.md), included through src/quant/block_tables.h.
add_library(xs_ggml_tables INTERFACE)
target_include_directories(xs_ggml_tables SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Jinja engine for chat templates (vendored sources; see third_party/README.md).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/jinja_cpp/CMakeLists.txt)
  add_subdirectory(jinja_cpp)
endif()

# Image decoding (stb_image, public domain / MIT): one compiled translation unit, its warnings silenced.
add_library(xs_stb_image STATIC stb/stb_image.cpp)
target_include_directories(xs_stb_image SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/stb)
target_compile_options(xs_stb_image PRIVATE -w)
set_target_properties(xs_stb_image PROPERTIES POSITION_INDEPENDENT_CODE ON)

# MP3 and FLAC decoding (dr_mp3, dr_flac; public domain / MIT-0): one compiled translation unit, its warnings silenced.
add_library(xs_dr_libs STATIC dr_libs/dr_libs.cpp)
target_include_directories(xs_dr_libs SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/dr_libs)
target_compile_options(xs_dr_libs PRIVATE -w)
set_target_properties(xs_dr_libs PROPERTIES POSITION_INDEPENDENT_CODE ON)
