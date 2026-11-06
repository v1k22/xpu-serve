# Each area is a static library whose sources are globbed from its directory, so adding a file never needs a
# CMakeLists edit. Public headers are included as "area/file.h" from src/.

function(xs_add_library name)
  cmake_parse_arguments(ARG "" "" "DIRS;DEPS" ${ARGN})
  set(srcs)
  foreach(d ${ARG_DIRS})
    file(GLOB_RECURSE found CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${d}/*.cpp)
    list(APPEND srcs ${found})
  endforeach()
  if(srcs)
    add_library(${name} STATIC ${srcs})
    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} PUBLIC ${ARG_DEPS} PRIVATE xs_warnings)
  else()
    add_library(${name} INTERFACE)
    target_include_directories(${name} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} INTERFACE ${ARG_DEPS})
  endif()
endfunction()

# Core: utilities, host tensors, safetensors I/O, the step graph, model code and the engine.
xs_add_library(xs_core DIRS util tensor io graph models engine DEPS xs_json Threads::Threads)

xs_add_library(xs_tokenizer DIRS tokenizer DEPS xs_core)

set(xs_chat_deps xs_core)
if(TARGET xs_jinja)
  list(APPEND xs_chat_deps xs_jinja)
endif()
xs_add_library(xs_chat DIRS chat DEPS ${xs_chat_deps})

xs_add_library(xs_backend_cpu DIRS backend/cpu_ref DEPS xs_core)

if(XS_WITH_OPENVINO)
  xs_add_library(xs_backend_openvino DIRS backend/openvino DEPS xs_core openvino::runtime)
endif()

# Backend registry: the one place that knows which backends were compiled in.
add_library(xs_backends STATIC backend/registry.cpp)
target_include_directories(xs_backends PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xs_backends PUBLIC xs_core xs_backend_cpu PRIVATE xs_warnings)
if(XS_WITH_OPENVINO)
  target_link_libraries(xs_backends PUBLIC xs_backend_openvino)
  target_compile_definitions(xs_backends PRIVATE XS_HAVE_OPENVINO=1)
endif()

xs_add_library(xs_server DIRS server DEPS xs_core xs_tokenizer xs_chat xs_httplib)

add_executable(xpu-serve cli/main.cpp)
file(GLOB xs_cli_extra CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cli/*.cpp)
list(REMOVE_ITEM xs_cli_extra ${CMAKE_CURRENT_SOURCE_DIR}/cli/main.cpp)
target_sources(xpu-serve PRIVATE ${xs_cli_extra})
target_link_libraries(xpu-serve PRIVATE xs_core xs_backends xs_tokenizer xs_chat xs_server xs_warnings)
target_compile_definitions(xpu-serve PRIVATE XS_VERSION="${PROJECT_VERSION}")
