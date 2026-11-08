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

# Core: utilities, host tensors, safetensors I/O, images, the step graph, model code and the engine.
xs_add_library(xs_core DIRS util tensor io image graph quant models engine DEPS xs_json xs_stb_image Threads::Threads)

xs_add_library(xs_tokenizer DIRS tokenizer DEPS xs_core)

set(xs_chat_deps xs_core)
if(TARGET xs_jinja)
  list(APPEND xs_chat_deps xs_jinja)
endif()
xs_add_library(xs_chat DIRS chat DEPS ${xs_chat_deps})

xs_add_library(xs_backend_cpu DIRS backend/cpu_ref DEPS xs_core)

if(XS_WITH_SYCL)
  xs_add_library(xs_backend_sycl DIRS backend/sycl DEPS xs_core)
  set(xs_sycl_flags -fsycl)
  if(XS_SYCL_TARGETS)
    list(APPEND xs_sycl_flags -fsycl-targets=${XS_SYCL_TARGETS})
  endif()
  target_compile_options(xs_backend_sycl PRIVATE ${xs_sycl_flags})
  # Executables that link the backend need the SYCL device link step.
  target_link_options(xs_backend_sycl INTERFACE ${xs_sycl_flags})
  target_link_libraries(xs_backend_sycl PRIVATE MKL::MKL_SYCL::BLAS)
endif()

# Backend registry: the one place that knows which backends were compiled in.
add_library(xs_backends STATIC backend/registry.cpp)
target_include_directories(xs_backends PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xs_backends PUBLIC xs_core xs_backend_cpu PRIVATE xs_warnings)
if(XS_WITH_SYCL)
  target_link_libraries(xs_backends PUBLIC xs_backend_sycl)
  target_compile_definitions(xs_backends PRIVATE XS_HAVE_SYCL=1)
endif()

xs_add_library(xs_server DIRS server DEPS xs_core xs_tokenizer xs_chat xs_httplib)

# `convert`: checkpoint -> pack; AWQ calibration runs the model on a backend and needs the tokenizer.
xs_add_library(xs_convert DIRS convert DEPS xs_core xs_backends xs_tokenizer)

# The policy runtime: a robot policy, its tokenizer and its compiled programs on one device.
xs_add_library(xs_policy DIRS policy DEPS xs_core xs_backends xs_tokenizer)

# The InferenceService that ties model, backend, tokenizer and chat template together.
xs_add_library(xs_service DIRS service DEPS xs_core xs_backends xs_tokenizer xs_chat)

add_executable(xpu-serve cli/main.cpp)
file(GLOB xs_cli_extra CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cli/*.cpp)
list(REMOVE_ITEM xs_cli_extra ${CMAKE_CURRENT_SOURCE_DIR}/cli/main.cpp)
target_sources(xpu-serve PRIVATE ${xs_cli_extra})
target_link_libraries(xpu-serve PRIVATE xs_service xs_server xs_convert xs_policy xs_warnings)
target_compile_definitions(xpu-serve PRIVATE XS_VERSION="${PROJECT_VERSION}")
