#pragma once

// A run of launches recorded as a command graph (sycl_ext_oneapi_graph), which the device then runs with no host work
// between two of its kernels. A recording holds its kernels' arguments as they were when it was made.

#include <memory>

#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/sycl.hpp>

namespace xs::sycl_be {

using Recording =
    sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;

// What a recording cannot hold and a recording queue does not refuse by itself: memory the host fills for a launch,
// which a replay would read as some later launch left it.
struct Unrecordable {};
inline void refuse_recording(const sycl::queue& q) {
    if (q.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording) throw Unrecordable{};
}

// `launch`'s launches on `q` as a recording, none of them run. Null when a recording cannot hold them: one waits on
// the queue, which a recording queue refuses, or is Unrecordable.
template <typename Launch>
std::unique_ptr<Recording> record(sycl::queue& q, Launch&& launch) {
    namespace syclex = sycl::ext::oneapi::experimental;
    syclex::command_graph<syclex::graph_state::modifiable> graph(q.get_context(), q.get_device());
    graph.begin_recording(q);
    try {
        launch();
    } catch (const sycl::exception&) {
        graph.end_recording(q);
        return nullptr;
    } catch (const Unrecordable&) {
        graph.end_recording(q);
        return nullptr;
    } catch (...) {
        graph.end_recording(q);
        throw;
    }
    graph.end_recording(q);
    return std::make_unique<Recording>(graph.finalize());
}

inline void replay(sycl::queue& q, Recording& recording) {
    sycl::ext::oneapi::experimental::execute_graph(q, recording);
}

}  // namespace xs::sycl_be
