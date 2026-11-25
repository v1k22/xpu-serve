#include "backend/backend.h"

#include <map>
#include <set>

#include "util/error.h"

namespace xs {

void Session::run_segments(size_t program, std::span<const Segment> segments, const TensorMap& inputs,
                           TensorMap& outputs, std::span<const std::string> only) {
    XS_CHECK(!segments.empty(), "a step of no segment");
    std::set<int64_t> seen;
    int64_t rows = 0;
    for (const Segment& s : segments) {
        XS_CHECK(s.tokens > 0 && seen.insert(s.sequence).second, "segments must each be tokens of another sequence");
        rows += s.tokens;
    }
    for (const auto& [name, t] : inputs) {
        XS_CHECK(t.rank() >= 1 && t.dim(0) == rows, "input '", name, "' has ", t.rank() >= 1 ? t.dim(0) : 0,
                 " rows; the segments have ", rows);
    }
    const int64_t current = current_sequence();
    std::vector<std::pair<int64_t, int64_t>> done;  // (sequence, tokens cached before its step)
    std::map<std::string, std::vector<HostTensor>> parts;
    try {
        int64_t first = 0;
        for (const Segment& s : segments) {
            use(s.sequence);
            TensorMap in, out;
            for (const auto& [name, t] : inputs) in[name] = rows_of(t, first, s.tokens);
            const int64_t before = num_cached(program);
            run(program, in, out, only);
            done.emplace_back(s.sequence, before);
            for (auto& [name, t] : out) parts[name].push_back(std::move(t));
            first += s.tokens;
        }
    } catch (...) {
        for (const auto& [sequence, before] : done) {  // the last step of each is undone whole
            use(sequence);
            truncate(program, before);
        }
        use(current);
        throw;
    }
    use(current);
    for (auto& [name, list] : parts) outputs[name] = concat_rows(list);
}

}  // namespace xs
