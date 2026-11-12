#include "decision/answers.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "util/error.h"

namespace xs {

double choice_confidence(std::span<const double> p) {
    if (p.size() <= 1) return 1.0;
    const double even = 1.0 / static_cast<double>(p.size());
    return std::clamp((*std::max_element(p.begin(), p.end()) - even) / (1.0 - even), 0.0, 1.0);
}

double score_confidence(std::span<const double> p) {
    if (p.size() <= 1) return 1.0;
    const auto n = static_cast<double>(p.size());
    const auto mode = static_cast<double>(std::max_element(p.begin(), p.end()) - p.begin());
    double distance = 0, even = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        distance += p[i] * std::abs(static_cast<double>(i) - mode);
        even += std::abs(static_cast<double>(i) - (n - 1) / 2) / n;
    }
    return std::max(0.0, 1.0 - distance / even);
}

DecisionAnswer make_answer(QuestionType type, std::vector<double> probabilities) {
    XS_CHECK(!probabilities.empty(), "a question without options");
    double total = 0;
    for (const double v : probabilities) {
        XS_CHECK(std::isfinite(v) && v >= 0, "a probability of ", v);
        total += v;
    }
    XS_CHECK(total > 0, "probabilities without mass");
    for (double& v : probabilities)
        v /= total;
    DecisionAnswer a;
    a.type = type;
    a.best = static_cast<size_t>(std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
    if (type == QuestionType::Noul) {
        XS_CHECK(probabilities.size() == 2, "a noul question has ", probabilities.size(), " outcomes");
        a.noul = probabilities[0];
    } else if (type == QuestionType::Choice) {
        a.confidence = choice_confidence(probabilities);
    } else {
        for (size_t i = 0; i < probabilities.size(); ++i)
            a.score += static_cast<double>(i) * probabilities[i];
        a.confidence = score_confidence(probabilities);
    }
    a.probabilities = std::move(probabilities);
    return a;
}

}  // namespace xs
