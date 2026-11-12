#pragma once

// A question's probabilities as the System One API answers it: the option chosen or the expected level, and how
// settled the distribution is.

#include <span>
#include <vector>

#include "models/decision.h"

namespace xs {

struct DecisionAnswer {
    QuestionType type = QuestionType::Choice;
    std::vector<double> probabilities;  // one per option, in DecisionQuestion::options order
    size_t best = 0;        // the most likely option (the first of equals): a choice's answer, a score's mode
    double noul = 0;        // noul: the probability of true
    double score = 0;       // score: the expected level
    double confidence = 0;  // choice and score, 0 to 1
};

// The share of the top option above an even spread: (max p - 1 / n) / (1 - 1 / n); 1 for a single option.
double choice_confidence(std::span<const double> p);
// 1 - (the mean distance from the most likely level) / (that distance for an even spread, taken from the middle
// level), not below 0: probability on a neighbouring level costs less than probability far away.
double score_confidence(std::span<const double> p);

// `probabilities` must be finite, not negative and sum to more than 0; they are normalized.
DecisionAnswer make_answer(QuestionType type, std::vector<double> probabilities);

}  // namespace xs
