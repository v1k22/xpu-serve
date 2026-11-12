// Answers from probabilities (decision/answers.h): the confidence formulas of the System One API and what each
// question type reports.

#include <doctest/doctest.h>

#include "decision/answers.h"

TEST_SUITE("decision answers") {
    TEST_CASE("choice confidence is the top option's share above an even spread") {
        CHECK(xs::choice_confidence(std::vector<double>{1.0, 0.0, 0.0}) == doctest::Approx(1.0));
        CHECK(xs::choice_confidence(std::vector<double>{0.25, 0.25, 0.25, 0.25}) == doctest::Approx(0.0));
        CHECK(xs::choice_confidence(std::vector<double>{0.8, 0.1, 0.1}) ==
              doctest::Approx((0.8 - 1.0 / 3) / (1 - 1.0 / 3)));
        CHECK(xs::choice_confidence(std::vector<double>{1.0}) == doctest::Approx(1.0));
    }

    TEST_CASE("score confidence weighs probability by its distance from the most likely level") {
        // The API's own example: (0, 0.57, 0.43) over three levels is 1 - 0.43 / (2 / 3).
        CHECK(xs::score_confidence(std::vector<double>{0.0, 0.57, 0.43}) == doctest::Approx(1 - 0.43 / (2.0 / 3)));
        CHECK(xs::score_confidence(std::vector<double>{0.0, 0.0, 1.0}) == doctest::Approx(1.0));
        // Far from the peak costs more than next to it; never below 0.
        CHECK(xs::score_confidence(std::vector<double>{0.6, 0.4, 0.0, 0.0, 0.0}) >
              xs::score_confidence(std::vector<double>{0.6, 0.0, 0.0, 0.0, 0.4}));
        CHECK(xs::score_confidence(std::vector<double>{0.5, 0.0, 0.0, 0.0, 0.5}) == doctest::Approx(0.0));
    }

    TEST_CASE("each question type reports its own answer") {
        const xs::DecisionAnswer noul = xs::make_answer(xs::QuestionType::Noul, {0.9, 0.1});
        CHECK(noul.noul == doctest::Approx(0.9));
        const xs::DecisionAnswer choice = xs::make_answer(xs::QuestionType::Choice, {0.2, 0.5, 0.3});
        CHECK(choice.best == 1);
        CHECK(choice.confidence == doctest::Approx((0.5 - 1.0 / 3) / (1 - 1.0 / 3)));
        const xs::DecisionAnswer score = xs::make_answer(xs::QuestionType::Score, {0.1, 0.1, 0.8});
        CHECK(score.score == doctest::Approx(1.7));
        CHECK(score.best == 2);
        // Equal options: the first one; probabilities that do not sum to 1 are normalized.
        const xs::DecisionAnswer tie = xs::make_answer(xs::QuestionType::Choice, {2.0, 2.0});
        CHECK(tie.best == 0);
        CHECK(tie.probabilities[1] == doctest::Approx(0.5));
        CHECK_THROWS_AS(xs::make_answer(xs::QuestionType::Choice, {0.0, 0.0}), xs::Error);
        CHECK_THROWS_AS(xs::make_answer(xs::QuestionType::Noul, {1.0}), xs::Error);
    }
}
