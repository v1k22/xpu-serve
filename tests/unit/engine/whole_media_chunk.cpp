// Prefill steps that cut no image (engine/media.h whole_media_chunk, ModelInfo::media_whole_steps: a model whose image
// tokens see each other both ways reads each image in one step): a chunk ending inside an image ends before it, or
// takes all of it when the image starts the chunk; text and runs that end inside the chunk pass as they are.
#include <doctest/doctest.h>

#include <vector>

#include "engine/media.h"
#include "util/error.h"

using xs::whole_media_chunk;

TEST_SUITE("engine whole media chunk") {
    TEST_CASE("a chunk never cuts an image's run of soft tokens") {
        // text 0-2, image 0 at 3-8, text 9, image 1 at 10-13, text 14-15
        const std::vector<int32_t> item = {-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, 1, 1, 1, 1, -1, -1};
        CHECK(whole_media_chunk(item, 0, 3) == 3);    // ends before the image
        CHECK(whole_media_chunk(item, 0, 9) == 9);    // ends at its end
        CHECK(whole_media_chunk(item, 0, 5) == 3);    // would cut it: ends before it
        CHECK(whole_media_chunk(item, 3, 2) == 6);    // the image starts the chunk: all of it, past the chunk size
        CHECK(whole_media_chunk(item, 9, 3) == 1);    // text, then image 1 would be cut: the text alone
        CHECK(whole_media_chunk(item, 14, 8) == 8);   // past the prompt: as it is
        CHECK(whole_media_chunk(item, 10, 4) == 4);   // the whole image
        // Two images back to back: the second's run starts where the first's ends.
        const std::vector<int32_t> pair = {0, 0, 0, 1, 1, 1};
        CHECK(whole_media_chunk(pair, 0, 4) == 3);
        CHECK(whole_media_chunk(pair, 3, 1) == 3);
        CHECK_THROWS_WITH_AS(whole_media_chunk(item, 5, 2), doctest::Contains("starts inside the soft tokens"),
                             xs::Error);
    }
}
