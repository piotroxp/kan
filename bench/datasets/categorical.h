#pragma once

#include "bench/datasets/dataset.h"

#include <random>

namespace bench::datasets {

struct CategoricalSpec {
    size_t vocab_size = 0;
    size_t active_tokens = 0;
};

DatasetSplit make_bag_of_categories(size_t vocab_size,
                                    size_t active_tokens,
                                    size_t total_samples,
                                    std::mt19937& rng);

DatasetSplit make_sequence_pattern(size_t vocab_size,
                                   size_t sequence_length,
                                   size_t total_samples,
                                   std::mt19937& rng);

}  // namespace bench::datasets
