#include "bench/datasets/categorical.h"

#include <algorithm>
#include <unordered_set>

namespace bench::datasets {

DatasetSplit make_bag_of_categories(size_t vocab_size,
                                    size_t active_tokens,
                                    size_t total_samples,
                                    std::mt19937& rng) {
    std::uniform_int_distribution<size_t> token_dist(0, vocab_size - 1);
    std::uniform_int_distribution<size_t> hidden_dist(0, vocab_size - 1);

    std::unordered_set<size_t> hidden;
    while (hidden.size() < 8) {
        hidden.insert(hidden_dist(rng));
    }

    Dataset dataset;
    dataset.input_dim = active_tokens;
    dataset.output_dim = 2;
    dataset.classification = true;
    dataset.features.reserve(total_samples);
    dataset.targets.reserve(total_samples);

    for (size_t i = 0; i < total_samples; ++i) {
        std::vector<double> tokens;
        tokens.reserve(active_tokens);
        size_t parity = 0;
        for (size_t k = 0; k < active_tokens; ++k) {
            size_t token = token_dist(rng);
            tokens.push_back(static_cast<double>(token));
            if (hidden.find(token) != hidden.end()) {
                parity ^= 1;
            }
        }
        dataset.features.push_back(tokens);
        dataset.targets.push_back(parity == 0 ? std::vector<double>{1.0, 0.0}
                                              : std::vector<double>{0.0, 1.0});
    }

    return split_dataset(dataset, 0.7, 0.15, rng);
}

DatasetSplit make_sequence_pattern(size_t vocab_size,
                                   size_t sequence_length,
                                   size_t total_samples,
                                   std::mt19937& rng) {
    std::uniform_int_distribution<size_t> token_dist(0, vocab_size - 1);
    std::uniform_real_distribution<double> prob(0.0, 1.0);

    const size_t token_a = 11;
    const size_t token_b = 27;
    const size_t token_c = 42;

    Dataset dataset;
    dataset.input_dim = sequence_length;
    dataset.output_dim = 2;
    dataset.classification = true;
    dataset.features.reserve(total_samples);
    dataset.targets.reserve(total_samples);

    for (size_t i = 0; i < total_samples; ++i) {
        std::vector<double> sequence(sequence_length);
        for (size_t t = 0; t < sequence_length; ++t) {
            sequence[t] = static_cast<double>(token_dist(rng));
        }

        bool inject = prob(rng) < 0.5;
        if (inject && sequence_length >= 6) {
            size_t start = std::uniform_int_distribution<size_t>(0, sequence_length - 6)(rng);
            sequence[start] = static_cast<double>(token_a);
            sequence[start + 2] = static_cast<double>(token_b);
            sequence[start + 5] = static_cast<double>(token_c);
        }

        bool has_pattern = false;
        for (size_t a = 0; a + 2 < sequence_length; ++a) {
            if (static_cast<size_t>(sequence[a]) != token_a) {
                continue;
            }
            for (size_t b = a + 1; b + 2 < sequence_length; ++b) {
                if (static_cast<size_t>(sequence[b]) != token_b) {
                    continue;
                }
                for (size_t c = b + 1; c < sequence_length; ++c) {
                    if (static_cast<size_t>(sequence[c]) == token_c) {
                        has_pattern = true;
                        break;
                    }
                }
                if (has_pattern) {
                    break;
                }
            }
            if (has_pattern) {
                break;
            }
        }

        dataset.features.push_back(sequence);
        dataset.targets.push_back(has_pattern ? std::vector<double>{0.0, 1.0}
                                              : std::vector<double>{1.0, 0.0});
    }

    return split_dataset(dataset, 0.7, 0.15, rng);
}

}  // namespace bench::datasets
