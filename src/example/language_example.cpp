#include <iostream>
#include "model/language_model.hpp"
#include "core/tensor.hpp"

int main() {
    std::cout << "=== Language Model Example ===\n\n";
    
    // 1. Create language model
    std::cout << "1. Creating MultilingualLanguageModel...\n";
    MultilingualLanguageModel lang_model(256, 10000, 512, 2);
    std::cout << "   Model created\n\n";
    
    // 2. Create semantic representation (from speech model)
    std::cout << "2. Creating semantic representation...\n";
    Tensor semantic_repr({256});
    for (size_t i = 0; i < 256; ++i) {
        semantic_repr[i] = 0.1 * (i % 10);
    }
    std::cout << "   Semantic dim: " << semantic_repr.shape()[0] << "\n\n";
    
    // 3. Generate text tokens
    std::cout << "3. Generating text tokens...\n";
    Tensor logits = lang_model.generate(semantic_repr, LanguageCode::ENGLISH);
    std::cout << "   Generated logits shape: [" << logits.shape()[0] << "]\n";
    std::cout << "   Vocabulary size: " << logits.shape()[0] << "\n\n";
    
    // 4. Generate sequence
    std::cout << "4. Generating token sequence...\n";
    auto sequence = lang_model.generate_sequence(semantic_repr, 20, LanguageCode::ENGLISH);
    std::cout << "   Generated " << sequence.size() << " tokens\n";
    std::cout << "   First 10 tokens: ";
    for (size_t i = 0; i < sequence.size() && i < 10; ++i) {
        std::cout << sequence[i] << " ";
    }
    std::cout << "\n\n";
    
    // 5. Test tokenizer
    std::cout << "5. Testing tokenizer...\n";
    Tokenizer tokenizer;
    std::string text = "Hello, world!";
    auto tokens = tokenizer.encode(text);
    std::cout << "   Encoded: \"" << text << "\" -> " << tokens.size() << " tokens\n";
    
    std::string decoded = tokenizer.decode(tokens);
    std::cout << "   Decoded: \"" << decoded << "\"\n\n";
    
    std::cout << "=== Language model example completed successfully! ===\n";
    
    return 0;
}


