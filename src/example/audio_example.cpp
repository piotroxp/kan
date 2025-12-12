#include <iostream>
#include "core/tensor.hpp"
#include "audio/audio_buffer.hpp"
#include "audio/preprocessing.hpp"
#include "audio/feature_extraction.hpp"
#include "quantum/quantum_field_embedding.hpp"

int main() {
    std::cout << "=== Audio Processing Pipeline Example ===\n\n";
    
    // 1. Create synthetic audio (sine wave at 440 Hz)
    std::cout << "1. Creating synthetic audio (440 Hz sine wave)...\n";
    AudioBuffer audio(44100, 44100, 1);  // 1 second at 44.1 kHz
    double frequency = 440.0;  // A4 note
    for (size_t i = 0; i < audio.num_samples(); ++i) {
        double t = static_cast<double>(i) / audio.sample_rate();
        audio[i] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * frequency * t));
    }
    std::cout << "   Audio duration: " << audio.duration() << " seconds\n";
    std::cout << "   Sample rate: " << audio.sample_rate() << " Hz\n";
    std::cout << "   Num samples: " << audio.num_samples() << "\n\n";
    
    // 2. Preprocess audio
    std::cout << "2. Preprocessing audio...\n";
    AudioPreprocessor::normalize(audio);
    std::cout << "   Normalized audio\n";
    
    // Apply augmentation
    AudioBuffer augmented = AudioPreprocessor::augment(audio, 0.1, 1.0, 0.01);
    std::cout << "   Augmented audio duration: " << augmented.duration() << " seconds\n\n";
    
    // 3. Extract mel-spectrogram features
    std::cout << "3. Extracting mel-spectrogram features...\n";
    SincKANMelSpectrogram extractor(80, 2048, 512, 2048, 0.0, 22050.0, 44100);
    Tensor mel_spec = extractor.process(audio);
    
    std::cout << "   Mel-spectrogram shape: [" 
              << mel_spec.shape()[0] << ", " << mel_spec.shape()[1] << "]\n";
    std::cout << "   Time frames: " << mel_spec.shape()[0] << "\n";
    std::cout << "   Mel bins: " << mel_spec.shape()[1] << "\n";
    std::cout << "   Mean value: " << mel_spec.mean() << "\n\n";
    
    // 4. Encode to quantum embeddings
    std::cout << "4. Encoding to quantum embeddings...\n";
    
    // Flatten mel-spectrogram for embedding (use mean pooling for simplicity)
    Tensor pooled({static_cast<size_t>(mel_spec.shape()[1])});
    pooled.fill(0.0);
    for (size_t t = 0; t < mel_spec.shape()[0]; ++t) {
        for (size_t m = 0; m < mel_spec.shape()[1]; ++m) {
            pooled[m] += mel_spec.at({t, m});
        }
    }
    for (size_t m = 0; m < pooled.size(); ++m) {
        pooled[m] /= mel_spec.shape()[0];
    }
    
    QuantumFieldEmbeddingCore embedding(
        static_cast<int>(pooled.size()),  // input_dim
        1,                                // embedding_dim
        1024,                             // grid_size
        12.0,                             // L
        1.5                               // sigma
    );
    
    auto wavefunctions = embedding.encode(pooled);
    std::cout << "   Encoded to " << wavefunctions.size() << " wavefunction(s)\n";
    std::cout << "   Wavefunction size: " << wavefunctions[0].size() << "\n";
    
    // Test similarity
    Tensor pooled2 = pooled;
    for (size_t i = 0; i < pooled2.size(); ++i) {
        pooled2[i] += 0.01;  // Slight variation
    }
    auto wavefunctions2 = embedding.encode(pooled2);
    double similarity = embedding.similarity(wavefunctions[0], wavefunctions2[0]);
    std::cout << "   Similarity between similar features: " << similarity << "\n\n";
    
    std::cout << "=== Audio pipeline completed successfully! ===\n";
    
    return 0;
}

