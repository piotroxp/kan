#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "audio/audio_buffer.hpp"
#include "audio/feature_extraction.hpp"
#include <cmath>

using Catch::Approx;

TEST_CASE("Mel-spectrogram extraction", "[audio][features]") {
    SECTION("Create mel-spectrogram extractor") {
        SincKANMelSpectrogram extractor(80, 2048, 512, 2048, 0.0, 22050.0, 44100);
        
        // Create synthetic audio (sine wave)
        AudioBuffer audio(44100, 44100, 1);  // 1 second
        double frequency = 440.0;  // A4 note
        for (size_t i = 0; i < audio.num_samples(); ++i) {
            double t = static_cast<double>(i) / audio.sample_rate();
            audio[i] = static_cast<float>(0.5 * std::sin(2.0 * M_PI * frequency * t));
        }
        
        // Extract features
        auto mel_spec = extractor.process(audio);
        
        // Check output shape
        REQUIRE(mel_spec.ndim() == 2);
        REQUIRE(mel_spec.shape()[1] == 80);  // n_mels
        
        // Check that values are finite
        for (size_t i = 0; i < mel_spec.size(); ++i) {
            REQUIRE(std::isfinite(mel_spec[i]));
        }
    }
    
    SECTION("Output shape calculation") {
        SincKANMelSpectrogram extractor(80, 2048, 512, 2048, 0.0, 22050.0, 44100);
        
        auto shape = extractor.output_shape(1.0);  // 1 second
        REQUIRE(shape.size() == 2);
        REQUIRE(shape[1] == 80);  // n_mels
        REQUIRE(shape[0] > 0);     // num_frames
    }
}


