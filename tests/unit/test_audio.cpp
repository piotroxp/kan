#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "audio/audio_buffer.hpp"
#include "audio/preprocessing.hpp"
#include <cmath>

using Catch::Approx;

TEST_CASE("AudioBuffer creation and operations", "[audio]") {
    SECTION("Create empty buffer") {
        AudioBuffer buffer;
        REQUIRE(buffer.num_samples() == 0);
        REQUIRE(buffer.sample_rate() == 44100);
        REQUIRE(buffer.channels() == 1);
    }
    
    SECTION("Create buffer with size") {
        AudioBuffer buffer(1000, 44100, 1);
        REQUIRE(buffer.num_samples() == 1000);
        REQUIRE(buffer.sample_rate() == 44100);
        REQUIRE(buffer.channels() == 1);
    }
    
    SECTION("Load from PCM16") {
        std::vector<int16_t> pcm_data = {32767, 0, -32768, 0};
        AudioBuffer buffer;
        buffer.load_from_pcm16(pcm_data.data(), pcm_data.size(), 44100, 1);
        
        REQUIRE(buffer.num_samples() == 4);
        REQUIRE(buffer[0] == Approx(1.0f).margin(0.0001f));
        REQUIRE(buffer[1] == Approx(0.0f));
        REQUIRE(buffer[2] == Approx(-1.0f).margin(0.0001f));
        REQUIRE(buffer[3] == Approx(0.0f));
    }
    
    SECTION("Duration calculation") {
        AudioBuffer buffer(44100, 44100, 1);  // 1 second
        REQUIRE(buffer.duration() == Approx(1.0));
    }
}

TEST_CASE("Audio preprocessing", "[audio][preprocessing]") {
    SECTION("Normalize audio") {
        AudioBuffer buffer(100);
        for (size_t i = 0; i < 100; ++i) {
            buffer[i] = 0.5f * (i % 10);
        }
        
        AudioPreprocessor::normalize(buffer);
        
        // Find max should be close to 1.0
        float max_val = 0.0f;
        for (size_t i = 0; i < 100; ++i) {
            max_val = std::max(max_val, std::abs(buffer[i]));
        }
        REQUIRE(max_val == Approx(1.0f).margin(0.01f));
    }
    
    SECTION("Pad or truncate") {
        AudioBuffer buffer(50);
        buffer.resize(50);
        for (size_t i = 0; i < 50; ++i) {
            buffer[i] = 0.1f;
        }
        
        // Pad
        AudioPreprocessor::pad_or_truncate(buffer, 100);
        REQUIRE(buffer.num_samples() == 100);
        REQUIRE(buffer[0] == Approx(0.1f));
        REQUIRE(buffer[99] == Approx(0.0f));
        
        // Truncate
        AudioPreprocessor::pad_or_truncate(buffer, 25);
        REQUIRE(buffer.num_samples() == 25);
    }
    
    SECTION("Time stretch") {
        AudioBuffer buffer(100);
        for (size_t i = 0; i < 100; ++i) {
            buffer[i] = 0.1f;
        }
        
        AudioBuffer stretched = AudioPreprocessor::time_stretch(buffer, 2.0);
        REQUIRE(stretched.num_samples() == 50);
    }
}

