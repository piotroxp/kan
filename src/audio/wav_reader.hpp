#pragma once

#include "../audio/audio_buffer.hpp"
#include <string>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>

// Simple WAV file reader
// Supports PCM 16-bit, mono/stereo, 44.1 kHz (FSD50K format)
class WAVReader {
public:
    struct WAVHeader {
        char riff[4];           // "RIFF"
        uint32_t file_size;     // File size - 8
        char wave[4];           // "WAVE"
        char fmt[4];            // "fmt "
        uint32_t fmt_size;      // Format chunk size (usually 16)
        uint16_t audio_format;  // 1 = PCM
        uint16_t num_channels;  // 1 = mono, 2 = stereo
        uint32_t sample_rate;   // 44100 for FSD50K
        uint32_t byte_rate;     // sample_rate * num_channels * bits_per_sample / 8
        uint16_t block_align;   // num_channels * bits_per_sample / 8
        uint16_t bits_per_sample; // 16 for FSD50K
        char data[4];           // "data"
        uint32_t data_size;     // Size of audio data
    };
    
    // Read WAV file into AudioBuffer
    static bool read(const std::string& filename, AudioBuffer& buffer) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        WAVHeader header;
        
        // Read RIFF header
        file.read(reinterpret_cast<char*>(&header.riff), 4);
        if (std::string(header.riff, 4) != "RIFF") {
            return false;
        }
        
        file.read(reinterpret_cast<char*>(&header.file_size), 4);
        
        // Read WAVE header
        file.read(reinterpret_cast<char*>(&header.wave), 4);
        if (std::string(header.wave, 4) != "WAVE") {
            return false;
        }
        
        // Read format chunk
        file.read(reinterpret_cast<char*>(&header.fmt), 4);
        if (std::string(header.fmt, 4) != "fmt ") {
            return false;
        }
        
        file.read(reinterpret_cast<char*>(&header.fmt_size), 4);
        file.read(reinterpret_cast<char*>(&header.audio_format), 2);
        file.read(reinterpret_cast<char*>(&header.num_channels), 2);
        file.read(reinterpret_cast<char*>(&header.sample_rate), 4);
        file.read(reinterpret_cast<char*>(&header.byte_rate), 4);
        file.read(reinterpret_cast<char*>(&header.block_align), 2);
        file.read(reinterpret_cast<char*>(&header.bits_per_sample), 2);
        
        // Skip any extra format bytes
        if (header.fmt_size > 16) {
            file.seekg(header.fmt_size - 16, std::ios::cur);
        }
        
        // Find data chunk
        char chunk_id[4];
        uint32_t chunk_size;
        
        while (file.read(reinterpret_cast<char*>(chunk_id), 4)) {
            file.read(reinterpret_cast<char*>(&chunk_size), 4);
            
            if (std::string(chunk_id, 4) == "data") {
                // Read audio data
                if (header.bits_per_sample == 16 && header.audio_format == 1) {
                    // PCM 16-bit
                    size_t num_samples = chunk_size / (header.num_channels * 2);
                    std::vector<int16_t> pcm_data(num_samples * header.num_channels);
                    file.read(reinterpret_cast<char*>(pcm_data.data()), chunk_size);
                    
                    // Convert to AudioBuffer
                    if (header.num_channels == 1) {
                        // Mono - direct conversion
                        buffer.load_from_pcm16(pcm_data.data(), num_samples, 
                                             header.sample_rate, 1);
                    } else if (header.num_channels == 2) {
                        // Stereo - convert to mono (average channels)
                        std::vector<int16_t> mono_data(num_samples);
                        for (size_t i = 0; i < num_samples; ++i) {
                            mono_data[i] = (pcm_data[i * 2] + pcm_data[i * 2 + 1]) / 2;
                        }
                        buffer.load_from_pcm16(mono_data.data(), num_samples,
                                             header.sample_rate, 1);
                    }
                    
                    return true;
                } else {
                    // Unsupported format
                    return false;
                }
            } else {
                // Skip this chunk
                file.seekg(chunk_size, std::ios::cur);
            }
        }
        
        return false;
    }
};



