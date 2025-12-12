#include "rocm_manager.hpp"

// Define static members for global memory tracking
size_t ROCmMemoryManager::total_memory_ = 0;
size_t ROCmMemoryManager::memory_limit_ = 0;
std::atomic<size_t> ROCmMemoryManager::allocated_memory_(0);
std::mutex ROCmMemoryManager::memory_mutex_;
std::unordered_map<void*, size_t> ROCmMemoryManager::allocation_sizes_;

