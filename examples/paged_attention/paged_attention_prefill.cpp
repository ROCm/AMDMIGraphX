/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Paged Attention Prefill Example
 * 
 * This example runs prefill inference on a pre-compiled model and reports timing.
 * Supports both paged attention (combined KV format) and standard KV cache models.
 * 
 * Usage:
 *   ./paged_attention_prefill <model.mxr> <prefill_len>
 */

#include <migraphx/migraphx.hpp>
#include <hip/hip_runtime_api.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <map>
#include <string>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if(err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

// Configuration
struct Config
{
    size_t block_size = 16;
    size_t max_seq_len = 4096;
    size_t batch_size = 1;
    
    size_t max_blocks_per_seq() const 
    { 
        return (max_seq_len + block_size - 1) / block_size; 
    }
};

// RAII wrapper for GPU memory
class GpuBuffer
{
public:
    GpuBuffer() : ptr_(nullptr), bytes_(0) {}
    
    GpuBuffer(size_t bytes) : bytes_(bytes)
    {
        if(bytes > 0)
        {
            HIP_CHECK(hipMalloc(&ptr_, bytes));
            HIP_CHECK(hipMemset(ptr_, 0, bytes));
        }
    }
    
    ~GpuBuffer()
    {
        if(ptr_) (void)hipFree(ptr_);
    }
    
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    
    GpuBuffer(GpuBuffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_)
    {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    
    GpuBuffer& operator=(GpuBuffer&& other) noexcept
    {
        if(this != &other)
        {
            if(ptr_) (void)hipFree(ptr_);
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }
    
    void* get() const { return ptr_; }
    size_t bytes() const { return bytes_; }
    
    void copy_from_host(const void* host_data, size_t size)
    {
        HIP_CHECK(hipMemcpy(ptr_, host_data, size, hipMemcpyHostToDevice));
    }
    
private:
    void* ptr_;
    size_t bytes_;
};

// Get slot mapping for prefill
std::vector<int32_t> get_slot_mapping(const Config& config, size_t num_tokens)
{
    std::vector<int32_t> slots;
    slots.reserve(num_tokens);
    
    for(size_t i = 0; i < num_tokens; ++i)
    {
        size_t block_idx = i / config.block_size;
        size_t slot_in_block = i % config.block_size;
        int32_t slot = block_idx * config.block_size + slot_in_block;
        slots.push_back(slot);
    }
    
    return slots;
}

// Get combined block table: {batch, 2, max_blocks}
std::vector<int32_t> get_block_table_combined(const Config& config)
{
    std::vector<int32_t> combined;
    size_t max_blocks = config.max_blocks_per_seq();
    combined.reserve(config.batch_size * 2 * max_blocks);
    
    for(size_t b = 0; b < config.batch_size; ++b)
    {
        // K blocks
        for(size_t i = 0; i < max_blocks; ++i)
        {
            combined.push_back(static_cast<int32_t>(b * max_blocks + i));
        }
        // V blocks (same as K)
        for(size_t i = 0; i < max_blocks; ++i)
        {
            combined.push_back(static_cast<int32_t>(b * max_blocks + i));
        }
    }
    
    return combined;
}

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model.mxr> <prefill_len>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    size_t prefill_len = std::stoul(argv[2]);
    
    Config config;
    config.block_size = 16;
    config.max_seq_len = 4096;
    config.batch_size = 1;
    
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Prefill length: " << prefill_len << std::endl;
    std::cout << std::endl;
    
    std::map<std::string, GpuBuffer> gpu_buffers;
    
    try
    {
        // Load model
        migraphx::program prog;
        migraphx::file_options file_opts;
        prog = migraphx::load(model_path.c_str(), file_opts);
        
        auto param_shapes = prog.get_parameter_shapes();
        
        // Check if paged
        bool is_paged = false;
        for(const auto& name : param_shapes.names())
        {
            if(std::string(name) == "block_table" || std::string(name) == "slot_mapping")
            {
                is_paged = true;
                break;
            }
        }
        
        std::cout << "Model type: " << (is_paged ? "Paged (combined KV)" : "Standard") << std::endl;
        std::cout << std::endl;
        
        // Prepare paged inputs if needed
        std::vector<int32_t> block_table_host;
        std::vector<int32_t> slot_mapping_host;
        if(is_paged)
        {
            block_table_host = get_block_table_combined(config);
            slot_mapping_host = get_slot_mapping(config, prefill_len);
        }
        
        // Prepare common inputs
        std::mt19937 rng(42);
        std::uniform_int_distribution<int64_t> token_dist(0, 151936);
        
        std::vector<int64_t> input_ids_host(config.batch_size * prefill_len);
        for(auto& id : input_ids_host) id = token_dist(rng);
        
        std::vector<int64_t> attention_mask_host(config.batch_size * config.max_seq_len, 0);
        for(size_t i = 0; i < prefill_len; ++i)
            attention_mask_host[i] = 1;
        
        // Allocate and initialize GPU buffers
        for(const auto& name_cstr : param_shapes.names())
        {
            std::string name(name_cstr);
            auto shape = param_shapes[name_cstr];
            size_t num_bytes = shape.bytes();
            
            gpu_buffers[name] = GpuBuffer(num_bytes);
            
            if(name == "block_table" && is_paged)
            {
                gpu_buffers[name].copy_from_host(block_table_host.data(), 
                                                  block_table_host.size() * sizeof(int32_t));
            }
            else if(name == "slot_mapping" && is_paged)
            {
                gpu_buffers[name].copy_from_host(slot_mapping_host.data(), 
                                                  slot_mapping_host.size() * sizeof(int32_t));
            }
            else if(name == "input_ids")
            {
                gpu_buffers[name].copy_from_host(input_ids_host.data(), 
                                                  input_ids_host.size() * sizeof(int64_t));
            }
            else if(name == "attention_mask")
            {
                gpu_buffers[name].copy_from_host(attention_mask_host.data(), 
                                                  attention_mask_host.size() * sizeof(int64_t));
            }
            // Other buffers stay zero-initialized
        }
        
        // Setup program parameters
        migraphx::program_parameters prog_params;
        for(const auto& name_cstr : param_shapes.names())
        {
            std::string name(name_cstr);
            auto shape = param_shapes[name_cstr];
            if(gpu_buffers.count(name))
            {
                prog_params.add(name_cstr, migraphx::argument(shape, gpu_buffers[name].get()));
            }
        }
        
        // Warmup
        for(int i = 0; i < 2; ++i)
        {
            prog.eval(prog_params);
            HIP_CHECK(hipDeviceSynchronize());
        }
        
        // Benchmark
        const int num_runs = 5;
        std::vector<double> times;
        
        for(int i = 0; i < num_runs; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            prog.eval(prog_params);
            HIP_CHECK(hipDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            times.push_back(ms);
        }
        
        // Calculate stats
        std::sort(times.begin(), times.end());
        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min_time = times.front();
        double max_time = times.back();
        
        // Report
        std::cout << "=== Results ===" << std::endl;
        std::cout << "Time (avg): " << std::fixed << std::setprecision(2) << avg << " ms" << std::endl;
        std::cout << "Time (min): " << std::fixed << std::setprecision(2) << min_time << " ms" << std::endl;
        std::cout << "Time (max): " << std::fixed << std::setprecision(2) << max_time << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(1) << (prefill_len * 1000.0 / avg) << " tokens/sec" << std::endl;
        
        HIP_CHECK(hipDeviceSynchronize());
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
