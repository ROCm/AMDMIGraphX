/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Paged Attention Decode Example
 * 
 * This example demonstrates how to use MIGraphX with paged attention for
 * autoregressive decode. It shows:
 * - Setting up block tables for paged KV cache
 * - Managing slot mappings for token placement
 * - Running multiple decode steps
 */

#include <migraphx/migraphx.hpp>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <map>
#include <string>
#include <cstring>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if(err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

// Configuration for paged attention
struct PagedAttentionConfig
{
    size_t block_size = 16;           // Tokens per block
    size_t max_seq_len = 4096;        // Maximum sequence length
    size_t batch_size = 1;            // Batch size
    size_t num_kv_heads = 32;         // Number of KV heads (from model)
    size_t head_dim = 128;            // Head dimension (from model)
    
    size_t max_blocks_per_seq() const 
    { 
        return (max_seq_len + block_size - 1) / block_size; 
    }
    
    size_t total_blocks() const
    {
        return batch_size * max_blocks_per_seq();
    }
};

// Manages block allocation and slot mapping for paged attention
class PagedKVCacheManager
{
public:
    PagedKVCacheManager(const PagedAttentionConfig& cfg)
        : config(cfg)
    {
        // Initialize block tables for each batch
        // Each batch gets sequential blocks initially
        block_tables.resize(config.batch_size);
        for(size_t b = 0; b < config.batch_size; ++b)
        {
            block_tables[b].resize(config.max_blocks_per_seq());
            for(size_t i = 0; i < config.max_blocks_per_seq(); ++i)
            {
                // Assign sequential blocks: batch 0 gets blocks 0..N-1, etc.
                block_tables[b][i] = static_cast<int32_t>(b * config.max_blocks_per_seq() + i);
            }
        }
        
        // Track current position for each sequence in batch
        current_positions.resize(config.batch_size, 0);
    }
    
    // Prefill: Mark tokens 0..prompt_len-1 as occupied
    void prefill(size_t batch_idx, size_t prompt_len)
    {
        current_positions[batch_idx] = prompt_len;
    }
    
    // Get slot mapping for new tokens being written
    // Returns flat slot indices for scatter operation
    std::vector<int32_t> get_slot_mapping(size_t batch_idx, size_t num_new_tokens = 1)
    {
        std::vector<int32_t> slots;
        slots.reserve(num_new_tokens);
        
        for(size_t i = 0; i < num_new_tokens; ++i)
        {
            size_t token_pos = current_positions[batch_idx] + i;
            size_t block_idx = token_pos / config.block_size;
            size_t slot_in_block = token_pos % config.block_size;
            
            // Physical block from block table
            int32_t physical_block = block_tables[batch_idx][block_idx];
            // Slot = physical_block * block_size + slot_in_block
            int32_t slot = physical_block * config.block_size + slot_in_block;
            slots.push_back(slot);
        }
        
        return slots;
    }
    
    // Advance position after decode step
    void advance(size_t batch_idx, size_t num_tokens = 1)
    {
        current_positions[batch_idx] += num_tokens;
    }
    
    // Get flattened block table for the model
    std::vector<int32_t> get_block_table_flat()
    {
        std::vector<int32_t> flat;
        flat.reserve(config.batch_size * config.max_blocks_per_seq());
        for(auto& bt : block_tables)
        {
            flat.insert(flat.end(), bt.begin(), bt.end());
        }
        return flat;
    }
    
    size_t get_current_position(size_t batch_idx) const
    {
        return current_positions[batch_idx];
    }
    
private:
    PagedAttentionConfig config;
    std::vector<std::vector<int32_t>> block_tables;  // [batch][block_idx] -> physical_block
    std::vector<size_t> current_positions;           // Current sequence length per batch
};

// Helper to print tensor info
void print_tensor_info(const migraphx::shape& s, const std::string& name)
{
    std::cout << "  " << name << ": [";
    auto lens = s.lengths();
    for(size_t i = 0; i < lens.size(); ++i)
    {
        std::cout << lens[i];
        if(i < lens.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

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
            HIP_CHECK(hipMemset(ptr_, 0, bytes));  // Zero-initialize
        }
    }
    
    ~GpuBuffer()
    {
        if(ptr_)
        {
            (void)hipFree(ptr_);  // Ignore error in destructor
        }
    }
    
    // No copy
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    
    // Move
    GpuBuffer(GpuBuffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_)
    {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    
    GpuBuffer& operator=(GpuBuffer&& other) noexcept
    {
        if(this != &other)
        {
            if(ptr_) (void)hipFree(ptr_);  // Ignore error in move
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }
    
    void* get() const { return ptr_; }
    size_t bytes() const { return bytes_; }
    
    // Copy data from host to GPU
    void copy_from_host(const void* host_data, size_t size)
    {
        if(size > bytes_)
        {
            std::cerr << "Error: trying to copy " << size << " bytes into " << bytes_ << " byte buffer" << std::endl;
            std::exit(1);
        }
        HIP_CHECK(hipMemcpy(ptr_, host_data, size, hipMemcpyHostToDevice));
    }
    
private:
    void* ptr_;
    size_t bytes_;
};

int main(int argc, char** argv)
{
    // Default model path - can be overridden by command line
    std::string model_path = "group_query_attention_defaults_test.onnx";
    if(argc > 1)
    {
        model_path = argv[1];
    }
    
    std::cout << "=== Paged Attention Decode Example ===" << std::endl;
    std::cout << std::endl;
    
    // Configuration matching the test model
    PagedAttentionConfig config;
    config.block_size = 16;
    config.max_seq_len = 4096;
    config.batch_size = 1;
    config.num_kv_heads = 32;
    config.head_dim = 128;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Block size: " << config.block_size << std::endl;
    std::cout << "  Max seq len: " << config.max_seq_len << std::endl;
    std::cout << "  Max blocks/seq: " << config.max_blocks_per_seq() << std::endl;
    std::cout << "  Total blocks: " << config.total_blocks() << std::endl;
    std::cout << std::endl;
    
    // Initialize KV cache manager
    PagedKVCacheManager cache_manager(config);
    
    // GPU buffers - declared before program to ensure proper destruction order
    std::map<std::string, GpuBuffer> gpu_buffers;
    
    // Host-side data for updating GPU buffers
    std::vector<int32_t> block_table_host;
    std::vector<char> slot_mapping_host;
    
    try
    {
        // Parse ONNX model
        std::cout << "Loading model: " << model_path << std::endl;
        migraphx::program prog;
        migraphx::onnx_options onnx_opts;
        prog = migraphx::parse_onnx(model_path.c_str(), onnx_opts);
        
        // Get parameter shapes before compilation
        auto param_shapes = prog.get_parameter_shapes();
        std::cout << "\nOriginal model parameters:" << std::endl;
        for(const auto& name : param_shapes.names())
        {
            print_tensor_info(param_shapes[name], name);
        }
        
        // Compile for GPU with paged attention enabled
        std::cout << "\nCompiling for GPU..." << std::endl;
        migraphx::compile_options compile_opts;
        
        migraphx::target targ("gpu");
        prog.compile(targ, compile_opts);
        std::cout << "Program compiled" << std::endl;
        prog.print();
        
        // Get parameter shapes after compilation (may include block_table, slot_mapping)
        param_shapes = prog.get_parameter_shapes();
        std::cout << "\nCompiled model parameters:" << std::endl;
        for(const auto& name : param_shapes.names())
        {
            print_tensor_info(param_shapes[name], name);
        }
        
        // Simulate prefill: assume prompt of 100 tokens has been processed
        size_t prompt_len = 100;
        cache_manager.prefill(0, prompt_len);
        std::cout << "\n=== Starting Decode ===" << std::endl;
        std::cout << "Prefilled " << prompt_len << " tokens" << std::endl;
        
        // Random number generator for dummy inputs
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        // Block table is persistent across decode steps
        block_table_host = cache_manager.get_block_table_flat();
        
        // Allocate GPU buffers for all parameters
        for(const auto& name_cstr : param_shapes.names())
        {
            std::string name(name_cstr);
            auto shape = param_shapes[name_cstr];
            size_t num_bytes = shape.bytes();
            
            // Allocate GPU buffer
            gpu_buffers[name] = GpuBuffer(num_bytes);
            
            // Initialize with data
            if(name == "block_table")
            {
                gpu_buffers[name].copy_from_host(block_table_host.data(), 
                                                  block_table_host.size() * sizeof(int32_t));
            }
            else if(name == "slot_mapping")
            {
                // Will be updated per decode step
                slot_mapping_host.resize(num_bytes);
            }
            else if(name.find("#output") != std::string::npos)
            {
                // Output buffers - just allocated, zero-initialized by GpuBuffer
            }
            else
            {
                // Other inputs - fill with random data on host and copy
                std::vector<char> host_data(num_bytes);
                for(size_t i = 0; i < num_bytes; ++i)
                {
                    host_data[i] = static_cast<char>(rand() % 256);
                }
                gpu_buffers[name].copy_from_host(host_data.data(), num_bytes);
            }
        }
        
        std::cout << "Allocated " << gpu_buffers.size() << " GPU buffers" << std::endl;
        
        // Run a few decode steps
        const size_t num_decode_steps = 5;
        
        for(size_t step = 0; step < num_decode_steps; ++step)
        {
            std::cout << "\n--- Decode Step " << (step + 1) << " ---" << std::endl;
            std::cout << "Current position: " << cache_manager.get_current_position(0) << std::endl;
            
            // Get slot mapping for this decode step
            auto slot_mapping = cache_manager.get_slot_mapping(0, 1);  // 1 new token
            std::cout << "Slot mapping: [" << slot_mapping[0] << "]" << std::endl;
            
            // Update slot_mapping on GPU
            if(gpu_buffers.count("slot_mapping"))
            {
                gpu_buffers["slot_mapping"].copy_from_host(slot_mapping.data(), 
                                                           slot_mapping.size() * sizeof(int32_t));
            }
            
            // Prepare program parameters using GPU buffers
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
            
            // Run inference
            std::cout << "Running inference..." << std::endl;
            auto outputs = prog.eval(prog_params);
            
            // Synchronize to ensure completion
            HIP_CHECK(hipDeviceSynchronize());
            
            // Process output (in real use, this would be the next token logits)
            auto output_shape = outputs[0].get_shape();
            auto output_lens = output_shape.lengths();
            std::cout << "Output shape: [";
            for(size_t i = 0; i < output_lens.size(); ++i)
            {
                std::cout << output_lens[i];
                if(i < output_lens.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            
            // Advance position for next decode step
            cache_manager.advance(0, 1);
        }
        
        std::cout << "\n=== Decode Complete ===" << std::endl;
        std::cout << "Final position: " << cache_manager.get_current_position(0) << std::endl;
        std::cout << "Total tokens generated: " << num_decode_steps << std::endl;
        
        // Synchronize before cleanup
        HIP_CHECK(hipDeviceSynchronize());
        
    }  // prog destructor called here, releases GPU resources
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // GPU buffers are destroyed after prog, in reverse order of construction
    
    return 0;
}
