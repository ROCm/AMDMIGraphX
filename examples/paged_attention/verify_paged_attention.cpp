/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Verify Paged Attention Output
 * 
 * This script compares outputs between paged attention and standard KV cache
 * models to verify correctness. Auto-detects model configuration from parameters.
 * 
 * Usage:
 *   ./verify_paged_attention <paged_model.mxr> <standard_model.mxr> [options]
 * 
 * Options:
 *   --seq_len N      Override sequence length (default: auto-detect from input_ids)
 *   --block_size N   Override block size (default: 16)
 */

#include <migraphx/migraphx.hpp>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <map>
#include <string>
#include <cmath>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if(err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

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

// Check if model has paged attention parameters
bool is_paged_model(const migraphx::program& prog)
{
    auto param_shapes = prog.get_parameter_shapes();
    for(const auto& name : param_shapes.names())
    {
        if(std::string(name) == "block_table" || std::string(name) == "slot_mapping")
            return true;
    }
    return false;
}

// Auto-detect configuration from model
struct DetectedConfig
{
    size_t batch_size = 1;
    size_t seq_len = 0;
    size_t max_seq_len = 0;
    size_t block_size = 16;
    size_t max_blocks = 0;
    bool has_combined_kv = false;  // block_table has shape {batch, 2, blocks}
};

DetectedConfig detect_config(const migraphx::program& prog)
{
    DetectedConfig cfg;
    auto param_shapes = prog.get_parameter_shapes();
    
    for(const auto& name_cstr : param_shapes.names())
    {
        std::string name(name_cstr);
        auto shape = param_shapes[name_cstr];
        auto lens = shape.lengths();
        
        if(name == "input_ids" && lens.size() >= 1)
        {
            if(lens.size() == 1)
            {
                cfg.seq_len = lens[0];
                cfg.batch_size = 1;
            }
            else if(lens.size() == 2)
            {
                cfg.batch_size = lens[0];
                cfg.seq_len = lens[1];
            }
        }
        else if(name == "block_table")
        {
            if(lens.size() == 2)
            {
                // Separate K/V: {batch, max_blocks}
                cfg.batch_size = lens[0];
                cfg.max_blocks = lens[1];
                cfg.has_combined_kv = false;
            }
            else if(lens.size() == 3)
            {
                // Combined K/V: {batch, 2, max_blocks}
                cfg.batch_size = lens[0];
                cfg.max_blocks = lens[2];
                cfg.has_combined_kv = true;
            }
        }
        else if(name == "slot_mapping" && lens.size() == 1)
        {
            // slot_mapping size = seq_len for prefill
            if(cfg.seq_len == 0)
                cfg.seq_len = lens[0];
        }
        else if(name == "attention_mask" && lens.size() >= 1)
        {
            cfg.max_seq_len = lens[lens.size() - 1];
        }
        else if(name.find("past_key") != std::string::npos && lens.size() >= 3)
        {
            // past_key_values: {batch, heads, max_seq, head_dim}
            cfg.max_seq_len = lens[lens.size() - 2];
        }
    }
    
    // Estimate block_size from max_blocks and max_seq_len
    if(cfg.max_blocks > 0 && cfg.max_seq_len > 0)
    {
        cfg.block_size = (cfg.max_seq_len + cfg.max_blocks - 1) / cfg.max_blocks;
    }
    
    return cfg;
}

// Get slot mapping for prefill
std::vector<int32_t> get_slot_mapping(size_t block_size, size_t num_tokens)
{
    std::vector<int32_t> slots;
    slots.reserve(num_tokens);
    
    for(size_t i = 0; i < num_tokens; ++i)
    {
        size_t block_idx = i / block_size;
        size_t slot_in_block = i % block_size;
        int32_t slot = block_idx * block_size + slot_in_block;
        slots.push_back(slot);
    }
    
    return slots;
}

// Get block table (separate format): {batch, max_blocks}
std::vector<int32_t> get_block_table_separate(size_t batch_size, size_t max_blocks)
{
    std::vector<int32_t> table;
    table.reserve(batch_size * max_blocks);
    
    for(size_t b = 0; b < batch_size; ++b)
    {
        for(size_t i = 0; i < max_blocks; ++i)
        {
            table.push_back(static_cast<int32_t>(b * max_blocks + i));
        }
    }
    
    return table;
}

// Get combined block table: {batch, 2, max_blocks}
std::vector<int32_t> get_block_table_combined(size_t batch_size, size_t max_blocks)
{
    std::vector<int32_t> combined;
    combined.reserve(batch_size * 2 * max_blocks);
    
    for(size_t b = 0; b < batch_size; ++b)
    {
        // K blocks
        for(size_t i = 0; i < max_blocks; ++i)
            combined.push_back(static_cast<int32_t>(b * max_blocks + i));
        // V blocks (same as K)
        for(size_t i = 0; i < max_blocks; ++i)
            combined.push_back(static_cast<int32_t>(b * max_blocks + i));
    }
    
    return combined;
}

// Compare results structure
struct CompareResult
{
    bool passed;
    double max_abs_diff;
    double max_rel_diff;
    double avg_abs_diff;
    size_t num_elements;
    size_t num_mismatches;
};

CompareResult compare_outputs(const std::vector<float>& a, const std::vector<float>& b, 
                              float atol, float rtol)
{
    CompareResult result;
    result.num_elements = std::min(a.size(), b.size());
    result.max_abs_diff = 0;
    result.max_rel_diff = 0;
    result.avg_abs_diff = 0;
    result.num_mismatches = 0;
    
    for(size_t i = 0; i < result.num_elements; ++i)
    {
        float abs_diff = std::abs(a[i] - b[i]);
        float rel_diff = abs_diff / (std::abs(b[i]) + 1e-8f);
        
        result.max_abs_diff = std::max(result.max_abs_diff, (double)abs_diff);
        result.max_rel_diff = std::max(result.max_rel_diff, (double)rel_diff);
        result.avg_abs_diff += abs_diff;
        
        if(abs_diff > atol && rel_diff > rtol)
            result.num_mismatches++;
    }
    
    result.avg_abs_diff /= result.num_elements;
    result.passed = (result.num_mismatches == 0);
    
    return result;
}

// Run inference and return output as float vector
std::vector<float> run_inference(
    migraphx::program& prog,
    const DetectedConfig& cfg,
    size_t block_size_override)
{
    auto param_shapes = prog.get_parameter_shapes();
    std::map<std::string, GpuBuffer> gpu_buffers;
    
    bool is_paged = is_paged_model(prog);
    size_t block_size = block_size_override > 0 ? block_size_override : cfg.block_size;
    
    // Prepare paged inputs
    std::vector<int32_t> block_table_host;
    std::vector<int32_t> slot_mapping_host;
    
    if(is_paged)
    {
        slot_mapping_host = get_slot_mapping(block_size, cfg.seq_len);
        
        if(cfg.has_combined_kv)
            block_table_host = get_block_table_combined(cfg.batch_size, cfg.max_blocks);
        else
            block_table_host = get_block_table_separate(cfg.batch_size, cfg.max_blocks);
    }
    
    // Random generator with fixed seed
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> token_dist(0, 1000);
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);
    
    // Allocate GPU buffers
    for(const auto& name_cstr : param_shapes.names())
    {
        std::string name(name_cstr);
        auto shape = param_shapes[name_cstr];
        auto lens = shape.lengths();
        size_t num_bytes = shape.bytes();
        size_t num_elements = 1;
        for(auto l : lens) num_elements *= l;
        
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
            std::vector<int64_t> data(num_elements);
            rng.seed(42);  // Reset seed for consistency
            for(auto& v : data) v = token_dist(rng);
            gpu_buffers[name].copy_from_host(data.data(), data.size() * sizeof(int64_t));
        }
        else if(name == "attention_mask")
        {
            std::vector<int64_t> data(num_elements, 0);
            // Set attention mask for seq_len tokens
            size_t mask_len = std::min(cfg.seq_len, num_elements);
            for(size_t i = 0; i < mask_len; ++i)
                data[i] = 1;
            gpu_buffers[name].copy_from_host(data.data(), data.size() * sizeof(int64_t));
        }
        else if(name.find("#output") == std::string::npos)
        {
            // Other inputs - zero or small random values
            // Keep zeros for KV cache
        }
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
    
    // Run inference
    auto outputs = prog.eval(prog_params);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy output to host
    auto output_shape = outputs[0].get_shape();
    size_t num_elements = 1;
    for(auto len : output_shape.lengths())
        num_elements *= len;
    
    std::vector<float> output_host(num_elements);
    
    auto type = output_shape.type();
    if(type == migraphx_shape_float_type)
    {
        HIP_CHECK(hipMemcpy(output_host.data(), outputs[0].data(), 
                           num_elements * sizeof(float), hipMemcpyDeviceToHost));
    }
    else if(type == migraphx_shape_half_type)
    {
        std::vector<uint16_t> half_data(num_elements);
        HIP_CHECK(hipMemcpy(half_data.data(), outputs[0].data(), 
                           num_elements * sizeof(uint16_t), hipMemcpyDeviceToHost));
        
        for(size_t i = 0; i < num_elements; ++i)
        {
            uint16_t h = half_data[i];
            uint32_t sign = (h >> 15) & 0x1;
            uint32_t exp = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            
            uint32_t f;
            if(exp == 0)
            {
                if(mant == 0)
                    f = sign << 31;
                else
                {
                    exp = 1;
                    while((mant & 0x400) == 0) { mant <<= 1; exp--; }
                    mant &= 0x3ff;
                    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
            }
            else if(exp == 31)
            {
                f = (sign << 31) | 0x7f800000 | (mant << 13);
            }
            else
            {
                f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            }
            
            memcpy(&output_host[i], &f, sizeof(float));
        }
    }
    else
    {
        HIP_CHECK(hipMemcpy(output_host.data(), outputs[0].data(), 
                           num_elements * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    return output_host;
}

void print_usage(const char* prog_name)
{
    std::cout << "Usage: " << prog_name << " <paged_model.mxr> <standard_model.mxr> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --block_size N   Override block size (default: auto-detect)" << std::endl;
    std::cout << std::endl;
    std::cout << "Compares outputs between paged and standard models to verify correctness." << std::endl;
    std::cout << "Model configuration is auto-detected from parameter shapes." << std::endl;
}

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string paged_path = argv[1];
    std::string standard_path = argv[2];
    size_t block_size_override = 0;
    
    // Parse options
    for(int i = 3; i < argc; ++i)
    {
        if(std::string(argv[i]) == "--block_size" && i + 1 < argc)
        {
            block_size_override = std::stoul(argv[++i]);
        }
    }
    
    std::cout << "=== Paged Attention Verification ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Paged model: " << paged_path << std::endl;
    std::cout << "Standard model: " << standard_path << std::endl;
    std::cout << std::endl;
    
    try
    {
        // Load models
        std::cout << "Loading paged model..." << std::endl;
        migraphx::program paged_prog;
        migraphx::file_options file_opts;
        paged_prog = migraphx::load(paged_path.c_str(), file_opts);
        
        std::cout << "Loading standard model..." << std::endl;
        migraphx::program standard_prog;
        standard_prog = migraphx::load(standard_path.c_str(), file_opts);
        
        // Detect configurations
        auto paged_cfg = detect_config(paged_prog);
        auto standard_cfg = detect_config(standard_prog);
        
        bool paged_is_paged = is_paged_model(paged_prog);
        bool standard_is_paged = is_paged_model(standard_prog);
        
        std::cout << std::endl;
        std::cout << "=== Detected Configuration ===" << std::endl;
        std::cout << std::endl;
        std::cout << "Paged model:" << std::endl;
        std::cout << "  Type: " << (paged_is_paged ? "Paged" : "Standard") << std::endl;
        std::cout << "  Batch size: " << paged_cfg.batch_size << std::endl;
        std::cout << "  Seq len: " << paged_cfg.seq_len << std::endl;
        std::cout << "  Max seq len: " << paged_cfg.max_seq_len << std::endl;
        if(paged_is_paged)
        {
            std::cout << "  Block size: " << (block_size_override > 0 ? block_size_override : paged_cfg.block_size) << std::endl;
            std::cout << "  Max blocks: " << paged_cfg.max_blocks << std::endl;
            std::cout << "  Combined KV: " << (paged_cfg.has_combined_kv ? "yes" : "no") << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "Standard model:" << std::endl;
        std::cout << "  Type: " << (standard_is_paged ? "Paged" : "Standard") << std::endl;
        std::cout << "  Batch size: " << standard_cfg.batch_size << std::endl;
        std::cout << "  Seq len: " << standard_cfg.seq_len << std::endl;
        std::cout << "  Max seq len: " << standard_cfg.max_seq_len << std::endl;
        
        // Show model parameters
        std::cout << std::endl;
        std::cout << "Paged model parameters:" << std::endl;
        auto paged_params = paged_prog.get_parameter_shapes();
        for(const auto& name : paged_params.names())
        {
            auto lens = paged_params[name].lengths();
            std::cout << "  " << name << ": [";
            for(size_t i = 0; i < lens.size(); ++i)
            {
                std::cout << lens[i];
                if(i < lens.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "Standard model parameters:" << std::endl;
        auto std_params = standard_prog.get_parameter_shapes();
        for(const auto& name : std_params.names())
        {
            auto lens = std_params[name].lengths();
            std::cout << "  " << name << ": [";
            for(size_t i = 0; i < lens.size(); ++i)
            {
                std::cout << lens[i];
                if(i < lens.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Run both models
        std::cout << std::endl;
        std::cout << "Running paged model..." << std::endl;
        auto paged_output = run_inference(paged_prog, paged_cfg, block_size_override);
        
        std::cout << "Running standard model..." << std::endl;
        auto standard_output = run_inference(standard_prog, standard_cfg, block_size_override);
        
        // Compare outputs
        std::cout << std::endl;
        std::cout << "=== Comparing Outputs ===" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Paged output size: " << paged_output.size() << " elements" << std::endl;
        std::cout << "Standard output size: " << standard_output.size() << " elements" << std::endl;
        
        if(paged_output.size() != standard_output.size())
        {
            std::cout << std::endl;
            std::cout << "WARNING: Output sizes don't match!" << std::endl;
        }
        
        size_t compare_size = std::min(paged_output.size(), standard_output.size());
        if(compare_size == 0)
        {
            std::cout << "ERROR: No elements to compare!" << std::endl;
            return 1;
        }
        
        // Compare with different tolerances
        std::cout << std::endl;
        std::cout << "Tolerance tests:" << std::endl;
        
        auto result_tight = compare_outputs(paged_output, standard_output, 1e-5f, 1e-4f);
        auto result_medium = compare_outputs(paged_output, standard_output, 1e-4f, 1e-3f);
        auto result_loose = compare_outputs(paged_output, standard_output, 1e-3f, 1e-2f);
        auto result_very_loose = compare_outputs(paged_output, standard_output, 1e-2f, 1e-1f);
        
        std::cout << "  Tight      (atol=1e-5, rtol=1e-4): " 
                  << (result_tight.passed ? "PASS" : "FAIL") 
                  << " (" << result_tight.num_mismatches << " mismatches)" << std::endl;
        std::cout << "  Medium     (atol=1e-4, rtol=1e-3): " 
                  << (result_medium.passed ? "PASS" : "FAIL")
                  << " (" << result_medium.num_mismatches << " mismatches)" << std::endl;
        std::cout << "  Loose      (atol=1e-3, rtol=1e-2): " 
                  << (result_loose.passed ? "PASS" : "FAIL")
                  << " (" << result_loose.num_mismatches << " mismatches)" << std::endl;
        std::cout << "  Very Loose (atol=1e-2, rtol=1e-1): " 
                  << (result_very_loose.passed ? "PASS" : "FAIL")
                  << " (" << result_very_loose.num_mismatches << " mismatches)" << std::endl;
        
        std::cout << std::endl;
        std::cout << "Statistics:" << std::endl;
        std::cout << "  Max absolute difference: " << std::scientific << std::setprecision(6) 
                  << result_medium.max_abs_diff << std::endl;
        std::cout << "  Max relative difference: " << std::scientific << std::setprecision(6) 
                  << result_medium.max_rel_diff << std::endl;
        std::cout << "  Avg absolute difference: " << std::scientific << std::setprecision(6) 
                  << result_medium.avg_abs_diff << std::endl;
        
        // Show sample values
        std::cout << std::endl;
        std::cout << "Sample values (first 20 elements):" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  " << std::setw(12) << "Paged" << std::setw(12) << "Standard" 
                  << std::setw(12) << "Diff" << std::endl;
        for(size_t i = 0; i < std::min(size_t(20), compare_size); ++i)
        {
            float diff = paged_output[i] - standard_output[i];
            std::cout << "  " << std::setw(12) << paged_output[i] 
                      << std::setw(12) << standard_output[i]
                      << std::setw(12) << diff << std::endl;
        }
        
        // Final verdict
        std::cout << std::endl;
        std::cout << "=== Result ===" << std::endl;
        if(result_medium.passed)
        {
            std::cout << "PASS: Paged attention output matches standard KV cache" << std::endl;
            return 0;
        }
        else if(result_loose.passed)
        {
            std::cout << "WARN: Outputs match with loose tolerance only" << std::endl;
            return 0;
        }
        else if(result_very_loose.passed)
        {
            std::cout << "WARN: Outputs match with very loose tolerance only" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "FAIL: Outputs differ significantly" << std::endl;
            return 1;
        }
        
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
