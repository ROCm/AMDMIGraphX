#pragma once

#include "config.hpp"

#include <iostream>
#include <vector>
#include <filesystem>

#include <migraphx/migraphx.hpp>


struct ModelLoadSettings
{
    size_t sequnce_length;
    bool quantize_fp16;
    bool offload_copy;
    bool fast_math;
    bool input_one_dim;
};

static std::string getModelPath(ModelLoadSettings& s)
{
    std::stringstream path;
    path << MODEL_FOLDER << "model-" << std::to_string(s.sequnce_length) << "_fp" << (s.quantize_fp16 ? "16" : "32") << "_";
    if (!s.offload_copy)
    {
        path << "no";
    }
    path << "offload_";
    if (!s.fast_math)
    {
        path << "no";
    }
    path << "fastmath";
    if (s.input_one_dim)
    {
        path << "_inputonedim";
    }
    path << ".mxr";
    return path.str();
}

[[maybe_unused]] static std::string getPastKeyString(size_t i)
{
    std::stringstream past_key;
    past_key << "past_key_values." << std::to_string(i) << ".key";
    return past_key.str();
}

[[maybe_unused]] static std::string getPastValueStr(size_t i)
{
    std::stringstream past_val;
    past_val << "past_key_values." << std::to_string(i) << ".value";
    return past_val.str();
}

[[maybe_unused]] static std::string getPresentKeyString(size_t i)
{
    std::stringstream past_key;
    past_key << "present." << std::to_string(i) << ".key";
    return past_key.str();
}

[[maybe_unused]] static std::string getPresentValueStr(size_t i)
{
    std::stringstream past_val;
    past_val << "present." << std::to_string(i) << ".value";
    return past_val.str();
}

static migraphx::program loadOnnx(ModelLoadSettings& settings)
{
    std::filesystem::path onnx_path(MODEL_FOLDER + ONNX_FILE);

    #ifdef TRACE
    std::cout << "Using model: " << MODEL_FOLDER + ONNX_FILE << std::endl;
    #endif

    migraphx::program prog;
    std::ifstream f(onnx_path.c_str());
    if (f.good())
    {
        migraphx::onnx_options onnx_opts;
        std::vector<std::size_t> dims = {1, SEQ_SIZE};
        std::vector<std::size_t> dimsPastKey = {1, HIDDEN_LAYERS_NUM, SEQ_SIZE, HEAD_SIZE};
        std::vector<std::size_t> inputDim;
        if (settings.input_one_dim)
        {
            inputDim = {1,1};
        }
        else
        {
            inputDim = dims;
        }
        onnx_opts.set_input_parameter_shape("input_ids", inputDim);
        onnx_opts.set_input_parameter_shape("attention_mask", dims);
        for (size_t i = 0; i < HIDDEN_LAYERS_NUM; ++i)
        {
            onnx_opts.set_input_parameter_shape(getPastKeyString(i), dimsPastKey);
            onnx_opts.set_input_parameter_shape(getPastValueStr(i), dimsPastKey);
        }
        std::cout << "Parsing onnx file ..." << std::endl;
        prog = parse_onnx(onnx_path.c_str(), onnx_opts);

        std::string target_str = "gpu";
        migraphx::target targ = migraphx::target(target_str.c_str());

        if (settings.quantize_fp16)
        {
            std::cout << "Quantize FP16 ..." << std::endl;
            migraphx::quantize_fp16(prog);
        }

        migraphx::compile_options comp_opts;

        if (settings.offload_copy)
            comp_opts.set_offload_copy();

        if (settings.fast_math)
            comp_opts.set_fast_math();

        comp_opts.set_exhaustive_tune_flag();

        std::cout << "Compile to target ..." << std::endl;
        prog.compile(targ, comp_opts);

        std::string modelPath = getModelPath(settings);
        migraphx::file_options file_options;
        file_options.set_file_format("msgpack");
        std::cout << "Saving mxr file to: " << modelPath << "\n";
        migraphx::save(prog, modelPath.c_str(), file_options);
    }
    else
    {
        std::cerr << "Onnx file is not available on path: " << onnx_path << std::endl;
        exit(1);
    }
    return prog;
};

static migraphx::program loadProgram(ModelLoadSettings& settings)
{
    std::filesystem::path compiled_path(getModelPath(settings));

    migraphx::file_options file_options;
    file_options.set_file_format("msgpack");

    migraphx::program prog;
    std::ifstream f(compiled_path.c_str());
    if (f.good())
    {
        std::cout << "Loading model from " << compiled_path << " ...\n";
        prog = migraphx::load(compiled_path.c_str(), file_options);
    }
    else
    {
        std::cout << "MXR file can't be loaded try to load ONNX\n";
        prog = loadOnnx(settings);
    }
    return prog;
};

static migraphx::program create_argmax_program(ModelLoadSettings& settings)
{
    migraphx::program prog;
    std::vector<size_t> dims {1, SEQ_SIZE, VOCAB_SIZE};
    if (settings.input_one_dim)
    {
        dims[1] = 1;
    }
    migraphx::shape s{migraphx_shape_half_type, dims};
    migraphx::module m = prog.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto argmax_ins    = m.add_instruction(migraphx::operation("argmax", "{axis: 2}"), {x});
    m.add_return({argmax_ins});

    std::cout << "Creating ArgMax program ..." << std::endl;
  
    std::string target_str = "gpu";
    migraphx::target targ = migraphx::target(target_str.c_str());

    if (settings.quantize_fp16)
    {
        std::cout << "Quantize FP16 ..." << std::endl;
        migraphx::quantize_fp16(prog);
    }

    migraphx::compile_options comp_opts;

    if (settings.offload_copy)
        comp_opts.set_offload_copy();

    if (settings.fast_math)
        comp_opts.set_fast_math();

    comp_opts.set_exhaustive_tune_flag();

    std::cout << "Compile to target ..." << std::endl;
    prog.compile(targ, comp_opts);

    return prog;
}

static void writeResults(const std::vector<std::vector<uint64_t>>& results)
{
    std::string RESULT_FILE = "result.txt";
    std::ofstream outFile(RESULT_FILE);
    for (auto& resVec : results)
    {
        for (auto& res : resVec)
        {
            outFile << res;
            if (&res != &resVec.back())
            {
                outFile << ", ";
            }
        }
        outFile << "\n";
    }
}
