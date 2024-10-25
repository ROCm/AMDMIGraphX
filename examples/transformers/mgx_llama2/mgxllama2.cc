#include "buffer.hpp"
#include "numpy.hpp"
#include <migraphx/migraphx.hpp>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace mlinfer;

// TODO: fix paths
const std::string MODEL_FOLDER = "/code/AMDMIGraphX/examples/transformers/python_llama2/models/llama-2-7b-chat-hf/";
const std::string ONNX_FILE = "model.onnx";
std::vector<int64_t> SAMPLE_IDS = {1,6804,5207,387,287,29973};
// sequence length from model config
const size_t SEQ_SIZE = 256;
// vocab size from model config
const size_t VOCAB_SIZE = 32000;
// EOS token from model config
const size_t EOS = 2;

struct ModelLoadSettings
{
    size_t sequnce_length;
    bool quantize_fp16;
    bool offload_copy;
    bool fast_math;
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
    path << "fastmath.mxr";
    return path.str();
}

static migraphx::program loadOnnx(ModelLoadSettings& settings)
{
    std::filesystem::path onnx_path(MODEL_FOLDER + ONNX_FILE);

    migraphx::program prog;
    std::ifstream f(onnx_path.c_str());
    if (f.good())
    {
        migraphx::onnx_options onnx_opts;
        std::vector<std::size_t> dims = {1, SEQ_SIZE};
        onnx_opts.set_input_parameter_shape("input_ids", dims);
        onnx_opts.set_input_parameter_shape("attention_mask", dims);
        onnx_opts.set_input_parameter_shape("position_ids", dims);
        std::cout << "Parsing onnx file ..." << std::endl;
        prog = parse_onnx(onnx_path.c_str(), onnx_opts);

        std::string target_str = "gpu";
        migraphx::target targ = migraphx::target(target_str.c_str());

        std::cout << "Quantize FP16 ..." << std::endl;
        if (settings.quantize_fp16)
            migraphx::quantize_fp16(prog);

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

struct LLama2Inputs
{
    LLama2Inputs(
        migraphx::program& prog,
        migraphx::program_parameters& prog_args,
        bool offload_copy)
        : offload_copy(offload_copy)
    {
        auto input_ids = SAMPLE_IDS;
        input_ids.resize(SEQ_SIZE, 0);
        std::vector<int64_t> attention_mask = input_ids;
        std::transform(std::begin(attention_mask), std::end(attention_mask), std::begin(attention_mask), [](auto i){
            return (i != 0) ? 1 : 0;
        });

        std::vector<int64_t> position_ids;
        for (int64_t i=0; i < SEQ_SIZE; ++i)
        {
            position_ids.emplace_back(i);
        }

        auto param_shapes = prog.get_parameter_shapes();
        auto input_ids_str = "input_ids";
        input_ids_buffer = std::make_unique<LLama2InputBuffer>(std::move(input_ids), offload_copy);
        prog_args.add(input_ids_str, migraphx::argument(param_shapes[input_ids_str], input_ids_buffer->data()));

        auto attention_mask_str = "attention_mask";
        attention_mask_buffer = std::make_unique<LLama2InputBuffer>(std::move(attention_mask), offload_copy);
        prog_args.add(attention_mask_str, migraphx::argument(param_shapes[attention_mask_str], attention_mask_buffer->data()));

        auto position_ids_str = "position_ids";
        position_ids_buffer = std::make_unique<LLama2InputBuffer>(std::move(position_ids), offload_copy);
        prog_args.add(position_ids_str, migraphx::argument(param_shapes[position_ids_str], position_ids_buffer->data()));
    };

    void upload_to_device(hipStream_t stream)
    {
        assert(not offload_copy);
        input_ids_buffer->upload_to_device(stream);
        attention_mask_buffer->upload_to_device(stream);
        position_ids_buffer->upload_to_device(stream);
    }

    LLama2Inputs() = delete;
    LLama2Inputs(const LLama2Inputs &buf) = delete;
    LLama2Inputs &operator=(const LLama2Inputs &buf) = delete;

    std::unique_ptr<LLama2InputBuffer> input_ids_buffer;
    std::unique_ptr<LLama2InputBuffer> attention_mask_buffer;
    std::unique_ptr<LLama2InputBuffer> position_ids_buffer;
    bool offload_copy;
};

int main() {
    bool offload_copy = false;
    std::cout << "Offload copy: " << std::boolalpha << offload_copy << std::endl;
    ModelLoadSettings settings = {SEQ_SIZE, true /*quantize_fp16*/, false /*offload_copy*/, true /*fast_math*/};
    migraphx::program prog = loadProgram(settings);
    std::cout << "Model loaded" << std::endl;

    // Setup model inputs
    auto output_tokens = SAMPLE_IDS;
    migraphx::program_parameters prog_args;
    hipStream_t stream;
    check_hip_status(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto model_inputs = LLama2Inputs(prog, prog_args, offload_copy);
    if (not offload_copy)
    {
        model_inputs.upload_to_device(stream);
    }

    // Setup model output for non-offload copy
    const size_t output_size = SEQ_SIZE * VOCAB_SIZE;
    auto output_name = "main:#output_0";
    auto output_buffer = LLama2OutputBuffer(std::vector<float>(output_size), offload_copy);
    migraphx::shape out_shape{migraphx_shape_float_type, {1, SEQ_SIZE, VOCAB_SIZE}};
    prog_args.add(output_name, migraphx::argument(out_shape, output_buffer.data()));

    std::cout << "Starting evaluation" << std::endl;
    for (int i = SAMPLE_IDS.size() - 1; i < SEQ_SIZE; ++i)
    {
        auto outputs = prog.run_async(prog_args, stream);
        if (not offload_copy)
        {
            output_buffer.download_from_device(stream, i * VOCAB_SIZE, (i + 1) * VOCAB_SIZE);
        }

        check_hip_status(hipStreamSynchronize(stream));
        float* results   = offload_copy ? reinterpret_cast<float*>(outputs[0].data()) : reinterpret_cast<float*>(output_buffer.hbuff.data());
        std::vector<float> logits(results, results + output_size);
        std::vector<float>::iterator max = std::max_element(std::begin(logits) + (i * VOCAB_SIZE), std::begin(logits) + ((i + 1) * VOCAB_SIZE));
        int64_t new_token = std::distance(std::begin(logits) + (i * VOCAB_SIZE), max);
        // std::cout << "New token: " << new_token << std::endl;
        output_tokens.push_back(new_token);
        if (new_token == EOS)
        {
            break;
        }
        model_inputs.input_ids_buffer->update_data(new_token, i +1, stream);
        model_inputs.attention_mask_buffer->update_data(1, i +1, stream);
    }

    std::cout << "######### Output token ids #########" << std::endl;
    // print output tokens
    for (auto tok: output_tokens){
        std::cout << tok << ", ";
    }
    std::cout << std::endl;
    return 0;
}
