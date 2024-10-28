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
const std::string MODEL_FOLDER = "/model/";
const std::string ONNX_FILE = "model.onnx";
const std::string DATASET_FOLDER = "/dataset/";
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

using NumpyVector = std::vector<std::vector<int64_t>>;

struct Dataset
{
    Dataset() = default;

    void initialize()
    {
        std::string input_file_path = DATASET_FOLDER + "input_ids_size_1_seq_256.npy";
        std::string attention_mask_file_path = DATASET_FOLDER + "attention_mask_size_1_seq_256.npy";
        std::string position_ids_file_path = DATASET_FOLDER + "position_ids_size_1_seq_256.npy";
        std::ifstream input_file(input_file_path.c_str());
        std::ifstream attention_mask_file(attention_mask_file_path.c_str());
        std::ifstream position_ids_file(position_ids_file_path.c_str());
        if (input_file.good() && attention_mask_file.good() && position_ids_file.good())
        {
            npy::NpyFile input_ids_npy{input_file_path};
            npy::NpyFile attention_mask_npy{attention_mask_file_path};
            npy::NpyFile position_ids_npy{position_ids_file_path};
            input_ids = loadNumpy(input_ids_npy);
            attention_mask = loadNumpy(attention_mask_npy);
            position_ids = loadNumpy(position_ids_npy);

            if (input_ids.size() == attention_mask.size() == position_ids.size())
            {
                std::cout << "Loaded numpy files\n";
                npy_files_loaded = true;
            }
            else
            {
                std::cout << "Numpy files do not have the same size\n";
                input_ids.clear();
                attention_mask.clear();
                position_ids.clear();
            }
        }

        if (!npy_files_loaded)
        {
            std::cout << "Numpy files are not loaded, using dummy data\n";
            auto input_ids_sample = SAMPLE_IDS;
            input_ids_sample.resize(SEQ_SIZE, EOS);
            input_ids.emplace_back(input_ids_sample);
            std::vector<int64_t> attention_mask_sample = input_ids_sample;
            std::transform(std::begin(attention_mask_sample), std::end(attention_mask_sample), std::begin(attention_mask_sample), [](auto i){
                return (i != EOS) ? 1 : 0;
            });
            attention_mask.emplace_back(attention_mask_sample);

            std::vector<int64_t> position_ids_sample;
            for (int64_t i=0; i < SEQ_SIZE; ++i)
            {
                position_ids_sample.emplace_back(i);
            }
            position_ids.emplace_back(std::move(position_ids_sample));
        }

    }

    NumpyVector loadNumpy(npy::NpyFile& file)
    {
        NumpyVector numpyData;
        auto load_size = file.GetTensorSize()/sizeof(int64_t);
        numpyData.push_back(std::vector<int64_t>(load_size));
        file.LoadAll(numpyData.back().data());

    #ifdef TRACE
        for (auto& vec: numpyData)
        {
            for (auto val: vec)
            {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    #endif
        return numpyData;
    }

    size_t getLastIdx() const
    {
        auto res = std::find_if(std::rbegin(attention_mask[current_idx]), std::rend(attention_mask[current_idx]), [](uint64_t val) { return 1 == val;});
        size_t last_idx = std::distance(res, std::rend(attention_mask[current_idx]));
        //std::cout << "Last input idx: " << last_idx << std::endl;
        return last_idx;
    }

    std::vector<int64_t> getInputIds() { return input_ids[current_idx]; }
    std::vector<int64_t> getAttentionMask() { return attention_mask[current_idx]; }
    std::vector<int64_t> getPositionIds() { return position_ids[current_idx]; }

    Dataset(const Dataset &buf) = delete;
    Dataset &operator=(const Dataset &buf) = delete;

    NumpyVector input_ids;
    NumpyVector attention_mask;
    NumpyVector position_ids;

    size_t current_idx = 0;
    bool npy_files_loaded = false;
};

struct LLama2Inputs
{
    LLama2Inputs(
        migraphx::program& prog,
        migraphx::program_parameters& prog_args,
        bool offload_copy)
        : offload_copy(offload_copy)
    {
        data.initialize();

        auto input_ids = data.getInputIds();
        auto param_shapes = prog.get_parameter_shapes();
        auto input_ids_str = "input_ids";
        input_ids_buffer = std::make_unique<LLama2InputBuffer>(std::move(input_ids), offload_copy);
        prog_args.add(input_ids_str, migraphx::argument(param_shapes[input_ids_str], input_ids_buffer->data()));

        auto attention_mask = data.getAttentionMask();
        auto attention_mask_str = "attention_mask";
        attention_mask_buffer = std::make_unique<LLama2InputBuffer>(std::move(attention_mask), offload_copy);
        prog_args.add(attention_mask_str, migraphx::argument(param_shapes[attention_mask_str], attention_mask_buffer->data()));

        auto position_ids = data.getPositionIds();
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

    size_t getLastInputIndex() { return data.getLastIdx(); }

    LLama2Inputs() = delete;
    LLama2Inputs(const LLama2Inputs &buf) = delete;
    LLama2Inputs &operator=(const LLama2Inputs &buf) = delete;

    std::unique_ptr<LLama2InputBuffer> input_ids_buffer;
    std::unique_ptr<LLama2InputBuffer> attention_mask_buffer;
    std::unique_ptr<LLama2InputBuffer> position_ids_buffer;
    Dataset data;
    bool offload_copy;
};

int main() {
    bool offload_copy = true;
    std::cout << "Offload copy: " << std::boolalpha << offload_copy << std::endl;
    ModelLoadSettings settings = {SEQ_SIZE, true /*quantize_fp16*/, offload_copy /*offload_copy*/, true /*fast_math*/};
    migraphx::program prog = loadProgram(settings);
    std::cout << "Model loaded" << std::endl;

    // Setup model inputs
    std::vector<uint64_t> output_tokens;
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
    size_t token_count = 0;
    auto start = std::chrono::steady_clock::now();
    for (int i = model_inputs.getLastInputIndex(); i < SEQ_SIZE - 1; ++i)
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
        token_count++;
        // std::cout << "New token: " << new_token << std::endl;
        output_tokens.push_back(new_token);
        if (new_token == EOS)
        {
            break;
        }
        model_inputs.input_ids_buffer->update_data(new_token, i +1, stream);
        model_inputs.attention_mask_buffer->update_data(1, i +1, stream);
    }
    float dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() / 1000.f;
    std::cout << "Duration: " << dur << " seconds." << std::endl;
    std::cout << "Completed " << token_count << " tokens." << std::endl;
    std::cout << "Tokens/sec: " << token_count / dur << std::endl;

    std::cout << "######### Output token ids #########" << std::endl;
    // print output tokens
    for (auto tok: output_tokens){
        std::cout << tok << ", ";
    }
    std::cout << std::endl;
    return 0;
}
