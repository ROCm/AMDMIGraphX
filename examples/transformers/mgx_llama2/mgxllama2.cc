#include "buffer.hpp"
#include "common.hpp"
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
const size_t DATASET_SIZE = 10;
// sequence length from model config
const size_t SEQ_SIZE = 4096;
// vocab size from model config
const size_t VOCAB_SIZE = 32000;
// EOS token from model config
const size_t EOS = 2;
// Write output tokens to file
const bool WRITE_RESULT_FILE = false;

const size_t HIDDEN_LAYERS_NUM = 32;
const size_t HEAD_SIZE = 128;
const size_t PAST_KEY_VAL_SIZE = HIDDEN_LAYERS_NUM*HEAD_SIZE*SEQ_SIZE;

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

std::string getPastKeyString(size_t i)
{
    std::stringstream past_key;
    past_key << "past_key_values." << std::to_string(i) << ".key";
    return past_key.str();
}

std::string getPastValueStr(size_t i)
{
    std::stringstream past_val;
    past_val << "past_key_values." << std::to_string(i) << ".value";
    return past_val.str();
}

std::string getPresentKeyString(size_t i)
{
    std::stringstream past_key;
    past_key << "present." << std::to_string(i) << ".key";
    return past_key.str();
}

std::string getPresentValueStr(size_t i)
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
        onnx_opts.set_input_parameter_shape("position_ids", dims);
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

using NumpyVector = std::vector<std::vector<int64_t>>;

struct Dataset
{
    Dataset() = default;

    void initialize()
    {
        loadDataset();
        if (!_npy_files_loaded)
        {
            prepareSampleDataset();
        }
    }

    NumpyVector loadNumpy(npy::NpyFile& file)
    {
        NumpyVector numpyDataAll;
        auto load_size = file.GetTensorSize()/sizeof(int64_t);
        numpyDataAll.push_back(std::vector<int64_t>(load_size));
        file.LoadAll(numpyDataAll.back().data());

        NumpyVector numpyData;
        for(size_t i = 0; i < numpyDataAll.back().size(); i += SEQ_SIZE)
        {
            auto last = std::min(numpyDataAll.back().size(), i + SEQ_SIZE);
            numpyData.emplace_back(numpyDataAll.back().begin() + i, numpyDataAll.back().begin() + last);
        }

#ifdef TRACE
        for (auto& vec: numpyData)
        {
            std::cout << "Vector size: " << vec.size() <<  std::endl;
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
        auto res = std::find_if(std::rbegin(attention_mask[_current_idx]), std::rend(attention_mask[_current_idx]), [](uint64_t val) { return 1 == val;});
        size_t last_idx = std::distance(res, std::rend(attention_mask[_current_idx]));
        #ifdef TRACE
        std::cout << "Last input idx: " << last_idx << std::endl;
        #endif
        return last_idx;
    }

    std::vector<int64_t> getInputIds() { return input_ids[_current_idx]; }
    std::vector<int64_t> getAttentionMask() { return attention_mask[_current_idx]; }
    std::vector<int64_t> getPositionIds() { return position_ids[_current_idx]; }

    size_t size() const { return _size; }
    size_t currentIdx() const { return _current_idx; }
    size_t getNext()
    {
        if (_current_idx < size() - 1)
        {
            ++_current_idx;
        }
        #ifdef TRACE
        std::cout << "Current idx: " << _current_idx << std::endl;
        #endif
        return _current_idx;
    }

    Dataset(const Dataset &buf) = delete;
    Dataset &operator=(const Dataset &buf) = delete;
private:

    // e.g.: /dataset/input_ids_size_3_seq_256.npy
    std::string getDatasetPath(const std::string& datasetName)
    {
        std::stringstream path;
        path << DATASET_FOLDER << datasetName << "_size_" << std::to_string(DATASET_SIZE) << "_seq_" << std::to_string(SEQ_SIZE) << ".npy";
        return path.str();
    }

    void loadDataset()
    {
        std::string input_file_path = getDatasetPath("input_ids");
        std::string attention_mask_file_path = getDatasetPath("attention_mask");
        std::string position_ids_file_path = getDatasetPath("position_ids");

        std::cout << "Input ids file: " << input_file_path << std::endl;
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

            _size = input_ids.size();

            if ((input_ids.size() == attention_mask.size()) && (attention_mask.size() == position_ids.size()))
            {
                std::cout << "Loaded numpy files\n";
                _npy_files_loaded = true;
            }
            else
            {
                std::cout << "Numpy files do not have the same size\n";
                input_ids.clear();
                attention_mask.clear();
                position_ids.clear();
            }
        }
    }

    void prepareSampleDataset()
    {
        std::cout << "Numpy files are not loaded, using dummy data\n";
        std::vector<int64_t> input_ids_sample = {1,6804,338,5207,387,287,29973};
        input_ids_sample.resize(SEQ_SIZE, 0);
        std::vector<int64_t> attention_mask_sample = input_ids_sample;
        input_ids.emplace_back(std::move(input_ids_sample));
        std::transform(std::begin(attention_mask_sample), std::end(attention_mask_sample), std::begin(attention_mask_sample), [](auto i){
            return (i != 0) ? 1 : 0;
        });
        attention_mask.emplace_back(std::move(attention_mask_sample));

        std::vector<int64_t> position_ids_sample;
        for (int64_t i=0; i < SEQ_SIZE; ++i)
        {
            position_ids_sample.emplace_back(i);
        }
        position_ids.emplace_back(std::move(position_ids_sample));

        _size = 1;
    }

    NumpyVector input_ids;
    NumpyVector attention_mask;
    NumpyVector position_ids;

    size_t _size = 0;
    size_t _current_idx = 0;
    bool _npy_files_loaded = false;
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

        auto param_shapes = prog.get_parameter_shapes();

        auto inputShape = param_shapes[INPUTS_ID_STR];
        auto input_ids = data.getInputIds();
        input_ids_buffer = std::make_unique<LLama2InputBuffer>(std::move(input_ids), offload_copy);
        prog_args.add(INPUTS_ID_STR, migraphx::argument(inputShape, input_ids_buffer->data()));


        auto attShape = param_shapes[ATTENTION_MASK_STR];
        auto attention_mask = data.getAttentionMask();
        attention_mask_buffer = std::make_unique<LLama2InputBuffer>(std::move(attention_mask), offload_copy);
        prog_args.add(ATTENTION_MASK_STR, migraphx::argument(attShape, attention_mask_buffer->data()));

        //auto positionShape = param_shapes[POSITION_IDS_STR];
        auto positionShape = inputShape;
        auto position_ids = data.getPositionIds();
        position_ids_buffer = std::make_unique<LLama2InputBuffer>(std::move(position_ids), offload_copy);
        prog_args.add(POSITION_IDS_STR, migraphx::argument(inputShape, position_ids_buffer->data()));

        // past_key_values.0.key = @param:past_key_values.0.key -> half_type, {1, 32, 1, 128}, {4096, 128, 128, 1}
        // past_key_values.0.value = @param:past_key_values.0.value -> half_type, {1, 32, 1, 128}, {4096, 128, 128, 1}
        for (size_t i = 0; i < HIDDEN_LAYERS_NUM; ++i)
        {
            auto past_keyStr = getPastKeyString(i);
            auto past_keyString = past_keyStr.c_str();
            past_key_buffers.emplace_back(std::make_unique<LLama2PastKeyValueBuffer>(std::vector<half>(PAST_KEY_VAL_SIZE, 0.0_h), offload_copy));
            auto pastKeyShape = param_shapes[past_keyString];
            prog_args.add(past_keyString, migraphx::argument(pastKeyShape, past_key_buffers[i]->data()));

            auto past_valueStr = getPastValueStr(i);
            auto past_valueString = past_valueStr.c_str();
            past_value_buffers.emplace_back(std::make_unique<LLama2PastKeyValueBuffer>(std::vector<half>(PAST_KEY_VAL_SIZE, 0.0_h), offload_copy));
            auto pastValueShape = param_shapes[past_valueString];
            prog_args.add(past_valueString, migraphx::argument(pastValueShape, past_value_buffers[i]->data()));
        }
    };

    void upload_to_device(hipStream_t stream)
    {
        assert(not offload_copy);
        input_ids_buffer->upload_to_device(stream);
        attention_mask_buffer->upload_to_device(stream);
        position_ids_buffer->upload_to_device(stream);
    }

    bool updateData(migraphx::program& prog, migraphx::program_parameters& prog_args)
    {
        auto currentIdx = data.currentIdx();
        if (currentIdx != data.getNext())
        {
            auto param_shapes = prog.get_parameter_shapes();

            auto input_ids = data.getInputIds();
            input_ids_buffer->update(std::move(input_ids));
            if (offload_copy)
            {
                prog_args.add(INPUTS_ID_STR, migraphx::argument(param_shapes[INPUTS_ID_STR], input_ids_buffer->data()));
            }

            auto attention_mask = data.getAttentionMask();
            attention_mask_buffer->update(std::move(attention_mask));
            if (offload_copy)
            {
                prog_args.add(ATTENTION_MASK_STR, migraphx::argument(param_shapes[ATTENTION_MASK_STR], attention_mask_buffer->data()));
            }

            auto position_ids = data.getPositionIds();
            position_ids_buffer->update(std::move(position_ids));
            if (offload_copy)
            {
                prog_args.add(POSITION_IDS_STR, migraphx::argument(param_shapes[POSITION_IDS_STR], position_ids_buffer->data()));
            }
            return true;
        }
        return false;
    }

    size_t getLastInputIndex() const { return data.getLastIdx(); }
    size_t dataSize() const { return data.size(); }

    LLama2Inputs() = delete;
    LLama2Inputs(const LLama2Inputs &buf) = delete;
    LLama2Inputs &operator=(const LLama2Inputs &buf) = delete;

    std::unique_ptr<LLama2InputBuffer> input_ids_buffer;
    std::unique_ptr<LLama2InputBuffer> attention_mask_buffer;
    std::unique_ptr<LLama2InputBuffer> position_ids_buffer;
    std::vector<std::unique_ptr<LLama2PastKeyValueBuffer>> past_key_buffers;
    std::vector<std::unique_ptr<LLama2PastKeyValueBuffer>> past_value_buffers;
    Dataset data;
    bool offload_copy;

    const char* INPUTS_ID_STR = "input_ids";
    const char* ATTENTION_MASK_STR = "attention_mask";
    const char* POSITION_IDS_STR = "position_ids";
};

void writeResults(const std::vector<std::vector<uint64_t>>& results)
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

int main() {
    bool offload_copy = true;
    std::cout << "Offload copy: " << std::boolalpha << offload_copy << std::endl;
    ModelLoadSettings settings = {SEQ_SIZE, false /*quantize_fp16*/, offload_copy /*offload_copy*/, false /*fast_math*/, false /*input_one_dim*/};
    migraphx::program prog = loadProgram(settings);
    std::cout << "Model loaded" << std::endl;

    // Load {1,1} input_ids model
    settings.input_one_dim = true;
    migraphx::program progSimpleInput = loadProgram(settings);
    std::cout << "Model 1 dim input loaded" << std::endl;

    // Setup model inputs
    std::vector<std::vector<uint64_t>> results;
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
    // const size_t output_size = SEQ_SIZE * VOCAB_SIZE;
    // auto output_name = "main:#output_0";
    // auto output_buffer = LLama2OutputBuffer(std::vector<float>(output_size), offload_copy);
    // migraphx::shape out_shape{migraphx_shape_float_type, {1, SEQ_SIZE, VOCAB_SIZE}};
    // prog_args.add(output_name, migraphx::argument(out_shape, output_buffer.data()));

    size_t output_size = SEQ_SIZE * VOCAB_SIZE;
    auto output_name = "logits";
    auto output_buffer = LLama2PastKeyValueBuffer(std::vector<half>(output_size), offload_copy);
    migraphx::shape out_shape{migraphx_shape_float_type, {1, SEQ_SIZE, VOCAB_SIZE}};
    prog_args.add(output_name, migraphx::argument(out_shape, output_buffer.data()));

    // std::vector<std::unique_ptr<LLama2PastKeyValueBuffer>> present_key_buffers;
    // std::vector<std::unique_ptr<LLama2PastKeyValueBuffer>> present_value_buffers;
    // for (size_t i = 0; i < HIDDEN_LAYERS_NUM; ++i)
    // {
    //     migraphx::shape present_shape{migraphx_shape_half_type, {1, HIDDEN_LAYERS_NUM, SEQ_SIZE, HEAD_SIZE}};
    //     auto present_keyStr = getPresentKeyString(i);
    //     auto present_keyString = present_keyStr.c_str();
    //     present_key_buffers.emplace_back(std::make_unique<LLama2PastKeyValueBuffer>(std::vector<half>(PAST_KEY_VAL_SIZE, 0.0_h), offload_copy));
    //     prog_args.add(present_keyString, migraphx::argument(present_shape, present_key_buffers[i]->data()));

    //     auto present_valueStr = getPresentValueStr(i);
    //     auto present_valueString = present_valueStr.c_str();
    //     present_value_buffers.emplace_back(std::make_unique<LLama2PastKeyValueBuffer>(std::vector<half>(PAST_KEY_VAL_SIZE, 0.0_h), offload_copy));
    //     prog_args.add(present_valueString, migraphx::argument(present_shape, present_value_buffers[i]->data()));
    // }

    std::cout << "Starting evaluation" << std::endl;
    size_t token_count = 0;
    auto start = std::chrono::steady_clock::now();
    std::cout << "Dataset size: " << model_inputs.dataSize() << std::endl;
    for (size_t i = 0; i < model_inputs.dataSize(); ++i)
    {
        #ifdef TRACE
        std::cout << "Iter #" << i << std::endl;
        #endif
        auto lastInputIdx = model_inputs.getLastInputIndex();
        for (size_t i = lastInputIdx; i < SEQ_SIZE - 1; ++i)
        {
            auto outputs = prog.run_async(prog_args, stream);
            if (not offload_copy)
            {
                output_buffer.download_from_device(stream, i * VOCAB_SIZE, (i + 1) * VOCAB_SIZE);
            }

            check_hip_status(hipStreamSynchronize(stream));
            half* results   = offload_copy ? reinterpret_cast<half*>(outputs[0].data()) : reinterpret_cast<half*>(output_buffer.hbuff.data());
            std::vector<half> logits(results, results + output_size);

            bool firstIter = (i == lastInputIdx);
            auto logits_begin = firstIter ? std::begin(logits) + (i * VOCAB_SIZE) : std::begin(logits);
            auto logits_end = firstIter ? std::begin(logits) + ((i + 1) * VOCAB_SIZE) : std::end(logits);
            std::vector<half>::iterator max = std::max_element(logits_begin, logits_end);
            int64_t new_token = std::distance(logits_begin, max);

            token_count++;
            #ifdef TRACE
            std::cout << "New token: " << new_token << std::endl;
            #endif
            output_tokens.push_back(new_token);

            for (size_t i = 0; i < HIDDEN_LAYERS_NUM; ++i)
            {
                migraphx::shape past_shape{migraphx_shape_half_type, {1, HIDDEN_LAYERS_NUM, SEQ_SIZE, HEAD_SIZE}};
                half* res   = reinterpret_cast<half*>(outputs[2*i+1].data());
                std::vector<half> present_key(res, res + PAST_KEY_VAL_SIZE);

                auto past_keyStr = getPastKeyString(i);
                model_inputs.past_key_buffers[i]->update(std::move(present_key));
                prog_args.add(past_keyStr.c_str(), migraphx::argument(past_shape, model_inputs.past_key_buffers[i]->data()));

                res = reinterpret_cast<half*>(outputs[2*i+2].data());
                std::vector<half> present_value(res, res + PAST_KEY_VAL_SIZE);

                auto past_valueStr = getPastValueStr(i);
                model_inputs.past_value_buffers[i]->update(std::move(present_value));
                prog_args.add(past_valueStr.c_str(), migraphx::argument(past_shape, model_inputs.past_value_buffers[i]->data()));
            }

            if (new_token == EOS)
            {
                break;
            }

            model_inputs.attention_mask_buffer->update_data(1, i + 1, stream);

            if (firstIter)
            {
                prog = progSimpleInput;
                output_size = VOCAB_SIZE;

                auto param_shapes = prog.get_parameter_shapes();
                auto inputShape = param_shapes[model_inputs.INPUTS_ID_STR];
                std::vector<int64_t> input_ids = {new_token};
                model_inputs.input_ids_buffer = std::make_unique<LLama2InputBuffer>(std::move(input_ids), offload_copy);
                prog_args.add(model_inputs.INPUTS_ID_STR, migraphx::argument(inputShape, model_inputs.input_ids_buffer->data()));
            }
            else
            {
                model_inputs.input_ids_buffer->update_data(new_token, 0, stream);
            }
        }

#ifdef TRACE
        std::cout << "######### Output token ids for #" << i << " #########" << std::endl;
        // print output tokens
        for (auto tok: output_tokens){
            std::cout << tok << ", ";
        }
        std::cout << std::endl;
#endif
        auto updated = model_inputs.updateData(prog, prog_args);

        if (updated && not offload_copy)
        {
            model_inputs.upload_to_device(stream);
        }
        results.emplace_back(output_tokens);
        output_tokens.clear();
    }

    float dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() / 1000.f;
    std::cout << "Duration: " << dur << " seconds." << std::endl;
    std::cout << "Completed " << token_count << " tokens." << std::endl;
    std::cout << "Tokens/sec: " << token_count / dur << std::endl;

    if (WRITE_RESULT_FILE)
    {
        writeResults(results);
    }
    return 0;
}
