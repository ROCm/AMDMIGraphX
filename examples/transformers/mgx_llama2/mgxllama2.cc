#include "buffer.hpp"
#include "numpy.hpp"
#include <migraphx/migraphx.hpp>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace mlinfer;

const std::string MODEL_FILE_PATH = "/code/AMDMIGraphX/examples/transformers/python_llama2/models/llama-2-7b-chat-hf/model-256.mxr";
std::vector<int64_t> SAMPLE_IDS = {1,6804,5207,387,287,29973};
// sequence length from model config
const size_t SEQ_SIZE = 256;
const size_t VOCAB_SIZE = 32000;
// EOS token from model config
const size_t EOS = 2;

static migraphx::program loadProgram()
{
    std::filesystem::path compiled_path(MODEL_FILE_PATH);

    migraphx::file_options file_options;
    file_options.set_file_format("msgpack");

    migraphx::program prog;
    std::ifstream f(compiled_path.c_str());
    if (f.good())
    {
        prog = migraphx::load(compiled_path.c_str(), file_options);
    }
    else
    {
        std::cout << "model is not good.\n";
    }
    return prog;
}

int main() {
    std::cout << "Loading model ..." << std::endl;
    migraphx::program prog = loadProgram();
    std::cout << "Model loaded" << std::endl;

    prog.print();

    auto output_tokens = SAMPLE_IDS;
    SAMPLE_IDS.resize(SEQ_SIZE, 0);
    std::vector<int64_t> attention_mask = SAMPLE_IDS;
    std::transform(std::begin(attention_mask), std::end(attention_mask), std::begin(attention_mask), [](auto i){
        return (i != 0) ? 1 : 0;
    });

    std::vector<int64_t> position_ids;
    for (int64_t i=0; i < SEQ_SIZE; ++i)
    {
        position_ids.emplace_back(i);
    }

    migraphx::program_parameters prog_args;
    auto param_shapes = prog.get_parameter_shapes();

    size_t alloc_size = SEQ_SIZE * sizeof(int64_t);

    std::cout << "Uploading input ids to the GPU" << std::endl;
    auto input_ids_str = "input_ids";
    // auto input_ids_buffer = ManagedBuffer(alloc_size);
    // input_ids_buffer.upload_to_device(static_cast<void*>(SAMPLE_IDS.data()), alloc_size);
    prog_args.add(input_ids_str, migraphx::argument(param_shapes[input_ids_str], SAMPLE_IDS.data()));

    std::cout << "Uploading attention mask to the GPU" << std::endl;
    auto attention_mask_str = "attention_mask";
    // auto attention_mask_buffer = ManagedBuffer(alloc_size);
    // attention_mask_buffer.upload_to_device(static_cast<void*>(attention_mask.data()), alloc_size);
    prog_args.add(attention_mask_str, migraphx::argument(param_shapes[attention_mask_str], attention_mask.data()));

    std::cout << "Uploading position ids to the GPU" << std::endl;
    auto position_ids_str = "position_ids";
    // auto position_ids_buffer = ManagedBuffer(alloc_size);
    // position_ids_buffer.upload_to_device(static_cast<void*>(position_ids.data()), alloc_size);
    prog_args.add(position_ids_str, migraphx::argument(param_shapes[position_ids_str], position_ids.data()));

    // Handle output tensors
    // std::cout << "Creating output buffer" << std::endl;
    const size_t output_size = SEQ_SIZE * VOCAB_SIZE;
    // name = "@return";
    // auto output_buffer = ManagedBuffer(output_size);
    // migraphx::shape outShape{migraphx_shape_float_type, {1, 256, 32000}};
    // prog_args.add(name, migraphx::argument(outShape, output_buffer.get_device_ptr<void*>()));

    std::cout << "Starting evaluation" << std::endl;
    for (int i = 5; i < SEQ_SIZE; ++i)
    {
        std::cout << "# iter: " << i << std::endl;
        auto outputs = prog.eval(prog_args);
        // TODO: Only download the relevant data range
        float* results   = reinterpret_cast<float*>(outputs[0].data());
        std::vector<float> logits(results, results + output_size);
        // std::cout << "## logits size: " << logits.size() << std::endl;
        std::vector<float>::iterator max = std::max_element(std::begin(logits) + (i * VOCAB_SIZE), std::begin(logits) + ((i + 1) * VOCAB_SIZE));
        int64_t new_token = std::distance(std::begin(logits) + (i * VOCAB_SIZE), max);
        output_tokens.push_back(new_token);
        if (new_token == EOS)
        {
            break;
        }
        SAMPLE_IDS[i + 1] = new_token;
        prog_args.add(input_ids_str, migraphx::argument(param_shapes[input_ids_str], SAMPLE_IDS.data()));
        attention_mask[i + 1] = 1;
        prog_args.add(attention_mask_str, migraphx::argument(param_shapes[attention_mask_str], attention_mask.data()));
        // input_ids_buffer.update_device_data<int64_t>(new_token, i + 1);
        // attention_mask_buffer.update_device_data<int64_t>(1, i + 1);
    }

    std::cout << "######### Output token ids #########" << std::endl;
    // print output tokens
    for (auto tok: output_tokens){
        std::cout << tok << ", ";
    }
    std::cout << std::endl;
    return 0;
}
