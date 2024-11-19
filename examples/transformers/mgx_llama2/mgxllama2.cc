#include "buffer.hpp"
#include "common.hpp"
#include "dataset.hpp"
#include "llama2inputs.hpp"
#include "utils.hpp"
#include <migraphx/migraphx.hpp>


int main() {
    bool offload_copy = false;
    check_hip_status(hipSetDevice(DEVICE_ID));
    std::cout << "Offload copy: " << std::boolalpha << offload_copy << std::endl;
    ModelLoadSettings settings = {SEQ_SIZE, false /*quantize_fp16*/, offload_copy /*offload_copy*/, false /*fast_math*/, false /*input_one_dim*/};
    migraphx::program progMultipleInputDim = loadProgram(settings);
    std::cout << "Model loaded" << std::endl;
    migraphx::program progArgMaxMultipleInputDim = create_argmax_program(settings);
    std::cout << "ArgMax model created" << std::endl;

    // Load {1,1} input_ids model
    settings.input_one_dim = true;
    migraphx::program progSimpleInput = loadProgram(settings);
    std::cout << "Model 1 dim input loaded" << std::endl;
    migraphx::program progArgMaxSimpleInput = create_argmax_program(settings);
    std::cout << "ArgMax model for 1 dim model created" << std::endl;

    migraphx::program *prog = &progMultipleInputDim;
    migraphx::program *progArgMax = &progArgMaxMultipleInputDim;

    // Setup model inputs
    std::vector<std::vector<uint64_t>> results;
    std::vector<uint64_t> output_tokens;
    migraphx::program_parameters prog_args;
    hipStream_t stream;
    check_hip_status(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto model_inputs = LLama2Inputs(*prog, prog_args, offload_copy);
    if (not offload_copy)
    {
        model_inputs.upload_to_device(stream);
    }

    auto output_name = "main:#output_0";

    size_t output_size = SEQ_SIZE * VOCAB_SIZE;
    auto output_buffer = LLama2PastKeyValueBuffer(std::vector<half>(output_size), offload_copy);
    auto output_buffer_oneDim = LLama2PastKeyValueBuffer(std::vector<half>(VOCAB_SIZE), offload_copy);
    migraphx::shape out_shape{migraphx_shape_half_type, {1, SEQ_SIZE, VOCAB_SIZE}};
    prog_args.add(output_name, migraphx::argument(out_shape, output_buffer.data()));

    // setting up argmax arguments
    migraphx::program_parameters prog_args_argmax;
    migraphx::shape x_shape{migraphx_shape_half_type, {1, SEQ_SIZE, VOCAB_SIZE}};
    prog_args_argmax.add("x", migraphx::argument(x_shape, output_buffer.data()));
    auto argm_output_buffer = ArgMaxOutputBuffer(std::vector<int64_t>(VOCAB_SIZE), offload_copy);
    migraphx::shape argm_out_shape{migraphx_shape_int64_type, {1, SEQ_SIZE, 1}};
    prog_args_argmax.add(output_name, migraphx::argument(argm_out_shape, argm_output_buffer.data()));

    migraphx::program_parameters prog_args_argmax_one_dim;
    migraphx::shape x_shape_one_dim{migraphx_shape_half_type, {1, 1, VOCAB_SIZE}};
    prog_args_argmax_one_dim.add("x", migraphx::argument(x_shape_one_dim, output_buffer_oneDim.data()));
    auto argm_output_buffer_one_dim = ArgMaxOutputBuffer(std::vector<int64_t>(1), offload_copy);
    migraphx::shape argm_out_shape_one_dim{migraphx_shape_int64_type, {1, 1, 1}};
    prog_args_argmax_one_dim.add(output_name, migraphx::argument(argm_out_shape_one_dim, argm_output_buffer_one_dim.data()));


    migraphx::program_parameters prog_args_one_dim;
    model_inputs.prepareProgArgs(progSimpleInput, prog_args_one_dim, true);
    auto param_shapes = progSimpleInput.get_parameter_shapes();
    auto inputShape = param_shapes[model_inputs.INPUTS_ID_STR];
    std::vector<int64_t> oneDimInput = {0};
    std::unique_ptr<LLama2InputBuffer> one_dim_input_buffer = std::make_unique<LLama2InputBuffer>(std::move(oneDimInput), offload_copy);
    prog_args_one_dim.add(model_inputs.INPUTS_ID_STR, migraphx::argument(inputShape, one_dim_input_buffer->data()));
    prog_args_one_dim.add(output_name, migraphx::argument(x_shape_one_dim, output_buffer_oneDim.data()));

    if (not offload_copy)
    {
        one_dim_input_buffer->upload_to_device(stream);
    }

    std::cout << "Dataset size: " << model_inputs.dataSize() << std::endl;
    std::cout << "Starting evaluation" << std::endl;
    size_t token_count = 0;
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < model_inputs.dataSize(); ++i)
    {
        #ifdef TRACE
        std::cout << "Iter #" << i << std::endl;
        #endif
        auto lastInputIdx = model_inputs.getLastInputIndex();
        for (size_t i = lastInputIdx; i < SEQ_SIZE - 1; ++i)
        {
            bool firstIter = (i == lastInputIdx);
            prog->run_async(firstIter ? prog_args : prog_args_one_dim, stream);
            auto outputs = progArgMax->run_async(firstIter ? prog_args_argmax : prog_args_argmax_one_dim, stream);
            if (not offload_copy)
            {
                firstIter ? argm_output_buffer.download_from_device(stream, i, i + 1) : argm_output_buffer_one_dim.download_from_device(stream);
            }

            check_hip_status(hipStreamSynchronize(stream));
            int64_t* results = offload_copy ? reinterpret_cast<int64_t*>(outputs[0].data()) : reinterpret_cast<int64_t*>( firstIter ? argm_output_buffer.hbuff.data() : argm_output_buffer_one_dim.hbuff.data());
            auto new_token_idx = firstIter ? i : 0; 
            int64_t new_token = results[new_token_idx];

            token_count++;
            #ifdef TRACE
            std::cout << "New token: " << new_token << std::endl;
            #endif
            output_tokens.push_back(new_token);

            if (new_token == EOS)
            {
                break;
            }

            model_inputs.attention_mask_buffer->update_data(1, i + 1, stream);

            if (firstIter)
            {
                prog = &progSimpleInput;
                progArgMax = &progArgMaxSimpleInput;
            }

            one_dim_input_buffer->update_data(new_token, 0, stream);
        }

#ifdef TRACE
        std::cout << "######### Output token ids for #" << i << " #########" << std::endl;
        // print output tokens
        for (auto tok: output_tokens){
            std::cout << tok << ", ";
        }
        std::cout << std::endl;
#endif
        prog = &progMultipleInputDim;
        progArgMax = &progArgMaxMultipleInputDim;

        auto updated = model_inputs.updateData(*prog, prog_args);

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
