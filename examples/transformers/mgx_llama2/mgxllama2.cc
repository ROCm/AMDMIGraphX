#include "buffer.hpp"
#include "common.hpp"
#include "dataset.hpp"
#include "llama2inputs.hpp"
#include "llama2outputs.hpp"
#include "utils.hpp"
#include <migraphx/migraphx.hpp>


struct MGXLlama2
{
    MGXLlama2()
    {
        check_hip_status(hipSetDevice(DEVICE_ID));
        check_hip_status(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

        loadPrograms();

        model_inputs = std::make_unique<LLama2Inputs>(*prog, prog_args, offload_copy);
        model_inputs->prepareOneDimProgArgs(progSimpleInput, prog_args_one_dim);
        if (not offload_copy)
        {
            model_inputs->upload_to_device(stream);
            model_inputs->one_dim_input_buffer->upload_to_device(stream);
        }

        model_outputs = std::make_unique<LLama2Outputs>(offload_copy);
        model_outputs->prepareProgArgs(prog_args,prog_args_one_dim);
        // setting up argmax arguments
        model_outputs->prepareProgArgsArgMax(prog_args_argmax, prog_args_argmax_one_dim);
    }

    void run()
    {
        std::cout << "Dataset size: " << model_inputs->dataSize() << std::endl;
        std::cout << "Starting evaluation" << std::endl;
        size_t token_count = 0;
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < model_inputs->dataSize(); ++i)
        {
            evaluateSample(i, token_count);

    #ifdef TRACE
            std::cout << "######### Output token ids for #" << i << " #########" << std::endl;
            // print output tokens
            for (auto tok: output_tokens){
                std::cout << tok << ", ";
            }
            std::cout << std::endl;
    #endif
            prepareNextSample();
        }

        float dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() / 1000.f;
        std::cout << "Duration: " << dur << " seconds." << std::endl;
        std::cout << "Completed " << token_count << " tokens." << std::endl;
        std::cout << "Tokens/sec: " << token_count / dur << std::endl;

        if (WRITE_RESULT_FILE)
        {
            writeResults(results);
        }
    }

    void loadPrograms()
    {
        std::cout << "Offload copy: " << std::boolalpha << offload_copy << std::endl;
        ModelLoadSettings settings = {SEQ_SIZE, false /*quantize_fp16*/, offload_copy /*offload_copy*/, false /*fast_math*/, false /*input_one_dim*/};
        progMultipleInputDim = loadProgram(settings);
        std::cout << "Model loaded" << std::endl;
        progArgMaxMultipleInputDim = create_argmax_program(settings);
        std::cout << "ArgMax model created" << std::endl;

        // Load {1,1} input_ids model
        settings.input_one_dim = true;
        progSimpleInput = loadProgram(settings);
        std::cout << "Model 1 dim input loaded" << std::endl;
        progArgMaxSimpleInput = create_argmax_program(settings);
        std::cout << "ArgMax model for 1 dim model created" << std::endl;

        prog = &progMultipleInputDim;
        progArgMax = &progArgMaxMultipleInputDim;
    }

    void evaluateSample(size_t sample_id, size_t& token_count)
    {
        #ifdef TRACE
        std::cout << "Iter #" << sample_id << std::endl;
        #endif
        auto lastInputIdx = model_inputs->getLastInputIndex();
        for (size_t i = lastInputIdx; i < SEQ_SIZE - 1; ++i)
        {
            bool firstIter = (i == lastInputIdx);
            prog->run_async(firstIter ? prog_args : prog_args_one_dim, stream);
            auto outputs = progArgMax->run_async(firstIter ? prog_args_argmax : prog_args_argmax_one_dim, stream);
            if (not offload_copy)
            {
                firstIter ? model_outputs->argm_output_buffer->download_from_device(stream, i, i + 1) : model_outputs->argm_output_buffer_one_dim->download_from_device(stream);
            }

            check_hip_status(hipStreamSynchronize(stream));
            int64_t* results = offload_copy ? reinterpret_cast<int64_t*>(outputs[0].data()) : reinterpret_cast<int64_t*>( firstIter ? model_outputs->argm_output_buffer->hbuff.data() : model_outputs->argm_output_buffer_one_dim->hbuff.data());
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

            model_inputs->attention_mask_buffer->update_data(1, i + 1, stream);
            model_inputs->one_dim_input_buffer->update_data(new_token, 0, stream);

            if (firstIter)
            {
                prog = &progSimpleInput;
                progArgMax = &progArgMaxSimpleInput;
            }
        }
    }

    void prepareNextSample()
    {
        prog = &progMultipleInputDim;
        progArgMax = &progArgMaxMultipleInputDim;

        auto updated = model_inputs->updateData(*prog, prog_args);

        if (updated && not offload_copy)
        {
            model_inputs->upload_to_device(stream);
        }
        results.emplace_back(output_tokens);
        output_tokens.clear();
    }

    MGXLlama2(const MGXLlama2 &buf) = delete;
    MGXLlama2 &operator=(const MGXLlama2 &buf) = delete;

    migraphx::program progMultipleInputDim;
    migraphx::program progArgMaxMultipleInputDim;
    migraphx::program progSimpleInput;
    migraphx::program progArgMaxSimpleInput;
    migraphx::program *prog = nullptr;
    migraphx::program *progArgMax = nullptr;

    migraphx::program_parameters prog_args;
    migraphx::program_parameters prog_args_one_dim;
    migraphx::program_parameters prog_args_argmax;
    migraphx::program_parameters prog_args_argmax_one_dim;

    std::vector<std::vector<uint64_t>> results;
    std::vector<uint64_t> output_tokens;
    hipStream_t stream;
    bool offload_copy = false;

    std::unique_ptr<LLama2Inputs> model_inputs;
    std::unique_ptr<LLama2Outputs> model_outputs;
};


int main()
{
    MGXLlama2 mgxllama2;
    mgxllama2.run();
    return 0;
}
