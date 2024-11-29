#include "config.hpp"
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

        model_inputs = std::make_unique<LLama2Inputs>(*prog, prog_args);
        model_inputs->prepareOneDimProgArgs(progSimpleInput, prog_args_one_dim);
        model_inputs->upload_to_device(stream);
        model_inputs->one_dim_input_buffer->upload_to_device(stream);

        model_outputs = std::make_unique<LLama2Outputs>();
        model_outputs->prepareProgArgs(prog_args,prog_args_one_dim);
        // setting up argmax arguments
        model_outputs->prepareProgArgsArgMax(prog_args_argmax, prog_args_argmax_one_dim);
        output_tokens.resize(BATCH_SIZE, std::vector<uint64_t>());
    }

    void run()
    {
        std::cout << "Dataset size: " << model_inputs->dataSize() << std::endl;
        std::cout << "Number of batches: " << model_inputs->batchNum() << std::endl;
        std::cout << "Starting evaluation" << std::endl;
        size_t token_count = 0;
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < model_inputs->batchNum(); ++i)
        {
            evaluateBatch(i, token_count);

    #ifdef TRACE
            std::cout << "######### Output token ids for #" << i << " #########" << std::endl;
            printOutputTokens();
    #endif
            prepareNextBatch();
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
        ModelLoadSettings settings = {SEQ_SIZE, false /*quantize_fp16*/, false /*fast_math*/, false /*input_one_dim*/, BATCH_SIZE};
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

    void evaluateBatch(size_t batch_idx, size_t& token_count)
    {
        #ifdef TRACE
        std::cout << "Iter #" << batch_idx << std::endl;
        #endif

        std::vector<size_t> sampleLastInputIdx;
        std::vector<size_t> batches = getBatchesToProcess(batch_idx);
        
        for (size_t i = 0; i < SEQ_SIZE - 1; ++i)
        {
            bool firstIter = (i == 0);
            prog->run_async(firstIter ? prog_args : prog_args_one_dim, stream);
            progArgMax->run_async(firstIter ? prog_args_argmax : prog_args_argmax_one_dim, stream);

            auto batchIt = std::begin(batches);
            while (batchIt != std::end(batches))
            {
                auto b = *batchIt;
                if (firstIter)
                {
                    sampleLastInputIdx.emplace_back(model_inputs->getLastInputIndex(b) + (b * SEQ_SIZE));
                }

                firstIter ? model_outputs->argm_output_buffer->download_from_device(stream, sampleLastInputIdx[b], sampleLastInputIdx[b] + 1) : model_outputs->argm_output_buffer_one_dim->download_from_device(stream);

                check_hip_status(hipStreamSynchronize(stream));
                int64_t* results = reinterpret_cast<int64_t*>( firstIter ? model_outputs->argm_output_buffer->hbuff.data() : model_outputs->argm_output_buffer_one_dim->hbuff.data());
                auto new_token_idx = firstIter ? sampleLastInputIdx[b] : b;
                int64_t new_token = results[new_token_idx];

                token_count++;
                #ifdef TRACE
                std::cout << "New token for batch (" << b << "): " << new_token << std::endl;
                #endif
                output_tokens[b].push_back(new_token);

                if (new_token == EOS)
                {
                #ifdef TRACE
                    std::cout << b << " batch is added to finished" << std::endl;
                #endif
                    batchIt = batches.erase(batchIt);
                }
                else
                {
                    ++batchIt;
                }

                model_inputs->attention_mask_buffer->update_data(1, sampleLastInputIdx[b] + i + 1, stream);
                model_inputs->one_dim_input_buffer->update_data(new_token, b, stream);
            }

            if (batches.empty())
                break;

            if (firstIter)
            {
                prog = &progSimpleInput;
                progArgMax = &progArgMaxSimpleInput;
            }
        }
    }

    void prepareNextBatch()
    {
        prog = &progMultipleInputDim;
        progArgMax = &progArgMaxMultipleInputDim;

        auto updated = model_inputs->updateData(*prog, prog_args);

        if (updated)
        {
            model_inputs->upload_to_device(stream);
        }
        for (auto& tokens : output_tokens)
        {
            results.emplace_back(tokens);
            tokens.clear();
        }
    }

    std::vector<size_t> getBatchesToProcess(size_t batch_idx)
    {
        std::vector<size_t> batches;
        size_t batchSizeRem = BATCH_SIZE;
        if (batch_idx == model_inputs->batchNum() - 1)
        {
            batchSizeRem = model_inputs->dataSize() % BATCH_SIZE;
        }
        batches.resize(batchSizeRem);
        std::iota(std::begin(batches), std::end(batches), 0);
        return batches;
    }

    void printOutputTokens() const
    {
        // print output tokens
        for (size_t b = 0; b < output_tokens.size(); ++b)
        {
            if (output_tokens[b].empty())
                continue;

            std::cout << "######### Batch #" << b << " #########" << std::endl;
            for (auto tok: output_tokens[b])
            {
                std::cout << tok << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
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
    std::vector<std::vector<uint64_t>> output_tokens;
    hipStream_t stream;

    std::unique_ptr<LLama2Inputs> model_inputs;
    std::unique_ptr<LLama2Outputs> model_outputs;
};


int main()
{
    MGXLlama2 mgxllama2;
    mgxllama2.run();
    return 0;
}
