#pragma once

#include "config.hpp"
#include "utils.hpp"

struct LLama2Inputs
{
    LLama2Inputs(
        migraphx::program& prog,
        migraphx::program_parameters& prog_args)
    {
        data.initialize();
        prepareProgArgs(prog, prog_args);
    }

    void prepareProgArgs(migraphx::program& prog, migraphx::program_parameters& prog_args, bool simple = false)
    {
        auto param_shapes = prog.get_parameter_shapes();
        if (!simple)
        {
            auto inputShape = param_shapes[INPUTS_ID_STR];
            auto input_ids = data.getInputIds();
            input_ids_buffer = std::make_unique<LLama2InputBuffer>(std::move(input_ids));
            prog_args.add(INPUTS_ID_STR, migraphx::argument(inputShape, input_ids_buffer->data()));
        }


        auto attShape = param_shapes[ATTENTION_MASK_STR];
        auto attention_mask = data.getAttentionMask();
        if (!simple)
        {
            attention_mask_buffer = std::make_unique<LLama2InputBuffer>(std::move(attention_mask));
        }
        prog_args.add(ATTENTION_MASK_STR, migraphx::argument(attShape, attention_mask_buffer->data()));

        // past_key_values.0.key = @param:past_key_values.0.key -> half_type, {1, 32, 1, 128}, {4096, 128, 128, 1}
        // past_key_values.0.value = @param:past_key_values.0.value -> half_type, {1, 32, 1, 128}, {4096, 128, 128, 1}
        for (size_t i = 0; i < HIDDEN_LAYERS_NUM; ++i)
        {
            auto past_keyStr = getPastKeyString(i);
            auto past_keyString = past_keyStr.c_str();
            if (!simple)
            {
                past_key_buffers.emplace_back(std::make_unique<LLama2PastKeyValueBuffer>(std::vector<half>(PAST_KEY_VAL_SIZE, 0.0_h)));
            }
            auto pastKeyShape = param_shapes[past_keyString];
            prog_args.add(past_keyString, migraphx::argument(pastKeyShape, past_key_buffers[i]->data()));

            auto past_valueStr = getPastValueStr(i);
            auto past_valueString = past_valueStr.c_str();
            if (!simple)
            {
                past_value_buffers.emplace_back(std::make_unique<LLama2PastKeyValueBuffer>(std::vector<half>(PAST_KEY_VAL_SIZE, 0.0_h)));
            }
            auto pastValueShape = param_shapes[past_valueString];
            prog_args.add(past_valueString, migraphx::argument(pastValueShape, past_value_buffers[i]->data()));
        }
    }

    void prepareOneDimProgArgs(migraphx::program& prog, migraphx::program_parameters& prog_args)
    {
        prepareProgArgs(prog, prog_args, true);
        auto param_shapes = prog.get_parameter_shapes();
        auto inputShape = param_shapes[INPUTS_ID_STR];
        std::vector<int64_t> oneDimInput = {0};
        one_dim_input_buffer = std::make_unique<LLama2InputBuffer>(std::move(oneDimInput));
        prog_args.add(INPUTS_ID_STR, migraphx::argument(inputShape, one_dim_input_buffer->data()));
    }

    void upload_to_device(hipStream_t stream)
    {
        input_ids_buffer->upload_to_device(stream);
        attention_mask_buffer->upload_to_device(stream);
    }

    bool updateData(migraphx::program& prog, migraphx::program_parameters& prog_args)
    {
        auto currentIdx = data.currentIdx();
        if (currentIdx != data.getNext())
        {
            auto param_shapes = prog.get_parameter_shapes();

            std::vector<int64_t> input_ids = data.getInputIds();
            input_ids_buffer = std::make_unique<LLama2InputBuffer>(std::move(input_ids));
            prog_args.add(INPUTS_ID_STR, migraphx::argument(param_shapes[INPUTS_ID_STR], input_ids_buffer->data()));

            auto attention_mask = data.getAttentionMask();
            attention_mask_buffer->update(std::move(attention_mask));

            return true;
        }
        return false;
    }

    void resetPastKeyValueBuffers(hipStream_t stream)
    {
        for (size_t i = 0; i < HIDDEN_LAYERS_NUM; ++i)
        {
            past_key_buffers[i]->update(std::vector<half>(PAST_KEY_VAL_SIZE, 0.0_h));
            past_value_buffers[i]->update(std::vector<half>(PAST_KEY_VAL_SIZE, 0.0_h));
            past_key_buffers[i]->upload_to_device(stream);
            past_value_buffers[i]->upload_to_device(stream);
        }
    }

    size_t getLastInputIndex() const { return data.getLastIdx(); }
    size_t dataSize() const { return data.size(); }

    LLama2Inputs() = delete;
    LLama2Inputs(const LLama2Inputs &buf) = delete;
    LLama2Inputs &operator=(const LLama2Inputs &buf) = delete;

    std::unique_ptr<LLama2InputBuffer> input_ids_buffer;
    std::unique_ptr<LLama2InputBuffer> one_dim_input_buffer;
    std::unique_ptr<LLama2InputBuffer> attention_mask_buffer;
    std::vector<std::unique_ptr<LLama2PastKeyValueBuffer>> past_key_buffers;
    std::vector<std::unique_ptr<LLama2PastKeyValueBuffer>> past_value_buffers;
    Dataset data;

    const char* INPUTS_ID_STR = "input_ids";
    const char* ATTENTION_MASK_STR = "attention_mask";
};
