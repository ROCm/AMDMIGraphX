#pragma once

#include "buffer.hpp"
#include "config.hpp"

#include <migraphx/migraphx.hpp>

struct LLama2Outputs
{
    LLama2Outputs(bool offload_copy)
        : offload_copy(offload_copy)
    {
    }

    void prepareProgArgs(migraphx::program_parameters& prog_args, migraphx::program_parameters& prog_args_one_dim)
    {
        output_buffer = std::make_unique<LLama2PastKeyValueBuffer>(std::vector<half>(OUTPUT_SIZE), offload_copy);
        one_dim_output_buffer = std::make_unique<LLama2PastKeyValueBuffer>(std::vector<half>(VOCAB_SIZE), offload_copy);
        migraphx::shape out_shape{migraphx_shape_half_type, {BATCH_SIZE, SEQ_SIZE, VOCAB_SIZE}};
        prog_args.add(OUTPUT_NAME, migraphx::argument(out_shape, output_buffer->data()));

        migraphx::shape x_shape_one_dim{migraphx_shape_half_type, {BATCH_SIZE, 1, VOCAB_SIZE}};
        prog_args_one_dim.add(OUTPUT_NAME, migraphx::argument(x_shape_one_dim, one_dim_output_buffer->data()));
    }

    void prepareProgArgsArgMax(migraphx::program_parameters& prog_args_argmax, migraphx::program_parameters& prog_args_argmax_one_dim)
    {
        // setting up argmax arguments
        migraphx::shape x_shape{migraphx_shape_half_type, {BATCH_SIZE, SEQ_SIZE, VOCAB_SIZE}};
        prog_args_argmax.add("x", migraphx::argument(x_shape, output_buffer->data()));
        argm_output_buffer = std::make_unique<ArgMaxOutputBuffer>(std::vector<int64_t>(VOCAB_SIZE), offload_copy);
        migraphx::shape argm_out_shape{migraphx_shape_int64_type, {BATCH_SIZE, SEQ_SIZE, 1}};
        prog_args_argmax.add(OUTPUT_NAME, migraphx::argument(argm_out_shape, argm_output_buffer->data()));

        migraphx::shape x_shape_one_dim{migraphx_shape_half_type, {BATCH_SIZE, 1, VOCAB_SIZE}};
        prog_args_argmax_one_dim.add("x", migraphx::argument(x_shape_one_dim, one_dim_output_buffer->data()));
        argm_output_buffer_one_dim = std::make_unique<ArgMaxOutputBuffer>(std::vector<int64_t>(1), offload_copy);
        migraphx::shape argm_out_shape_one_dim{migraphx_shape_int64_type, {BATCH_SIZE, 1, 1}};
        prog_args_argmax_one_dim.add(OUTPUT_NAME, migraphx::argument(argm_out_shape_one_dim, argm_output_buffer_one_dim->data()));
    }

    LLama2Outputs() = delete;
    LLama2Outputs(const LLama2Outputs &buf) = delete;
    LLama2Outputs &operator=(const LLama2Outputs &buf) = delete;

    std::unique_ptr<LLama2PastKeyValueBuffer> output_buffer;
    std::unique_ptr<LLama2PastKeyValueBuffer> one_dim_output_buffer;
    std::unique_ptr<ArgMaxOutputBuffer> argm_output_buffer;
    std::unique_ptr<ArgMaxOutputBuffer> argm_output_buffer_one_dim;

    bool offload_copy = false;

    const char* OUTPUT_NAME = "main:#output_0";
    const size_t OUTPUT_SIZE = BATCH_SIZE * SEQ_SIZE * VOCAB_SIZE;
};