/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/simplify_symbolic_dimensions.hpp>
#include <migraphx/split_sym_dim.hpp>
#include <onnx_test.hpp>

// The mask, of length attention_mask_sequence_length, is added to the scores of the past keys
// and the new key, of length past_sequence_length + 1, as in a decoder with a KV cache
static void run_symbolic_mask_add(const std::string& name)
{
    migraphx::onnx_options options;
    options.use_symbolic_shapes = true;
    options.dim_params          = {{"past_sequence_length", {1, 64}},
                                   {"attention_mask_sequence_length", {2, 65}},
                                   {"total_sequence_length", {2, 65}}};
    auto p                      = read_onnx(name, options);
    migraphx::run_passes(p,
                         {migraphx::simplify_symbolic_dimensions{},
                          migraphx::dead_code_elimination{},
                          migraphx::split_sym_dim{},
                          migraphx::dead_code_elimination{}});
    p.compile(migraphx::make_target("ref"));

    const std::size_t heads    = 2;
    const std::size_t head_dim = 4;
    for(std::size_t past : {1, 2, 7, 64})
    {
        const std::size_t keys = past + 1;
        std::vector<float> q(heads * head_dim);
        std::vector<float> k(heads * head_dim);
        std::vector<float> past_key(heads * past * head_dim);
        std::vector<float> mask(keys);
        for(std::size_t i = 0; i < q.size(); ++i)
            q[i] = static_cast<float>(i % 3) - 1.0f;
        for(std::size_t i = 0; i < k.size(); ++i)
            k[i] = static_cast<float>(i % 5) - 2.0f;
        for(std::size_t i = 0; i < past_key.size(); ++i)
            past_key[i] = static_cast<float>(i % 7) - 3.0f;
        for(std::size_t i = 0; i < mask.size(); ++i)
            mask[i] = static_cast<float>(i);

        migraphx::parameter_map params;
        params["q"] =
            migraphx::argument{{migraphx::shape::float_type, {1, heads, 1, head_dim}}, q.data()};
        params["k"] =
            migraphx::argument{{migraphx::shape::float_type, {1, heads, 1, head_dim}}, k.data()};
        params["past_key"] = migraphx::argument{
            {migraphx::shape::float_type, {1, heads, past, head_dim}}, past_key.data()};
        params["mask"] =
            migraphx::argument{{migraphx::shape::float_type, {1, 1, 1, keys}}, mask.data()};
        auto result = p.eval(params).back();

        std::vector<float> gold(heads * keys);
        for(std::size_t h = 0; h < heads; ++h)
        {
            for(std::size_t j = 0; j < keys; ++j)
            {
                const float* key =
                    j < past ? &past_key[(h * past + j) * head_dim] : &k[h * head_dim];
                float sum = mask[j];
                for(std::size_t d = 0; d < head_dim; ++d)
                    sum += q[h * head_dim + d] * key[d];
                gold[h * keys + j] = sum;
            }
        }
        EXPECT(result.get_shape().lens() == std::vector<std::size_t>{1, heads, 1, keys});
        EXPECT(result.to_vector<float>() == gold);

        // A mask whose length is in range but not past_sequence_length + 1 is rejected
        const std::size_t wrong = past + 2 <= 65 ? past + 2 : past;
        std::vector<float> wrong_mask(wrong);
        params["mask"] =
            migraphx::argument{{migraphx::shape::float_type, {1, 1, 1, wrong}}, wrong_mask.data()};
        EXPECT(test::throws([&] { std::ignore = p.eval(params); }));
    }
}

TEST_CASE(symbolic_mask_add_test) { run_symbolic_mask_add("symbolic_mask_add_test.onnx"); }

TEST_CASE(symbolic_mask_add_mask_first_test)
{
    run_symbolic_mask_add("symbolic_mask_add_mask_first_test.onnx");
}
