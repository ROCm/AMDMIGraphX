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

#include <test.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

TEST_CASE(lstm_builder_returns_three_outputs)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;

    migraphx::module m;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

    auto seq  = m.add_parameter("seq", in_shape);
    auto w    = m.add_parameter("w", w_shape);
    auto r    = m.add_parameter("r", r_shape);
    auto bias = m.add_parameter("bias", b_shape);
    auto und  = m.add_instruction(migraphx::make_op("undefined"));
    auto ih   = m.add_parameter("ih", ih_shape);
    auto ic   = m.add_parameter("ic", ic_shape);

    auto results = migraphx::op::builder::add(
        "lstm",
        m,
        {seq, w, r, bias, und, ih, ic, und},
        {{"hidden_size", hidden_size},
         {"actv_func",
          migraphx::to_value(std::vector<migraphx::operation>{
              migraphx::make_op("sigmoid"), migraphx::make_op("tanh"), migraphx::make_op("tanh")})},
         {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
         {"clip", 0.0f},
         {"input_forget", 0}});

    EXPECT(results.size() == 3);

    // hidden_states: [seq_len, num_directions, batch_size, hidden_size]
    auto hs_shape = results.at(0)->get_shape();
    EXPECT(hs_shape.lens()[0] == seq_len);
    EXPECT(hs_shape.lens()[1] == num_dirct);
    EXPECT(hs_shape.lens()[2] == batch_size);
    EXPECT(hs_shape.lens()[3] == hidden_size);

    // last_hs_output: [num_directions, batch_size, hidden_size]
    auto lho_shape = results.at(1)->get_shape();
    EXPECT(lho_shape.lens().size() == 3);
    EXPECT(lho_shape.lens()[0] == num_dirct);
    EXPECT(lho_shape.lens()[1] == batch_size);
    EXPECT(lho_shape.lens()[2] == hidden_size);

    // last_cell_output: [num_directions, batch_size, hidden_size]
    auto lco_shape = results.at(2)->get_shape();
    EXPECT(lco_shape.lens().size() == 3);
    EXPECT(lco_shape.lens()[0] == num_dirct);
    EXPECT(lco_shape.lens()[1] == batch_size);
    EXPECT(lco_shape.lens()[2] == hidden_size);
}

TEST_CASE(lstm_builder_bidirectional_outputs)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;

    migraphx::module m;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

    auto seq  = m.add_parameter("seq", in_shape);
    auto w    = m.add_parameter("w", w_shape);
    auto r    = m.add_parameter("r", r_shape);
    auto bias = m.add_parameter("bias", b_shape);
    auto und  = m.add_instruction(migraphx::make_op("undefined"));
    auto ih   = m.add_parameter("ih", ih_shape);
    auto ic   = m.add_parameter("ic", ic_shape);

    auto results = migraphx::op::builder::add(
        "lstm",
        m,
        {seq, w, r, bias, und, ih, ic, und},
        {{"hidden_size", hidden_size},
         {"actv_func",
          migraphx::to_value(std::vector<migraphx::operation>{migraphx::make_op("sigmoid"),
                                                              migraphx::make_op("tanh"),
                                                              migraphx::make_op("tanh"),
                                                              migraphx::make_op("sigmoid"),
                                                              migraphx::make_op("tanh"),
                                                              migraphx::make_op("tanh")})},
         {"direction", migraphx::to_value(migraphx::op::rnn_direction::bidirectional)},
         {"clip", 0.0f},
         {"input_forget", 0}});

    EXPECT(results.size() == 3);

    auto hs_shape = results.at(0)->get_shape();
    EXPECT(hs_shape.lens()[0] == seq_len);
    EXPECT(hs_shape.lens()[1] == num_dirct);
    EXPECT(hs_shape.lens()[2] == batch_size);
    EXPECT(hs_shape.lens()[3] == hidden_size);
}

TEST_CASE(lstm_builder_numerical_forward)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 1;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

    auto seq  = mm->add_parameter("seq", in_shape);
    auto w    = mm->add_parameter("w", w_shape);
    auto r    = mm->add_parameter("r", r_shape);
    auto bias = mm->add_parameter("bias", b_shape);
    auto und  = mm->add_instruction(migraphx::make_op("undefined"));
    auto ih   = mm->add_parameter("ih", ih_shape);
    auto ic   = mm->add_parameter("ic", ic_shape);

    auto results = migraphx::op::builder::add(
        "lstm",
        *mm,
        {seq, w, r, bias, und, ih, ic, und},
        {{"hidden_size", hidden_size},
         {"actv_func",
          migraphx::to_value(std::vector<migraphx::operation>{
              migraphx::make_op("sigmoid"), migraphx::make_op("tanh"), migraphx::make_op("tanh")})},
         {"direction", migraphx::to_value(migraphx::op::rnn_direction::forward)},
         {"clip", 0.0f},
         {"input_forget", 0}});
    mm->add_return({results.at(0), results.at(1), results.at(2)});

    p.compile(migraphx::make_target("ref"));

    std::vector<float> seq_data(seq_len * batch_size * input_size, 0.1f);
    std::vector<float> w_data(num_dirct * 4 * hidden_size * input_size, 0.1f);
    std::vector<float> r_data(num_dirct * 4 * hidden_size * hidden_size, 0.1f);
    std::vector<float> b_data(num_dirct * 8 * hidden_size, 0.0f);
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0.0f);
    std::vector<float> ic_data(num_dirct * batch_size * hidden_size, 0.0f);

    migraphx::parameter_map params;
    params["seq"]  = migraphx::argument(in_shape, seq_data.data());
    params["w"]    = migraphx::argument(w_shape, w_data.data());
    params["r"]    = migraphx::argument(r_shape, r_data.data());
    params["bias"] = migraphx::argument(b_shape, b_data.data());
    params["ih"]   = migraphx::argument(ih_shape, ih_data.data());
    params["ic"]   = migraphx::argument(ic_shape, ic_data.data());

    auto result = p.eval(params);
    EXPECT(result.size() == 3);

    auto hs_result = result[0];
    EXPECT(hs_result.get_shape().lens()[0] == seq_len);
    EXPECT(hs_result.get_shape().lens()[1] == num_dirct);
    EXPECT(hs_result.get_shape().lens()[2] == batch_size);
    EXPECT(hs_result.get_shape().lens()[3] == hidden_size);
}
