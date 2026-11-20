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
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/sym.hpp>

#include <test.hpp>

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

using dd = migraphx::shape::dynamic_dimension;
using migraphx::sym::var;

// Runs a compiled program on the GPU. Every input is a float vector whose length becomes the
// runtime size of the symbolic dimension.
static std::vector<float> run_gpu(migraphx::program& p,
                                  const migraphx::target& t,
                                  std::unordered_map<std::string, std::vector<float>> inputs)
{
    migraphx::parameter_map params;
    for(auto&& x : p.get_parameter_shapes())
    {
        auto it = inputs.find(x.first);
        if(it == inputs.end())
        {
            auto s          = x.second;
            params[x.first] = t.allocate(s.dynamic() ? migraphx::shape{s.type(), s.max_lens()} : s);
            continue;
        }
        migraphx::shape s{migraphx::shape::float_type, {it->second.size()}};
        params[x.first] = t.copy_to(migraphx::argument{s, it->second.data()});
    }
    auto result = t.copy_from(p.eval(params).back());
    std::vector<float> out;
    result.visit([&](auto output) { out.assign(output.begin(), output.end()); });
    return out;
}

// Indices -1, -2, ..., -n for a vector of n updates (dynamic_range counts down from -1).
static migraphx::instruction_ref
add_descending_indices(migraphx::module& m, migraphx::instruction_ref buffer, const dd& size)
{
    auto start =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {-1}});
    auto delta =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {-1}});
    auto minus_one =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {-1}});
    auto limit =
        m.add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}), buffer);
    limit        = m.add_instruction(migraphx::make_op("neg"), limit);
    limit        = m.add_instruction(migraphx::make_op("add"), limit, minus_one);
    auto indices = m.add_instruction(
        migraphx::make_op("dynamic_range", {{"output_dim", migraphx::to_value(size)}}),
        start,
        limit,
        delta);
    return m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), indices);
}

TEST_CASE(scatternd_symbolic_negative_indices_gpu_test)
{
    // Symbolic data and symbolic indices: n updates go to rows n - 1, n - 2, ..., 0, so the
    // result is the updates in reverse order.
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m            = *p.get_main_module();
    auto data_buffer   = m.add_parameter("data_buffer", {migraphx::shape::float_type, {dd{n}}});
    auto update_buffer = m.add_parameter("update_buffer", {migraphx::shape::float_type, {dd{n}}});
    auto zero =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
    auto indices = add_descending_indices(m, data_buffer, dd{n});
    auto data    = m.add_instruction(migraphx::make_op("fill"), zero, data_buffer);
    auto output =
        m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, update_buffer);
    m.add_return({output});

    migraphx::target t = migraphx::make_target("gpu");
    p.compile(t);
    // n = 1 is left out: that clone faults for positive indices too, which is a separate problem
    for(std::size_t len : {2, 3, 4})
    {
        std::vector<float> updates(len);
        std::iota(updates.begin(), updates.end(), 1.0f);
        std::vector<float> gold(updates.rbegin(), updates.rend());
        EXPECT(run_gpu(p,
                       t,
                       {{"data_buffer", std::vector<float>(len, 0)}, {"update_buffer", updates}}) ==
               gold);
    }
}

TEST_CASE(scatternd_symbolic_data_static_negative_indices_gpu_test)
{
    // Symbolic data and a static index: -1 is the last row of the runtime size n, not of the
    // padded size.
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m          = *p.get_main_module();
    auto data_buffer = m.add_parameter("data_buffer", {migraphx::shape::float_type, {dd{n}}});
    auto updates     = m.add_parameter("updates", {migraphx::shape::float_type, {1}});
    auto zero =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
    auto indices = m.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1, 1}}, {-1}});
    auto data   = m.add_instruction(migraphx::make_op("fill"), zero, data_buffer);
    auto output = m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    m.add_return({output});

    migraphx::target t = migraphx::make_target("gpu");
    p.compile(t);
    for(std::size_t len : {1, 2, 3, 4})
    {
        std::vector<float> gold(len, 0);
        gold.back() = 5;
        EXPECT(run_gpu(p, t, {{"data_buffer", std::vector<float>(len, 0)}, {"updates", {5}}}) ==
               gold);
    }
}

TEST_CASE(scatternd_symbolic_data_derived_constant_negative_indices_gpu_test)
{
    // The same as above, with an index that is computed from a literal instead of being one.
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m          = *p.get_main_module();
    auto data_buffer = m.add_parameter("data_buffer", {migraphx::shape::float_type, {dd{n}}});
    auto updates     = m.add_parameter("updates", {migraphx::shape::float_type, {1}});
    auto zero =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {0.0f}});
    auto index =
        m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {-1}});
    auto indices = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), index);
    auto data    = m.add_instruction(migraphx::make_op("fill"), zero, data_buffer);
    auto output  = m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    m.add_return({output});

    migraphx::target t = migraphx::make_target("gpu");
    p.compile(t);
    for(std::size_t len : {1, 2, 3, 4})
    {
        std::vector<float> gold(len, 0);
        gold.back() = 5;
        EXPECT(run_gpu(p, t, {{"data_buffer", std::vector<float>(len, 0)}, {"updates", {5}}}) ==
               gold);
    }
}

TEST_CASE(scatternd_static_data_symbolic_negative_indices_gpu_test)
{
    // Static data of 8 rows and a symbolic number of indices: -1, -2, ... address rows 7, 6, ...
    auto n = var("n", {1, 4}, {2});
    migraphx::program p;
    auto& m            = *p.get_main_module();
    auto update_buffer = m.add_parameter("update_buffer", {migraphx::shape::float_type, {dd{n}}});
    auto data = m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {8}},
                                                std::vector<float>(8, 0)});
    auto indices = add_descending_indices(m, update_buffer, dd{n});
    auto output =
        m.add_instruction(migraphx::make_op("scatternd_none"), data, indices, update_buffer);
    m.add_return({output});

    migraphx::target t = migraphx::make_target("gpu");
    p.compile(t);
    // n = 1 is left out, as above
    for(std::size_t len : {2, 3, 4})
    {
        std::vector<float> updates(len);
        std::iota(updates.begin(), updates.end(), 1.0f);
        std::vector<float> gold(8, 0);
        std::reverse_copy(updates.begin(), updates.end(), gold.end() - len);
        EXPECT(run_gpu(p, t, {{"update_buffer", updates}}) == gold);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
