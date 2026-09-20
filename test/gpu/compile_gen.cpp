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
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>

static const auto find_fast_axis = test::make_function("find_fast_axis", [](auto&&... xs) {
    return migraphx::gpu::gen::find_fast_axis(static_cast<decltype(xs)>(xs)...);
});

TEST_CASE(test_find_fast_axis)
{
    EXPECT(find_fast_axis(migraphx::shape{migraphx::shape::float_type, {2, 2, 2, 6, 3}}) == 4);
    EXPECT(find_fast_axis(migraphx::shape{
               migraphx::shape::float_type, {2, 2, 2, 6, 3}, {72, 6, 1, 12, 2}}) == 2);
    EXPECT(find_fast_axis(
               migraphx::shape{migraphx::shape::float_type, {64, 512, 32, 32}, {0, 1, 0, 0}}) == 1);
    EXPECT(find_fast_axis(
               migraphx::shape{migraphx::shape::float_type, {64, 512, 32, 32}, {0, 0, 0, 0}}) == 3);
}

// A fused int4 dequant reduce: unpack -> pointwise(dequantizelinear, mul) -> reduce_sum,
// with the zero point either a literal of the pointwise or a tensor input
static migraphx::program make_unpack_reduce(bool literal_zero_point)
{
    migraphx::program p;
    auto* pm = p.create_module("pw");
    {
        auto x0 = pm->add_parameter("x0", migraphx::shape{migraphx::shape::uint8_type});
        auto x1 = pm->add_parameter("x1", migraphx::shape{migraphx::shape::half_type});
        auto x2 = pm->add_parameter("x2", migraphx::shape{migraphx::shape::half_type});
        migraphx::instruction_ref zp;
        if(literal_zero_point)
            zp = pm->add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::uint8_type}, {8}});
        else
            zp = pm->add_parameter("x3", migraphx::shape{migraphx::shape::uint8_type});
        auto dq  = pm->add_instruction(migraphx::make_op("dequantizelinear"), x0, x1, zp);
        auto mul = pm->add_instruction(migraphx::make_op("mul"), dq, x2);
        pm->add_return({mul});
    }
    auto* rm = p.create_module("reduce");
    {
        migraphx::shape s{migraphx::shape::half_type, {1, 4, 32}};
        auto packed = rm->add_parameter("x0", {migraphx::shape::uint8_type, {1, 4, 16}});
        std::vector<migraphx::instruction_ref> inputs = {
            rm->add_instruction(migraphx::make_op("unpack_int4"), packed),
            rm->add_parameter("x1", s),
            rm->add_parameter("x2", s)};
        if(not literal_zero_point)
            inputs.push_back(rm->add_parameter("x3", {migraphx::shape::uint8_type, {1, 4, 32}}));
        auto pw = rm->add_instruction(migraphx::make_op("pointwise"), inputs, {pm});
        auto rs = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), pw);
        rm->add_return({rs});
    }
    // Fused modules are bypassed by the program passes, as in a compiled program
    pm->set_bypass();
    rm->set_bypass();
    return p;
}

TEST_CASE(generate_reduce_unpack_int4_literal_zero_point)
{
    auto p   = make_unpack_reduce(true);
    auto src = migraphx::gpu::gen::generate_reduce(*p.get_module("reduce"), "kernel");
    EXPECT(src.find("unpack_int4_as<half>(x, half(-8))") != std::string::npos);
    // The zero point is folded into the unpack, not added in the pointwise
    EXPECT(src.find("half(-8)") == src.rfind("half(-8)"));
}

TEST_CASE(generate_reduce_unpack_int4_zero_point_tensor)
{
    auto p   = make_unpack_reduce(false);
    auto src = migraphx::gpu::gen::generate_reduce(*p.get_module("reduce"), "kernel");
    EXPECT(src.find("unpack_int4_as<half>(x, half(0))") != std::string::npos);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
