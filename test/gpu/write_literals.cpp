/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>
#include <migraphx/generate.hpp>
#include <test.hpp>

void run_pass(migraphx::module& m, migraphx::gpu::write_literals p = {}) { p.apply(m); }

TEST_CASE(single_literal_basic)
{
    migraphx::module m1;
    {
        auto lit = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2, 3}}, 1));
        m1.add_return({lit});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto gpu_lit = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2, 3}}, 1))},
                               {"host", false}}));
        m2.add_return({gpu_lit});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(multiple_literals)
{
    migraphx::module m1;
    {
        auto lit1 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2}}, 1));
        auto lit2 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::int32_type, {3}}, 2));
        auto lit3 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1, 4}}, 3));
        m1.add_return({lit1, lit2, lit3});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto gpu_lit1 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2}}, 1))},
                               {"host", false}}));
        auto gpu_lit2 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::int32_type, {3}}, 2))},
                               {"host", false}}));
        auto gpu_lit3 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {1, 4}}, 3))},
                               {"host", false}}));
        m2.add_return({gpu_lit1, gpu_lit2, gpu_lit3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(memory_limit_single_literal)
{
    migraphx::module m1;
    {
        // Create a literal of 24 bytes (6 floats * 4 bytes each)
        auto lit = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2, 3}}, 4));
        m1.add_return({lit});
    }
    // Set max memory to something very small to force copy
    run_pass(m1, {.max_memory = 10});

    migraphx::module m2;
    {
        auto gpu_lit = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2, 3}}, 4))},
                               {"host", true}}));
        auto alloc = m2.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape", migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {2, 3}})}}));
        auto copy  = m2.add_instruction(migraphx::make_op("hip::copy"), gpu_lit, alloc);
        m2.add_return({copy});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(memory_limit_multiple_literals)
{
    migraphx::module m1;
    {
        // Create multiple literals with different sizes
        auto lit1 = m1.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {10}}, 5)); // 40 bytes
        auto lit2 = m1.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {5}}, 6)); // 20 bytes
        auto lit3 = m1.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {2}}, 7)); // 8 bytes
        m1.add_return({lit1, lit2, lit3});
    }
    // Set memory limit to force all literals to be copied due to memory estimation
    run_pass(m1, {.max_memory = 100});

    migraphx::module m2;
    {
        auto gpu_lit1 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {10}}, 5))},
                               {"host", true}}));
        auto alloc1   = m2.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape", migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {10}})}}));
        auto copy1    = m2.add_instruction(migraphx::make_op("hip::copy"), gpu_lit1, alloc1);
        auto gpu_lit2 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {5}}, 6))},
                               {"host", true}}));
        auto alloc2   = m2.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape", migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {5}})}}));
        auto copy2    = m2.add_instruction(migraphx::make_op("hip::copy"), gpu_lit2, alloc2);
        auto gpu_lit3 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2}}, 7))},
                               {"host", true}}));
        auto alloc3 = m2.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape", migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {2}})}}));
        auto copy3  = m2.add_instruction(migraphx::make_op("hip::copy"), gpu_lit3, alloc3);
        m2.add_return({copy1, copy2, copy3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(empty_module)
{
    migraphx::module m1;
    run_pass(m1);

    migraphx::module m2;
    EXPECT(m1 == m2);
}

TEST_CASE(no_literals)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto y   = m1.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto sum = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_return({sum});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        // Parameters may be sorted during the pass
        auto y   = m2.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto x   = m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto sum = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({sum});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(mixed_instructions_with_literals)
{
    migraphx::module m1;
    {
        auto lit1 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2, 3}}, 8));
        auto x    = m1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto sum  = m1.add_instruction(migraphx::make_op("add"), lit1, x);
        auto lit2 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2, 3}}, 9));
        auto mul = m1.add_instruction(migraphx::make_op("mul"), sum, lit2);
        m1.add_return({mul});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto gpu_lit1 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2, 3}}, 8))},
                               {"host", false}}));
        auto x        = m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto sum      = m2.add_instruction(migraphx::make_op("add"), gpu_lit1, x);
        auto gpu_lit2 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2, 3}}, 9))},
                               {"host", false}}));
        auto mul = m2.add_instruction(migraphx::make_op("mul"), sum, gpu_lit2);
        m2.add_return({mul});
    }
    // Sort both modules before comparison due to instruction ordering differences
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(very_large_literal)
{
    migraphx::module m1;
    {
        // Create a very large literal (1000 floats = 4000 bytes)
        auto lit = m1.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {10, 100}}, 10));
        m1.add_return({lit});
    }
    // Set max memory to force copy
    run_pass(m1, {.max_memory = 2000});

    migraphx::module m2;
    {
        auto gpu_lit = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {10, 100}}, 10))},
                               {"host", true}}));
        auto alloc = m2.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape",
              migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {10, 100}})}}));
        auto copy  = m2.add_instruction(migraphx::make_op("hip::copy"), gpu_lit, alloc);
        m2.add_return({copy});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(different_types_literals)
{
    migraphx::module m1;
    {
        auto lit_float = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2}}, 11));
        auto lit_int32 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::int32_type, {2}}, 12));
        auto lit_int8 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::int8_type, {4}}, 13));
        auto lit_half = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::half_type, {2}}, 14));
        m1.add_return({lit_float, lit_int32, lit_int8, lit_half});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto gpu_lit_float = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2}}, 11))},
                               {"host", false}}));
        auto gpu_lit_int32 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::int32_type, {2}}, 12))},
                               {"host", false}}));
        auto gpu_lit_int8 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::int8_type, {4}}, 13))},
                               {"host", false}}));
        auto gpu_lit_half = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::half_type, {2}}, 14))},
                               {"host", false}}));
        m2.add_return({gpu_lit_float, gpu_lit_int32, gpu_lit_int8, gpu_lit_half});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(zero_memory_limit)
{
    migraphx::module m1;
    {
        auto lit = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2}}, 15));
        m1.add_return({lit});
    }
    // Zero memory should use available GPU memory (default behavior)
    run_pass(m1, {.max_memory = 0});

    migraphx::module m2;
    {
        auto gpu_lit = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2}}, 15))},
                               {"host", false}}));
        m2.add_return({gpu_lit});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(scalar_literal)
{
    migraphx::module m1;
    {
        auto lit = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type}, 16));
        m1.add_return({lit});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto gpu_lit = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type}, 16))},
                               {"host", false}}));
        m2.add_return({gpu_lit});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(force_all_copy_memory_limit)
{
    migraphx::module m1;
    {
        auto lit1 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2}}, 17));
        auto lit2 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {3}}, 18));
        m1.add_return({lit1, lit2});
    }
    // Set memory to 1 byte to force all literals to be copied
    run_pass(m1, {.max_memory = 1});

    migraphx::module m2;
    {
        auto gpu_lit1 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2}}, 17))},
                               {"host", true}}));
        auto alloc1   = m2.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape", migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {2}})}}));
        auto copy1    = m2.add_instruction(migraphx::make_op("hip::copy"), gpu_lit1, alloc1);
        auto gpu_lit2 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {3}}, 18))},
                               {"host", true}}));
        auto alloc2 = m2.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape", migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {3}})}}));
        auto copy2  = m2.add_instruction(migraphx::make_op("hip::copy"), gpu_lit2, alloc2);
        m2.add_return({copy1, copy2});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(check_literal_order_preserved)
{
    migraphx::module m1;
    {
        // Create literals that will be used in specific order
        auto lit1 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2}}, 19));
        auto x    = m1.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2}});
        auto add1 = m1.add_instruction(migraphx::make_op("add"), lit1, x);
        auto lit2 = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {2}}, 20));
        auto add2 = m1.add_instruction(migraphx::make_op("add"), add1, lit2);
        m1.add_return({add2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto gpu_lit1 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2}}, 19))},
                               {"host", false}}));
        auto x        = m2.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2}});
        auto add1     = m2.add_instruction(migraphx::make_op("add"), gpu_lit1, x);
        auto gpu_lit2 = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {2}}, 20))},
                               {"host", false}}));
        auto add2 = m2.add_instruction(migraphx::make_op("add"), add1, gpu_lit2);
        m2.add_return({add2});
    }
    // Sort both modules before comparison due to instruction ordering differences
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(allocations_with_literals)
{
    migraphx::module m1;
    {
        // Add an allocation instruction before literals
        auto alloc = m1.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape", migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {10}})}}));
        auto lit   = m1.add_literal(
            migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {10}}, 21));
        auto copy = m1.add_instruction(migraphx::make_op("hip::copy"), lit, alloc);
        m1.add_return({copy});
    }
    // This should trigger memory estimation that includes existing allocations
    run_pass(m1, {.max_memory = 100});

    migraphx::module m2;
    {
        // The literal gets transformed and then copied into the existing allocation
        auto gpu_lit = m2.add_instruction(
            migraphx::make_op("gpu::literal",
                              {{"data",
                                migraphx::to_value(migraphx::generate_argument(
                                    migraphx::shape{migraphx::shape::float_type, {10}}, 21))},
                               {"host", true}}));
        auto alloc1 = m2.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape", migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {10}})}}));
        auto copy1  = m2.add_instruction(migraphx::make_op("hip::copy"), gpu_lit, alloc1);
        auto alloc2 = m2.add_instruction(migraphx::make_op(
            "hip::allocate",
            {{"shape", migraphx::to_value(migraphx::shape{migraphx::shape::float_type, {10}})}}));
        auto copy2  = m2.add_instruction(migraphx::make_op("hip::copy"), copy1, alloc2);
        m2.add_return({copy2});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }