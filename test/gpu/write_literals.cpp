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
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/value.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m, migraphx::gpu::write_literals p = {})
{
    migraphx::run_passes(m, {p, migraphx::dead_code_elimination{}});
}

// An input literal that the pass should rewrite.
static migraphx::instruction_ref
add_literal(migraphx::module& m, const migraphx::shape& s, unsigned long seed)
{
    return m.add_literal(migraphx::generate_literal(s, seed));
}

// The gpu::literal the pass produces when the literal stays in GPU memory.
static migraphx::instruction_ref add_gpu_literal(migraphx::module& m,
                                                 const migraphx::shape& s,
                                                 unsigned long seed,
                                                 bool host = false)
{
    return m.add_instruction(migraphx::make_op(
        "gpu::literal",
        {{"data", migraphx::to_value(migraphx::generate_argument(s, seed))}, {"host", host}}));
}

// The host-literal + allocate + copy sequence the pass produces under memory pressure.
static migraphx::instruction_ref
add_host_copy(migraphx::module& m, const migraphx::shape& s, unsigned long seed)
{
    auto lit = add_gpu_literal(m, s, seed, true);
    auto alloc =
        m.add_instruction(migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s)}}));
    return m.add_instruction(migraphx::make_op("hip::copy"), lit, alloc);
}

TEST_CASE(single_literal)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        m1.add_return({add_literal(m1, s, 1)});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        m2.add_return({add_gpu_literal(m2, s, 1)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(scalar_literal)
{
    migraphx::shape s{migraphx::shape::float_type};

    migraphx::module m1;
    {
        m1.add_return({add_literal(m1, s, 16)});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        m2.add_return({add_gpu_literal(m2, s, 16)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(multiple_literals)
{
    migraphx::shape sf{migraphx::shape::float_type, {2}};
    migraphx::shape si{migraphx::shape::int32_type, {3}};
    migraphx::shape sb{migraphx::shape::int8_type, {4}};
    migraphx::shape sh{migraphx::shape::half_type, {2}};

    migraphx::module m1;
    {
        m1.add_return({add_literal(m1, sf, 11),
                       add_literal(m1, si, 12),
                       add_literal(m1, sb, 13),
                       add_literal(m1, sh, 14)});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        m2.add_return({add_gpu_literal(m2, sf, 11),
                       add_gpu_literal(m2, si, 12),
                       add_gpu_literal(m2, sb, 13),
                       add_gpu_literal(m2, sh, 14)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(literal_with_ops)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto sum = m1.add_instruction(migraphx::make_op("add"), add_literal(m1, s, 8), x);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), sum, add_literal(m1, s, 9));
        m1.add_return({mul});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto sum = m2.add_instruction(migraphx::make_op("add"), add_gpu_literal(m2, s, 8), x);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), sum, add_gpu_literal(m2, s, 9));
        m2.add_return({mul});
    }
    // The pass sorts the module, so sort both before comparing.
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(no_literals)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto sum = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_return({sum});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        // The pass sorts the module, which orders parameters by name.
        auto y   = m2.add_parameter("y", s);
        auto x   = m2.add_parameter("x", s);
        auto sum = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({sum});
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

TEST_CASE(memory_limit_copies_literal)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        m1.add_return({add_literal(m1, s, 4)});
    }
    // A tiny memory limit forces the literal to be staged in host memory.
    run_pass(m1, {.max_memory = 10});

    migraphx::module m2;
    {
        m2.add_return({add_host_copy(m2, s, 4)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(memory_limit_copies_all_literals)
{
    migraphx::shape s1{migraphx::shape::float_type, {2}};
    migraphx::shape s2{migraphx::shape::float_type, {3}};

    migraphx::module m1;
    {
        m1.add_return({add_literal(m1, s1, 17), add_literal(m1, s2, 18)});
    }
    // A 1-byte limit forces every literal onto the host.
    run_pass(m1, {.max_memory = 1});

    migraphx::module m2;
    {
        m2.add_return({add_host_copy(m2, s1, 17), add_host_copy(m2, s2, 18)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(existing_allocation)
{
    migraphx::shape s{migraphx::shape::float_type, {10}};

    migraphx::module m1;
    {
        auto alloc = m1.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s)}}));
        auto copy =
            m1.add_instruction(migraphx::make_op("hip::copy"), add_literal(m1, s, 21), alloc);
        m1.add_return({copy});
    }
    run_pass(m1, {.max_memory = 100});

    migraphx::module m2;
    {
        // The literal is staged on the host, then copied into the existing allocation.
        auto host  = add_host_copy(m2, s, 21);
        auto alloc = m2.add_instruction(
            migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(s)}}));
        auto copy = m2.add_instruction(migraphx::make_op("hip::copy"), host, alloc);
        m2.add_return({copy});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(memory_limit_copies_some_literals)
{
    migraphx::shape small{migraphx::shape::float_type, {2}};
    migraphx::shape big{migraphx::shape::float_type, {10000}};

    migraphx::module m1;
    {
        m1.add_return(
            {add_literal(m1, small, 1), add_literal(m1, small, 2), add_literal(m1, big, 3)});
    }
    // The literals total ~40KB; this limit only fits the small ones on the GPU, so the pass
    // copies the single large literal (the last one) to the host and keeps the rest on the GPU.
    run_pass(m1, {.max_memory = 160000});

    migraphx::module m2;
    {
        m2.add_return({add_gpu_literal(m2, small, 1),
                       add_gpu_literal(m2, small, 2),
                       add_host_copy(m2, big, 3)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(memory_limit_copies_largest_literals)
{
    migraphx::shape small{migraphx::shape::float_type, {2}};
    migraphx::shape big{migraphx::shape::float_type, {10000}};

    migraphx::module m1;
    {
        m1.add_return(
            {add_literal(m1, small, 1), add_literal(m1, big, 2), add_literal(m1, big, 3)});
    }
    // Only one large literal fits on the GPU at this limit. The pass walks literals from the end,
    // so it copies the two large literals to the host and keeps the small one on the GPU.
    run_pass(m1, {.max_memory = 160000});

    migraphx::module m2;
    {
        m2.add_return(
            {add_gpu_literal(m2, small, 1), add_host_copy(m2, big, 2), add_host_copy(m2, big, 3)});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
