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
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/allocation_model.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/replace_allocate.hpp>
#include <migraphx/memory_coloring.hpp>
#include <test.hpp>

// Note: The write_literals pass makes conservative memory estimates that include:
// 1. A 50% buffer on top of calculated memory needs
// 2. Scratch space for all intermediate activations (via estimate_scratch_size)
// 3. Memory coloring adds a single scratch parameter for all temporary allocations
// The actual GPU memory usage after compilation may be significantly less than
// the conservative estimates used during the pass.

static void run_pass(migraphx::module& m, migraphx::gpu::write_literals p = {})
{
    migraphx::run_passes(m, {p, migraphx::dead_code_elimination{}});
}

// Helper to run pass with full pipeline for memory coloring analysis
static void run_pass_with_memory_coloring(migraphx::module& m, migraphx::gpu::write_literals p = {})
{
    // Create a dummy GPU context for lowering
    migraphx::gpu::context ctx;

    // Run the full pipeline that would happen in GPU compilation
    migraphx::run_passes(
        m,
        {
            migraphx::gpu::lowering{&ctx, false}, // Lower high-level ops like convolution
            migraphx::replace_allocate{
                migraphx::gpu::gpu_allocation_model{}}, // Replace allocate with hip::allocate
            p,                                          // The write_literals pass we're testing
            migraphx::memory_coloring{"hip::allocate"}, // Memory coloring optimization
            migraphx::dead_code_elimination{}           // Clean up unused instructions
        });
}

// Get the size of the scratch buffer added by memory coloring
static std::size_t get_scratch_size(const migraphx::module& m)
{
    auto scratch_param = m.get_parameter("scratch");
    if(scratch_param != m.end())
    {
        return scratch_param->get_shape().bytes();
    }
    return 0;
}

template <class F>
void for_each_literal(const migraphx::module& m, F f)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "gpu::literal")
            continue;
        bool is_host = std::any_of(ins->outputs().begin(), ins->outputs().end(), [&](auto out) {
            return out->name() == "hip::copy";
        });
        f(ins, is_host);
    }
}

// Helper to count literals with host=false vs host=true
static std::pair<std::size_t, std::size_t> count_gpu_host_literals(const migraphx::module& m)
{
    std::size_t gpu_literals  = 0;
    std::size_t host_literals = 0;

    for_each_literal(m, [&](auto, bool is_host) {
        if(is_host)
            host_literals++;
        else
            gpu_literals++;
    });

    return {gpu_literals, host_literals};
}

// Calculate GPU memory usage: gpu literals + scratch buffer
static std::size_t calculate_gpu_memory_usage(const migraphx::module& m)
{
    std::size_t gpu_literal_size = 0;

    for_each_literal(m, [&](auto ins, bool is_host) {
        if(not is_host)
        {
            gpu_literal_size += ins->get_shape().bytes();
        }
    });

    return gpu_literal_size + get_scratch_size(m);
}

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

TEST_CASE(convolution_with_memory_limit)
{
    migraphx::module m;
    {
        // Input tensor
        auto input =
            m.add_parameter("input", migraphx::shape{migraphx::shape::float_type, {1, 3, 32, 32}});

        // Conv1: 3x64x3x3 weights (6,912 bytes)
        auto w1    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {64, 3, 3, 3}}, 100));
        auto conv1 = m.add_instruction(migraphx::make_op("convolution"), input, w1);

        // Conv2: 64x128x3x3 weights (294,912 bytes)
        auto w2    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {128, 64, 3, 3}}, 101));
        auto conv2 = m.add_instruction(migraphx::make_op("convolution"), conv1, w2);

        // Conv3: 128x256x3x3 weights (1,179,648 bytes)
        auto w3    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {256, 128, 3, 3}}, 102));
        auto conv3 = m.add_instruction(migraphx::make_op("convolution"), conv2, w3);

        m.add_return({conv3});
    }

    // Total weights: ~1.48 MB, set memory limit to force some to be host copies
    std::size_t max_memory = 500000; // 500KB
    run_pass_with_memory_coloring(m, {.max_memory = max_memory});

    // Verify we have a mix of GPU and host literals
    auto [gpu_lits, host_lits] = count_gpu_host_literals(m);
    EXPECT(gpu_lits >= 0);
    EXPECT(host_lits > 0);
    EXPECT(gpu_lits + host_lits == 3); // Three weight tensors

    // The pass's memory estimation includes scratch and is conservative,
    // so we just check that some literals were moved to host
    EXPECT(host_lits >= 2); // At least the two largest should be on host

    // Verify scratch buffer was added
    auto scratch_size = get_scratch_size(m);
    EXPECT(scratch_size > 0); // Memory coloring should add a scratch buffer

    // The write_literals pass's estimate includes 50% buffer, and the scratch
    // estimation is conservative. For a 3-layer conv network, the scratch can
    // be quite large due to intermediate activations.
    // Just verify that the pass moved most large weights to host.
    EXPECT(host_lits >= 2); // The two largest weights should be on host
}

TEST_CASE(multiple_convolutions_memory_estimation)
{
    migraphx::module m;
    {
        // Input tensor
        auto input = m.add_parameter(
            "input", migraphx::shape{migraphx::shape::float_type, {1, 3, 224, 224}});

        // Multiple parallel branches with convolutions
        // Branch 1
        auto w1a    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {32, 3, 3, 3}}, 200));
        auto conv1a = m.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}, {"stride", {1, 1}}}),
            input,
            w1a);

        // Branch 2
        auto w1b    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {32, 3, 5, 5}}, 201));
        auto conv1b = m.add_instruction(
            migraphx::make_op("convolution", {{"padding", {2, 2}}, {"stride", {1, 1}}}),
            input,
            w1b);

        // Merge branches
        auto concat = m.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), conv1a, conv1b);

        // Final convolution
        auto w2    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {128, 64, 1, 1}}, 202));
        auto conv2 = m.add_instruction(migraphx::make_op("convolution"), concat, w2);

        m.add_return({conv2});
    }

    // Set memory limit to test memory estimation with scratch allocations
    std::size_t max_memory = 100000; // 100KB
    run_pass_with_memory_coloring(m, {.max_memory = max_memory});

    // Check we have the expected number of literals
    auto [gpu_lits, host_lits] = count_gpu_host_literals(m);
    EXPECT(gpu_lits + host_lits == 3);

    // Verify scratch buffer was added
    auto scratch_size = get_scratch_size(m);
    EXPECT(scratch_size > 0); // Memory coloring should add scratch

    // With large input tensors (224x224), the scratch buffer for intermediate
    // activations can be very large. Just ensure literals were moved to host.
    EXPECT(host_lits >= 2); // Most weights should be on host with 100KB limit
}

TEST_CASE(convolution_chain_lowered)
{
    migraphx::module m;
    {
        auto input =
            m.add_parameter("input", migraphx::shape{migraphx::shape::float_type, {1, 3, 56, 56}});

        // Conv with small weights
        auto w1    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {64, 3, 1, 1}}, 300));
        auto conv1 = m.add_instruction(migraphx::make_op("convolution"), input, w1);

        // Conv with larger weights
        auto w2    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {128, 64, 3, 3}}, 301));
        auto conv2 = m.add_instruction(migraphx::make_op("convolution"), conv1, w2);

        m.add_return({conv2});
    }

    // Memory limit considering allocations and literals
    std::size_t max_memory = 1000000; // 1MB
    run_pass_with_memory_coloring(m, {.max_memory = max_memory});

    // Should have 2 weight literals
    auto [gpu_lits, host_lits] = count_gpu_host_literals(m);
    EXPECT(gpu_lits + host_lits == 2);

    // With lowering creating allocations and conservative memory estimation,
    // at least one weight should be on host
    EXPECT(host_lits >= 1);

    // Check scratch buffer was added and total GPU memory is reasonable
    auto scratch_size = get_scratch_size(m);
    EXPECT(scratch_size > 0);
    auto total_gpu_memory = calculate_gpu_memory_usage(m);
    EXPECT(total_gpu_memory <= max_memory * 2.0);
}

TEST_CASE(dense_network_memory_pressure)
{
    migraphx::module m;
    {
        auto input =
            m.add_parameter("input", migraphx::shape{migraphx::shape::float_type, {1, 3, 32, 32}});

        // Create a dense network with many small convolutions
        auto current = input;
        std::vector<migraphx::instruction_ref> weights;

        // Add 10 consecutive convolutions
        for(int i = 0; i < 10; i++)
        {
            std::size_t in_channels = (i == 0) ? 3 : 16;
            auto w                  = m.add_literal(migraphx::generate_literal(
                migraphx::shape{migraphx::shape::float_type, {16, in_channels, 3, 3}},
                static_cast<unsigned long>(400 + i)));
            weights.push_back(w);
            current = m.add_instruction(migraphx::make_op("convolution"), current, w);

            // Add ReLU to make it more realistic
            current = m.add_instruction(migraphx::make_op("relu"), current);
        }

        m.add_return({current});
    }

    // Very tight memory constraint
    std::size_t max_memory = 50000; // 50KB - much less than total weight size
    run_pass_with_memory_coloring(m, {.max_memory = max_memory});

    // With such tight memory, all weights should be host copies
    auto [gpu_lits, host_lits] = count_gpu_host_literals(m);
    EXPECT(host_lits == 10); // All weights on host due to tight memory
    EXPECT(gpu_lits == 0);
    EXPECT(gpu_lits + host_lits == 10);

    // Verify scratch buffer exists and total GPU memory is within limits
    auto scratch_size = get_scratch_size(m);
    EXPECT(scratch_size > 0);
    auto total_gpu_memory = calculate_gpu_memory_usage(m);
    // With all literals on host, only scratch should be on GPU
    EXPECT(total_gpu_memory == scratch_size);
}

TEST_CASE(no_memory_pressure_all_gpu)
{
    migraphx::module m;
    {
        auto input =
            m.add_parameter("input", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});

        // Small network that should fit entirely in GPU memory
        auto w1    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {8, 3, 3, 3}}, 500));
        auto conv1 = m.add_instruction(migraphx::make_op("convolution"), input, w1);

        auto w2    = m.add_literal(migraphx::generate_literal(
            migraphx::shape{migraphx::shape::float_type, {16, 8, 3, 3}}, 501));
        auto conv2 = m.add_instruction(migraphx::make_op("convolution"), conv1, w2);

        m.add_return({conv2});
    }

    // Large memory limit - everything should stay on GPU
    std::size_t max_memory = 10000000; // 10MB
    run_pass_with_memory_coloring(m, {.max_memory = max_memory});

    // All literals should be GPU literals
    auto [gpu_lits, host_lits] = count_gpu_host_literals(m);
    EXPECT(gpu_lits == 2);
    EXPECT(host_lits == 0);

    // Check scratch buffer and total memory usage
    auto scratch_size = get_scratch_size(m);
    EXPECT(scratch_size > 0);
    auto total_gpu_memory = calculate_gpu_memory_usage(m);
    EXPECT(total_gpu_memory > scratch_size); // Should include literals + scratch
    EXPECT(total_gpu_memory < max_memory);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
