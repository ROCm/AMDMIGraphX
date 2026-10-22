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
#include <migraphx/gpu/write_literals.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/memory_coloring.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/schedule.hpp>
#include <migraphx/gpu/schedule_model.hpp>
#include <migraphx/value.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/hip.hpp>
#include <algorithm>
#include <numeric>
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

// A kernel stand-in that, like gather, reads its first input only at index-selected positions.
struct indexed_op
{
    std::string name() const { return "test::indexed_op"; }
    migraphx::value attributes() const { return {{"indexed_inputs", {0}}}; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return {inputs.front().type(), inputs.back().lens()};
    }
};

// A kernel stand-in that reads every input in full
struct dense_op
{
    std::string name() const { return "test::dense_op"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs.front();
    }
};

// Reads the literal with its own kernel, so its staged copy is live apart from the others
static migraphx::instruction_ref add_read(migraphx::module& m, migraphx::instruction_ref lit)
{
    return m.add_instruction(dense_op{}, lit);
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
    // The literal and the allocation it is copied into total 100 bytes with the scratch pad, so
    // this limit forces the literal off the GPU.
    run_pass(m1, {.max_memory = 50});

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

// A kernel stand-in with the given code object, writing its output in place of its last input
static migraphx::operation make_code_object(const std::string& image, const migraphx::shape& s)
{
    return migraphx::make_op(
        "gpu::code_object",
        {{"code_object", migraphx::value::binary{image}},
         {"symbol_name", "kernel"},
         {"global", 1},
         {"local", 1},
         {"expected_inputs", migraphx::to_value(std::vector<migraphx::shape>{s})},
         {"output", migraphx::to_value(s)}});
}

TEST_CASE(literal_is_not_scratch)
{
    migraphx::shape s{migraphx::shape::float_type, {10000}}; // 40000 bytes

    migraphx::module m1;
    {
        m1.add_return({add_literal(m1, s, 7)});
    }
    // Only allocations occupy scratch, so a literal exactly filling the limit stays on the GPU.
    run_pass(m1, {.max_memory = 40000});

    migraphx::module m2;
    {
        m2.add_return({add_gpu_literal(m2, s, 7)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(staged_copies_sort_next_to_consumer)
{
    migraphx::shape s{migraphx::shape::float_type, {10000}};

    migraphx::module m;
    auto x  = m.add_parameter("x", s);
    auto l1 = add_literal(m, s, 1);
    auto l2 = add_literal(m, s, 2);
    auto a  = m.add_instruction(make_code_object("kernel-a", s), l1);
    auto b  = m.add_instruction(make_code_object("kernel-b", s), l2);
    m.add_return({x, a, b});
    // Stage every literal, then schedule with one stream, which sorts the module by weight
    migraphx::run_passes(m,
                         {migraphx::gpu::write_literals{.max_memory = 1},
                          migraphx::schedule{migraphx::gpu::schedule_model{1}, true},
                          migraphx::dead_code_elimination{}});

    // Each staged copy is placed just before its consumer, not hoisted with the other copies
    for(auto ins : migraphx::iterator_for(m))
    {
        if(ins->name() != "hip::copy")
            continue;
        EXPECT(ins->outputs().size() == 1);
        EXPECT(std::next(ins) == ins->outputs().front());
    }
    EXPECT(std::count_if(
               m.begin(), m.end(), [](const auto& ins) { return ins.name() == "hip::copy"; }) == 2);
}

TEST_CASE(memory_limit_stages_smallest_first)
{
    migraphx::shape medium{migraphx::shape::float_type, {1000}}; // 4000 bytes
    migraphx::shape big{migraphx::shape::float_type, {10000}};   // 40000 bytes

    migraphx::module m1;
    {
        m1.add_return({add_read(m1, add_literal(m1, medium, 1)),
                       add_read(m1, add_literal(m1, medium, 2)),
                       add_read(m1, add_literal(m1, medium, 3)),
                       add_read(m1, add_literal(m1, medium, 4)),
                       add_read(m1, add_literal(m1, big, 5))});
    }
    // The literals total 56000 bytes with no scratch, so this limit leaves 3000 bytes to free.
    // Each literal is read on its own, so staging nets every staged literal but the largest: two
    // medium ones free 4000 bytes at a 4000-byte peak while staging the big one alone would free
    // nothing.
    run_pass(m1, {.max_memory = 53000});

    migraphx::module m2;
    {
        m2.add_return({add_read(m2, add_host_copy(m2, medium, 1)),
                       add_read(m2, add_host_copy(m2, medium, 2)),
                       add_read(m2, add_gpu_literal(m2, medium, 3)),
                       add_read(m2, add_gpu_literal(m2, medium, 4)),
                       add_read(m2, add_gpu_literal(m2, big, 5))});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(memory_limit_drops_unneeded_small_literal)
{
    migraphx::shape small{migraphx::shape::float_type, {2}};   // 8 bytes
    migraphx::shape big{migraphx::shape::float_type, {10000}}; // 40000 bytes

    migraphx::module m1;
    {
        m1.add_return({add_read(m1, add_literal(m1, small, 1)),
                       add_read(m1, add_literal(m1, big, 2)),
                       add_read(m1, add_literal(m1, big, 3))});
    }
    // The literals total 80008 bytes with no scratch, so this limit leaves 20008 bytes to free.
    // Staging smallest-first reaches that only once both big literals are staged, and the two of
    // them cover it alone, so the small one is left on the GPU.
    run_pass(m1, {.max_memory = 60000});

    migraphx::module m2;
    {
        m2.add_return({add_read(m2, add_gpu_literal(m2, small, 1)),
                       add_read(m2, add_host_copy(m2, big, 2)),
                       add_read(m2, add_host_copy(m2, big, 3))});
    }
    EXPECT(m1 == m2);
}

// Twelve 4000-byte literals then two 40000-byte ones (128000 bytes), each added by add(m, shape,
// seed) and read on its own
template <class Add>
static void add_staging_literals(migraphx::module& m, Add add)
{
    migraphx::shape medium{migraphx::shape::float_type, {1000}};
    migraphx::shape big{migraphx::shape::float_type, {10000}};
    std::vector<migraphx::instruction_ref> outs;
    std::generate_n(std::back_inserter(outs), 12, [&, seed = 0ul]() mutable {
        return add_read(m, add(m, medium, ++seed));
    });
    outs.push_back(add_read(m, add(m, big, 13)));
    outs.push_back(add_read(m, add(m, big, 14)));
    m.add_return(outs);
}

TEST_CASE(staging_peak_stages_largest_first)
{
    migraphx::module m1;
    add_staging_literals(m1, &add_literal);
    // The literals total 128000 bytes with no scratch, so this limit leaves 44000 bytes to free.
    // With a peak budget equal to the shortfall the big literals are taken first and one medium
    // then covers the rest, so three copies do.
    run_pass(m1, {.max_memory = 84000, .staging_peak_percent = 100});

    migraphx::module m2;
    add_staging_literals(m2, [](auto& m, const auto& s, auto seed) {
        return (seed == 1 or seed > 12) ? add_host_copy(m, s, seed) : add_gpu_literal(m, s, seed);
    });
    EXPECT(m1 == m2);
}

TEST_CASE(staging_peak_zero_stages_smallest_first)
{
    migraphx::module m1;
    add_staging_literals(m1, &add_literal);
    // The same 44000-byte shortfall without a peak budget takes all twelve medium copies instead.
    run_pass(m1, {.max_memory = 84000, .staging_peak_percent = 0});

    migraphx::module m2;
    add_staging_literals(m2, [](auto& m, const auto& s, auto seed) {
        return seed <= 12 ? add_host_copy(m, s, seed) : add_gpu_literal(m, s, seed);
    });
    EXPECT(m1 == m2);
}

TEST_CASE(staging_peak_exceeded_when_needed)
{
    migraphx::shape small{migraphx::shape::float_type, {2}};   // 8 bytes
    migraphx::shape big{migraphx::shape::float_type, {10000}}; // 40000 bytes

    migraphx::module m1;
    {
        m1.add_return({add_read(m1, add_literal(m1, small, 1)),
                       add_read(m1, add_literal(m1, small, 2)),
                       add_read(m1, add_literal(m1, big, 3)),
                       add_read(m1, add_literal(m1, big, 4))});
    }
    // The literals total 80016 bytes with no scratch, so this limit leaves 20016 bytes to free.
    // Only the small literals fit the 10008-byte peak budget and they cannot cover it, so the big
    // ones are staged as well and the small ones are dropped as unneeded.
    run_pass(m1, {.max_memory = 60000, .staging_peak_percent = 50});

    migraphx::module m2;
    {
        m2.add_return({add_read(m2, add_gpu_literal(m2, small, 1)),
                       add_read(m2, add_gpu_literal(m2, small, 2)),
                       add_read(m2, add_host_copy(m2, big, 3)),
                       add_read(m2, add_host_copy(m2, big, 4))});
    }
    EXPECT(m1 == m2);
}

// Three 40000-byte literals read together by one kernel, two read on their own and one read only
// at indexed positions, each added by add(m, shape, seed)
template <class Add>
static void add_stacked_literals(migraphx::module& m, Add add)
{
    migraphx::shape s{migraphx::shape::float_type, {10000}};
    migraphx::shape idx{migraphx::shape::int32_type, {1}};
    auto l1       = add(m, s, 1);
    auto l2       = add(m, s, 2);
    auto l3       = add(m, s, 3);
    auto together = m.add_instruction(dense_op{}, l1, l2, l3);
    auto alone1   = add_read(m, add(m, s, 4));
    auto alone2   = add_read(m, add(m, s, 5));
    auto data     = add(m, s, 6);
    auto i        = m.insert_parameter(m.end(), "i", idx);
    auto indexed  = m.add_instruction(indexed_op{}, data, i);
    m.add_return({together, alone1, alone2, indexed});
}

TEST_CASE(literals_read_together_stack_in_scratch)
{
    migraphx::module m1;
    add_stacked_literals(m1, &add_literal);
    // The literals and parameter total 240004 bytes with no scratch, so this limit leaves 90004
    // bytes to free, 40000 of which the indexed literal frees zero-copy. Copies of the three
    // literals read together are all live at once, so staging them frees nothing; one of them with
    // the two read on their own covers the rest at a 40000-byte peak.
    run_pass(m1, {.max_memory = 150000});

    migraphx::module m2;
    add_stacked_literals(m2, [](auto& m, const auto& s, auto seed) {
        if(seed >= 3 and seed <= 5)
            return add_host_copy(m, s, seed);
        return add_gpu_literal(m, s, seed, seed == 6);
    });
    EXPECT(m1 == m2);
}

// GPU memory the module needs once coloring has sized the scratch buffer: the literals left
// resident and the parameters, scratch included
static std::size_t gpu_memory(const migraphx::module& m)
{
    return migraphx::transform_accumulate(
        m.begin(), m.end(), std::size_t{0}, std::plus<>{}, [](const migraphx::instruction& ins) {
            if(ins.name() == "@param")
                return ins.get_shape().bytes();
            if(ins.name() != "gpu::literal" or ins.get_operator().to_value().at("host").to<bool>())
                return std::size_t{0};
            return ins.get_shape().bytes();
        });
}

TEST_CASE(memory_fits_after_coloring_literals_read_together)
{
    migraphx::module m;
    add_stacked_literals(m, &add_literal);
    // The copies of the three literals read together cannot share scratch, so crediting them as if
    // only the largest were live would leave the colored module over the limit
    migraphx::run_passes(m,
                         {migraphx::gpu::write_literals{.max_memory = 150000},
                          migraphx::schedule{migraphx::gpu::schedule_model{1}, true},
                          migraphx::memory_coloring{"hip::allocate"},
                          migraphx::dead_code_elimination{}});
    EXPECT(gpu_memory(m) <= 150000);
}

// A 64000-byte allocation read by one kernel, then two 40000-byte literals each read on its own,
// the literals added by add(m, shape, seed)
template <class Add>
static void add_literals_under_scratch(migraphx::module& m, Add add)
{
    migraphx::shape scratch{migraphx::shape::float_type, {16000}};
    migraphx::shape s{migraphx::shape::float_type, {10000}};
    auto alloc = m.add_instruction(
        migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(scratch)}}));
    auto k = m.add_instruction(dense_op{}, alloc);
    m.add_return({k, add_read(m, add(m, s, 1)), add_read(m, add(m, s, 2))});
}

TEST_CASE(copy_under_scratch_peak_is_free)
{
    migraphx::module m1;
    add_literals_under_scratch(m1, &add_literal);
    // The literals total 80000 bytes and the allocation's padded estimate 96000, so this limit
    // leaves 26000 bytes to free. Neither literal is read while the allocation is live, so one copy
    // fits under the scratch peak at no cost and covers the shortfall alone.
    run_pass(m1, {.max_memory = 150000});

    migraphx::module m2;
    add_literals_under_scratch(m2, [](auto& m, const auto& s, auto seed) {
        return seed == 1 ? add_host_copy(m, s, seed) : add_gpu_literal(m, s, seed);
    });
    EXPECT(m1 == m2);
}

TEST_CASE(indexed_literal_zero_copy)
{
    migraphx::shape data{migraphx::shape::float_type, {10000}};
    migraphx::shape idx{migraphx::shape::int32_type, {1}};

    migraphx::module m1;
    {
        auto lit = add_literal(m1, data, 1);
        auto i   = m1.add_parameter("i", idx);
        m1.add_return({m1.add_instruction(indexed_op{}, lit, i)});
    }
    // A 1-byte limit forces the literal off the GPU. It only feeds an indexed input, so the kernel
    // reads it from host memory and no staging copy is made.
    run_pass(m1, {.max_memory = 1});

    migraphx::module m2;
    {
        auto lit = add_gpu_literal(m2, data, 1, true);
        auto i   = m2.insert_parameter(m2.end(), "i", idx);
        m2.add_return({m2.add_instruction(indexed_op{}, lit, i)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(indexed_literal_stays_when_memory_fits)
{
    migraphx::shape data{migraphx::shape::float_type, {10000}};
    migraphx::shape idx{migraphx::shape::int32_type, {1}};

    migraphx::module m1;
    {
        auto lit = add_literal(m1, data, 1);
        auto i   = m1.add_parameter("i", idx);
        m1.add_return({m1.add_instruction(indexed_op{}, lit, i)});
    }
    run_pass(m1, {.max_memory = 1 << 30});

    migraphx::module m2;
    {
        auto lit = add_gpu_literal(m2, data, 1);
        auto i   = m2.insert_parameter(m2.end(), "i", idx);
        m2.add_return({m2.add_instruction(indexed_op{}, lit, i)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(indexed_literal_freed_before_staging)
{
    migraphx::shape data{migraphx::shape::float_type, {10000}}; // 40000 bytes
    migraphx::shape idx{migraphx::shape::int32_type, {1}};

    migraphx::module m1;
    {
        auto lit = add_literal(m1, data, 1);
        auto i   = m1.add_parameter("i", idx);
        m1.add_return({m1.add_instruction(indexed_op{}, lit, i), add_literal(m1, data, 2)});
    }
    // The literals and parameter total 80004 bytes with no scratch, so this limit leaves 20004
    // bytes to free. The indexed literal is freed in full at no scratch cost, which covers it, so
    // the dense literal stays on the GPU instead of being staged.
    run_pass(m1, {.max_memory = 60000});

    migraphx::module m2;
    {
        auto lit = add_gpu_literal(m2, data, 1, true);
        auto i   = m2.insert_parameter(m2.end(), "i", idx);
        m2.add_return({m2.add_instruction(indexed_op{}, lit, i), add_gpu_literal(m2, data, 2)});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(mixed_use_literal_is_staged)
{
    migraphx::shape data{migraphx::shape::float_type, {10000}};
    migraphx::shape idx{migraphx::shape::int32_type, {1}};

    migraphx::module m1;
    {
        auto lit = add_literal(m1, data, 1);
        auto i   = m1.add_parameter("i", idx);
        m1.add_return({m1.add_instruction(indexed_op{}, lit, i), lit});
    }
    // The return also uses the literal as a plain input, so it cannot stay on the host and is
    // staged.
    run_pass(m1, {.max_memory = 1});

    migraphx::module m2;
    {
        auto copy = add_host_copy(m2, data, 1);
        auto i    = m2.insert_parameter(m2.end(), "i", idx);
        m2.add_return({m2.add_instruction(indexed_op{}, copy, i), copy});
    }
    EXPECT(m1 == m2);
}

// Runs a gather-style kernel whose data literal is forced off the GPU by a 1-byte limit, so the
// kernel reads it zero-copy from host memory.
struct zero_copy_gather_test
{
    std::string op;
    migraphx::value attrs;
    migraphx::literal data;
    migraphx::shape idx_s;
    migraphx::shape out_s;

    migraphx::argument run(std::vector<int32_t> indices) const
    {
        migraphx::gpu::context ctx;
        auto kernel = migraphx::gpu::compile_op(op, ctx, {data.get_shape(), idx_s, out_s}, attrs);
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto lit = mm->add_literal(data);
        auto idx = mm->add_parameter("indices", idx_s);
        auto out = mm->add_parameter("output", out_s);
        mm->add_return({mm->add_instruction(kernel, lit, idx, out)});
        run_pass(*mm, {.max_memory = 1});
        EXPECT(migraphx::none_of(migraphx::iterator_for(*mm),
                                 [](auto ins) { return ins->name() == "hip::copy"; }));
        p.compile(migraphx::make_target("gpu"), migraphx::compile_options{});
        return migraphx::gpu::from_gpu(
            p.eval({{"indices", migraphx::gpu::to_gpu(migraphx::argument(idx_s, indices.data()))},
                    {"output", migraphx::gpu::allocate_gpu(out_s)}})
                .front());
    }

    // The reference implementation of the op gives the expected result
    migraphx::argument expected(std::vector<int32_t> indices) const
    {
        return migraphx::make_op(op, attrs).compute(
            out_s, {data.get_argument(), migraphx::argument(idx_s, indices.data())});
    }
};

TEST_CASE(zero_copy_gather_reads_host_literal)
{
    migraphx::shape data_s{migraphx::shape::float_type, {4, 3}};
    zero_copy_gather_test t{"gather",
                            {{"axis", 0}},
                            migraphx::generate_literal(data_s, 1),
                            {migraphx::shape::int32_type, {2}},
                            {migraphx::shape::float_type, {2, 3}}};
    std::vector<int32_t> indices = {2, 0};
    EXPECT(t.run(indices) == t.expected(indices));
}

TEST_CASE(zero_copy_gathernd_reads_host_literal)
{
    migraphx::shape data_s{migraphx::shape::half_type, {4096, 64}};
    zero_copy_gather_test t{"gathernd",
                            {{"batch_dims", 0}},
                            migraphx::generate_literal(data_s, 1),
                            {migraphx::shape::int32_type, {1, 1, 1}},
                            {migraphx::shape::half_type, {1, 1, 64}}};
    std::vector<int32_t> indices = {4000};
    EXPECT(t.run(indices) == t.expected(indices));
}

// Build a module of bare hip::allocate buffers with controlled live ranges to exercise memory
// coloring. Each entry is {start, end, nelems}: the allocation is live over [start, end). To make
// the conflicts survive the topological sort that write_literals performs, every allocation is
// anchored to a scalar "spine" chain (via reduce_sum + add) at both its start and end time. The
// spine forces a linear order, so each buffer is genuinely live across [start, end) regardless of
// how the module is sorted. An optional literal is appended so write_literals has data to place.
static migraphx::module make_interval_module(const std::vector<std::array<std::size_t, 3>>& ivals,
                                             std::size_t lit_floats)
{
    std::size_t steps = std::accumulate(
        ivals.begin(), ivals.end(), std::size_t{0}, [](std::size_t acc, const auto& iv) {
            return std::max(acc, iv[1]);
        });

    migraphx::module m;
    auto spine = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1}});
    std::vector<migraphx::instruction_ref> allocs(ivals.size());
    auto anchor = [&](migraphx::instruction_ref a) {
        auto r = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), a);
        spine  = m.add_instruction(migraphx::make_op("add"), spine, r);
    };
    for(std::size_t t = 0; t <= steps; t++)
    {
        for(std::size_t i : migraphx::range(ivals.size()))
            if(ivals[i][1] == t)
                anchor(allocs[i]); // end anchor: last use of the buffer
        for(std::size_t i : migraphx::range(ivals.size()))
            if(ivals[i][0] == t)
            {
                allocs[i] = m.add_instruction(
                    migraphx::make_op("hip::allocate",
                                      {{"shape",
                                        migraphx::to_value(migraphx::shape{
                                            migraphx::shape::float_type, {ivals[i][2]}})}}));
                anchor(allocs[i]); // start anchor: first use of the buffer
            }
    }
    std::vector<migraphx::instruction_ref> outs{spine};
    outs.push_back(m.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {lit_floats}}, 1)));
    m.add_return(outs);
    return m;
}

struct coloring_result
{
    std::size_t scratch     = 0; // bytes of the scratch buffer memory coloring produced
    std::size_t gpu_literal = 0; // bytes of literals left in GPU memory
    bool literal_on_host    = false;
};

static coloring_result
run_write_literals_and_color(const std::vector<std::array<std::size_t, 3>>& ivals,
                             std::size_t lit_floats,
                             std::size_t max_memory,
                             std::size_t overhead)
{
    auto m = make_interval_module(ivals, lit_floats);
    migraphx::run_passes(m,
                         {migraphx::gpu::write_literals{.max_memory               = max_memory,
                                                        .scratch_overhead_percent = overhead},
                          migraphx::memory_coloring{"hip::allocate"},
                          migraphx::dead_code_elimination{}});
    coloring_result r;
    auto scratch = m.get_parameter("scratch");
    if(scratch != m.end())
        r.scratch = scratch->get_shape().bytes();
    for(auto ins : migraphx::iterator_for(m))
    {
        if(ins->name() != "gpu::literal")
            continue;
        if(ins->get_operator().to_value()["host"].to<bool>())
            r.literal_on_host = true;
        else
            r.gpu_literal += ins->get_shape().bytes();
    }
    return r;
}

// write_literals estimates the scratch memory from a liveness peak, then inflates it by
// scratch_overhead_percent because memory coloring is NP-hard and may use more than that lower
// bound. This test demonstrates why the padding is needed: the five buffers below have an
// interference graph that is the path b0-b1-b2-b3-b4, so no three are ever live at once (peak = 2
// buffers = 4096 bytes), but the greedy coloring packs them into 3 buffers = 6144 bytes.
//
// With scratch_overhead_percent = 0 the pass trusts the 4096-byte estimate, decides a 512-byte
// literal fits on the GPU, and the compiled program then needs 6144 + 512 > max_memory. The default
// 50% pad raises the estimate to 6144 bytes, so the pass moves the literal to host and the program
// stays within max_memory.
TEST_CASE(scratch_overhead_accounts_for_coloring)
{
    std::size_t sz = 512; // floats per buffer (2048 bytes); peak = 2, coloring uses 3
    std::vector<std::array<std::size_t, 3>> ivals = {
        {0, 2, sz}, {1, 3, sz}, {2, 4, sz}, {3, 5, sz}, {4, 5, sz}};
    std::size_t lit_floats = 128;  // 512-byte literal
    std::size_t max_memory = 6400; // between the 4096-byte peak and the 6144-byte colored size

    auto no_pad = run_write_literals_and_color(ivals, lit_floats, max_memory, 0);
    // The unpadded estimate underprovisions: the literal stays on the GPU and the program needs
    // more than max_memory once coloring actually runs.
    EXPECT(not no_pad.literal_on_host);
    EXPECT(no_pad.scratch + no_pad.gpu_literal > max_memory);

    auto padded = run_write_literals_and_color(ivals, lit_floats, max_memory, 50);
    // The 50% pad accounts for the coloring overhead, so the pass moves the literal to host and the
    // program fits within max_memory.
    EXPECT(padded.literal_on_host);
    EXPECT(padded.scratch + padded.gpu_literal <= max_memory);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
