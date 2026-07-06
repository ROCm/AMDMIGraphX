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

#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/context.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <vector>
#include "test.hpp"

// Exercises the parallel path in module::finalize, gated by the
// MIGRAPHX_FINALIZE_PARALLEL environment variable (set in main() below because
// the value is read once and cached). An op opts into being finalized on a
// worker thread by exposing the "parallel_finalize" attribute.
//
// The ops must be empty (no data members) to satisfy the operation
// type-erasure requirements, so instead of instance state they count their
// finalize() calls into the target's context, which module::finalize passes to
// each op. The counters live in shared_ptrs so the copies of the context made
// during finalize all update the same counts; they are atomic because the
// parallel pass finalizes on multiple threads.

// A minimal target whose context carries the finalize counters. module::finalize
// consumes a vector of these contexts.
struct final_target
{
    struct context
    {
        std::shared_ptr<std::atomic<int>> parallel_count = std::make_shared<std::atomic<int>>(0);
        std::shared_ptr<std::atomic<int>> serial_count   = std::make_shared<std::atomic<int>>(0);
        void finish() const {}
    };
    context ctx{};

    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return migraphx::pack();
    }

    std::string name() const { return "final"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {};
    }
    migraphx::context get_context() const { return ctx; }
};

// Opts into the parallel finalize pass; counts its finalize() calls in the ctx.
struct parallel_final_op
{
    std::string name() const { return "parallel_final_op"; }
    migraphx::value attributes() const { return {{"parallel_finalize", true}}; }

    migraphx::argument compute(const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }
    void finalize(final_target::context& ctx,
                  const migraphx::shape&,
                  const std::vector<migraphx::shape>&)
    {
        ++(*ctx.parallel_count);
    }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>&) const { return {0}; }
};

// Does NOT opt in; must be handled by the serial pass.
struct serial_final_op
{
    std::string name() const { return "serial_final_op"; }

    migraphx::argument compute(const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }
    void finalize(final_target::context& ctx,
                  const migraphx::shape&,
                  const std::vector<migraphx::shape>&)
    {
        ++(*ctx.serial_count);
    }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>&) const { return {0}; }
};

// Covers the parallel pass: every opted-in op is finalized across worker threads.
TEST_CASE(parallel_finalize_runs_all_opted_in_ops)
{
    final_target t{};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto lit = mm->add_literal(1);

    const int n = 64;
    for(int i = 0; i < n; ++i)
        mm->add_instruction(parallel_final_op{}, lit);

    p.finalize(t);

    EXPECT(t.ctx.parallel_count->load() == n);
    EXPECT(p.is_compiled());
}

// Covers the serial pass for non-opted-in ops running alongside opted-in ops.
TEST_CASE(parallel_finalize_serial_pass_handles_non_opted_in_ops)
{
    final_target t{};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto lit = mm->add_literal(1);

    mm->add_instruction(parallel_final_op{}, lit);
    mm->add_instruction(serial_final_op{}, lit);

    p.finalize(t);

    // The opted-in op is finalized by the parallel pass; the non-opted-in op is
    // finalized by the serial pass. Both must run exactly once.
    EXPECT(t.ctx.parallel_count->load() == 1);
    EXPECT(t.ctx.serial_count->load() == 1);
}

// Covers the serial pass recursing into submodules (ops with module_inputs are
// excluded from the parallel pass and finalized serially, including submodules).
TEST_CASE(parallel_finalize_recurses_into_submodules)
{
    final_target t{};
    migraphx::shape cond_s{migraphx::shape::bool_type};

    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto cond = mm->add_parameter("cond", cond_s);
    auto lit  = mm->add_literal(1);

    auto* then_mod = p.create_module("then");
    auto then_r    = then_mod->add_instruction(parallel_final_op{}, lit);
    then_mod->add_return({then_r});

    auto* else_mod = p.create_module("else");
    auto else_r    = else_mod->add_instruction(parallel_final_op{}, lit);
    else_mod->add_return({else_r});

    // The "if" op carries module_inputs, so it is excluded from the parallel pass
    // and finalized serially; the serial pass then finalizes its submodules.
    mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});

    p.finalize(t);

    // Both submodule ops are reached via the serial submodule recursion, even
    // though they declare "parallel_finalize".
    EXPECT(t.ctx.parallel_count->load() == 2);
    EXPECT(p.is_compiled());
}

// Covers correctness: forcing the parallel path on must not change evaluation.
TEST_CASE(parallel_finalize_output_matches_serial)
{
    final_target t{};
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto lit = mm->add_literal(migraphx::literal{7});
    mm->add_instruction(parallel_final_op{}, lit);

    p.finalize(t);

    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{7});
}

int main(int argc, const char* argv[])
{
    // module::finalize reads MIGRAPHX_FINALIZE_PARALLEL once and caches it, so it
    // must be set before the first finalize call in this process.
#ifdef _WIN32
    _putenv_s("MIGRAPHX_FINALIZE_PARALLEL", "4");
#else
    setenv("MIGRAPHX_FINALIZE_PARALLEL", "4", 1);
#endif
    test::run(argc, argv);
}
