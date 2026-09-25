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
#include <migraphx/gpu/time_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/reflect.hpp>
#include <test.hpp>
#include <chrono>
#include <memory>
#include <thread>

// Identity op that sleeps on the host, so a candidate's measured time is deterministic
struct sleep_op
{
    std::size_t usec = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.usec, "usec"));
    }

    std::string name() const { return "test::sleep"; }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs.front();
    }

    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        std::this_thread::sleep_for(std::chrono::microseconds{usec});
        return args.front();
    }
};

// Benchmark candidate that runs for `usec` microseconds and is identified by its solution id
struct test_candidate
{
    std::size_t usec = 0;
    int id           = 0;
    bool fail        = false;
    // Counts make_program calls to observe which candidates each benchmark pass builds
    std::shared_ptr<std::size_t> programs_built = std::make_shared<std::size_t>(0);

    std::vector<migraphx::argument> generate_arguments(const migraphx::gpu::context&) const
    {
        return {migraphx::fill_argument({migraphx::shape::float_type, {4}}, 1)};
    }

    migraphx::program make_program() const
    {
        if(fail)
            MIGRAPHX_THROW("test candidate failure");
        ++(*programs_built);
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {4}});
        mm->add_return({mm->add_instruction(sleep_op{usec}, x)});
        return p;
    }

    migraphx::tracer trace() const { return {}; }

    migraphx::value solution() const { return id; }

    void before_run(const migraphx::program&) const {}
};

static migraphx::gpu::adaptive_topk_benchmark small_adaptive_benchmark(std::size_t top_k)
{
    migraphx::gpu::adaptive_topk_benchmark bench;
    bench.top_k              = top_k;
    bench.precise_ms         = 5;
    bench.coarse_ms          = 2;
    bench.precise_min_bundle = 1;
    bench.max_runs           = 4;
    return bench;
}

TEST_CASE(simple_benchmark_picks_fastest)
{
    migraphx::gpu::context ctx{};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {
        test_candidate{2000, 0}, test_candidate{100, 1}, test_candidate{1000, 2}};
    migraphx::gpu::simple_benchmark bench{/* bundle */ 1, /* nruns */ 4};
    const auto& winner = bench.run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 1);
}

TEST_CASE(simple_benchmark_no_candidates_throws)
{
    migraphx::gpu::context ctx{};
    std::vector<migraphx::gpu::benchmark_candidate> candidates;
    EXPECT(test::throws([&] { migraphx::gpu::simple_benchmark{}.run(ctx, candidates); }));
}

TEST_CASE(adaptive_benchmark_picks_fastest)
{
    migraphx::gpu::context ctx{};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {
        test_candidate{2000, 0}, test_candidate{100, 1}, test_candidate{1000, 2}};
    const auto& winner = small_adaptive_benchmark(2).run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 1);
}

TEST_CASE(adaptive_benchmark_no_candidates_throws)
{
    migraphx::gpu::context ctx{};
    std::vector<migraphx::gpu::benchmark_candidate> candidates;
    EXPECT(test::throws([&] { small_adaptive_benchmark(2).run(ctx, candidates); }));
}

TEST_CASE(adaptive_benchmark_skips_failed_candidates)
{
    migraphx::gpu::context ctx{};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {
        test_candidate{0, 0, true}, test_candidate{100, 1}, test_candidate{1000, 2}};
    const auto& winner = small_adaptive_benchmark(2).run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 1);
}

TEST_CASE(adaptive_benchmark_all_failed_throws)
{
    migraphx::gpu::context ctx{};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {test_candidate{0, 0, true},
                                                                  test_candidate{0, 1, true}};
    EXPECT(test::throws([&] { small_adaptive_benchmark(2).run(ctx, candidates); }));
}

TEST_CASE(adaptive_benchmark_only_times_top_k_precisely)
{
    migraphx::gpu::context ctx{};
    test_candidate fast{100, 1};
    test_candidate mid{1000, 2};
    test_candidate slow{2000, 3};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {slow, fast, mid};
    const auto& winner = small_adaptive_benchmark(1).run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 1);
    // The coarse pass builds every candidate once; only the fastest is rebuilt for the precise pass
    EXPECT(*fast.programs_built == 2);
    EXPECT(*mid.programs_built == 1);
    EXPECT(*slow.programs_built == 1);
}

TEST_CASE(adaptive_benchmark_top_k_zero_times_all_precisely)
{
    migraphx::gpu::context ctx{};
    test_candidate fast{100, 1};
    test_candidate slow{1000, 2};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {slow, fast};
    const auto& winner = small_adaptive_benchmark(0).run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 1);
    EXPECT(*fast.programs_built == 2);
    EXPECT(*slow.programs_built == 2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
