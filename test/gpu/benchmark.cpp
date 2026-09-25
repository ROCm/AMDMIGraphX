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
#include <migraphx/gpu/hip.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/reflect.hpp>
#include <test.hpp>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Identity op that sleeps on the host, so a candidate's measured time is deterministic
struct sleep_op
{
    std::size_t usec = 0;
    // The first slow_launches launches sleep for slow_usec instead, like a transiently slow GPU
    std::size_t slow_launches = 0;
    std::size_t slow_usec     = 0;
    std::shared_ptr<std::size_t> launches{};
    // Input of each launch, kept alive so separately generated inputs never share an address
    std::shared_ptr<std::vector<migraphx::argument>> launch_inputs{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.usec, "usec"),
                              f(self.slow_launches, "slow_launches"),
                              f(self.slow_usec, "slow_usec"));
    }

    std::string name() const { return "test::sleep"; }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs.front();
    }

    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(launches)
            ++(*launches);
        if(launch_inputs)
            launch_inputs->push_back(args.front());
        const bool slow = launches != nullptr and *launches <= slow_launches;
        std::this_thread::sleep_for(std::chrono::microseconds{slow ? slow_usec : usec});
        return args.front();
    }
};

// Benchmark candidate that runs for `usec` microseconds and is identified by its solution id
struct test_candidate
{
    std::size_t usec = 0;
    int id           = 0;
    bool fail        = false;
    // Makes the first launches of the candidate's programs slower, see sleep_op
    std::size_t slow_launches                     = 0;
    std::size_t slow_usec                         = 0;
    std::unordered_map<std::string, double> fills = {};
    // Size of an unused int8 parameter, like a solution's scratch; zero adds none
    std::size_t scratch = 0;
    // Counts make_program calls to observe which candidates each benchmark pass builds
    std::shared_ptr<std::size_t> programs_built = std::make_shared<std::size_t>(0);
    std::shared_ptr<std::size_t> launches       = std::make_shared<std::size_t>(0);
    std::shared_ptr<std::vector<migraphx::argument>> launch_inputs =
        std::make_shared<std::vector<migraphx::argument>>();
    // Key and shape of each argument the benchmark had this candidate generate
    std::shared_ptr<std::vector<std::pair<std::string, migraphx::shape>>> generated =
        std::make_shared<std::vector<std::pair<std::string, migraphx::shape>>>();
    // Trace output, to observe which candidates the precise pass times
    std::shared_ptr<std::stringstream> log = std::make_shared<std::stringstream>();

    std::vector<std::pair<std::string, migraphx::shape>>
    generate_argument_keys(const migraphx::program& p) const
    {
        return migraphx::gpu::fill_map_argument_keys(p, fills);
    }

    migraphx::argument generate_argument(const migraphx::gpu::context& ctx,
                                         const std::string& key,
                                         const migraphx::shape& s) const
    {
        generated->emplace_back(key, s);
        return migraphx::gpu::generate_fill_map_argument(ctx, fills, key, s);
    }

    migraphx::program make_program() const
    {
        if(fail)
            MIGRAPHX_THROW("test candidate failure");
        ++(*programs_built);
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {4}});
        if(scratch > 0)
            mm->add_parameter("scratch", {migraphx::shape::int8_type, {scratch}});
        mm->add_return({mm->add_instruction(
            sleep_op{usec, slow_launches, slow_usec, launches, launch_inputs}, x)});
        return p;
    }

    migraphx::tracer trace() const { return {*log}; }

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
    EXPECT(fast.log->str().find("Precise solution") != std::string::npos);
    EXPECT(mid.log->str().find("Precise solution") == std::string::npos);
    EXPECT(slow.log->str().find("Precise solution") == std::string::npos);
    // The precise pass reuses the program fast was coarse timed with
    EXPECT(*fast.programs_built == 1);
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
    EXPECT(*fast.programs_built == 1);
    EXPECT(*slow.programs_built == 1);
}

TEST_CASE(adaptive_benchmark_measures_a_candidate_whose_estimate_misses_the_top_k)
{
    migraphx::gpu::context ctx{};
    test_candidate steady{300, 1};
    test_candidate transient{100, 2};
    // The warmup and the estimate are slower than steady, the later runs are faster
    transient.slow_launches                                    = 2;
    transient.slow_usec                                        = 500;
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {steady, transient};
    auto bench                                                 = small_adaptive_benchmark(1);
    bench.coarse_max_runs                                      = 4;
    const auto& winner                                         = bench.run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 2);
}

TEST_CASE(adaptive_benchmark_zero_coarse_max_runs_throws)
{
    migraphx::gpu::context ctx{};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {test_candidate{100, 1}};
    auto bench                                                 = small_adaptive_benchmark(1);
    bench.coarse_max_runs                                      = 0;
    EXPECT(test::throws([&] { bench.run(ctx, candidates); }));
}

TEST_CASE(adaptive_benchmark_does_not_precisely_time_candidates_far_behind_the_best)
{
    migraphx::gpu::context ctx{};
    test_candidate fast{100, 1};
    test_candidate slow{2000, 2};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {fast, slow};
    const auto& winner = small_adaptive_benchmark(2).run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 1);
    EXPECT(fast.log->str().find("Precise solution") != std::string::npos);
    EXPECT(slow.log->str().find("Precise solution") == std::string::npos);
}

TEST_CASE(adaptive_benchmark_precisely_times_candidates_close_to_the_best)
{
    migraphx::gpu::context ctx{};
    test_candidate fast{100, 1};
    test_candidate near_best{150, 2};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {fast, near_best};
    (void)small_adaptive_benchmark(2).run(ctx, candidates);
    EXPECT(fast.log->str().find("Precise solution") != std::string::npos);
    EXPECT(near_best.log->str().find("Precise solution") != std::string::npos);
}

TEST_CASE(adaptive_benchmark_precise_pass_reuses_the_coarse_programs_of_the_top_k)
{
    migraphx::gpu::context ctx{};
    test_candidate fast{100, 1};
    test_candidate near_best{150, 2};
    test_candidate slow{1000, 3};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {slow, fast, near_best};
    (void)small_adaptive_benchmark(2).run(ctx, candidates);
    EXPECT(fast.log->str().find("Precise solution") != std::string::npos);
    EXPECT(near_best.log->str().find("Precise solution") != std::string::npos);
    // slow is pushed out of the top_k by near_best
    EXPECT(slow.log->str().find("Precise solution") == std::string::npos);
    EXPECT(*fast.programs_built == 1);
    EXPECT(*near_best.programs_built == 1);
    EXPECT(*slow.programs_built == 1);
}

TEST_CASE(simple_benchmark_shares_inputs_between_candidates_with_the_same_fill)
{
    migraphx::gpu::context ctx{};
    migraphx::shape s{migraphx::shape::float_type, {4}};
    test_candidate random{100, 1};
    test_candidate random_again{100, 2};
    test_candidate filled{100, 3};
    filled.fills = {{s.type_string() + migraphx::shape::to_sizes_string({s}), 3}};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {random, random_again, filled};
    (void)migraphx::gpu::simple_benchmark{/* bundle */ 1, /* nruns */ 2}.run(ctx, candidates);
    EXPECT(random.launch_inputs->front().data() == random_again.launch_inputs->front().data());
    EXPECT(random.launch_inputs->front().data() != filled.launch_inputs->front().data());
}

TEST_CASE(adaptive_benchmark_shares_inputs_between_candidates_with_the_same_fill)
{
    migraphx::gpu::context ctx{};
    migraphx::shape s{migraphx::shape::float_type, {4}};
    test_candidate random{100, 1};
    test_candidate random_again{100, 2};
    test_candidate filled{100, 3};
    filled.fills = {{s.type_string() + migraphx::shape::to_sizes_string({s}), 3}};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {random, random_again, filled};
    (void)small_adaptive_benchmark(3).run(ctx, candidates);
    EXPECT(random.launch_inputs->front().data() == random_again.launch_inputs->front().data());
    EXPECT(random.launch_inputs->front().data() != filled.launch_inputs->front().data());
}

TEST_CASE(adaptive_benchmark_zero_cutoff_factor_precisely_times_every_top_k_candidate)
{
    migraphx::gpu::context ctx{};
    test_candidate fast{100, 1};
    test_candidate slow{2000, 2};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {fast, slow};
    auto bench                                                 = small_adaptive_benchmark(2);
    bench.coarse_cutoff_factor                                 = 0;
    const auto& winner                                         = bench.run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 1);
    EXPECT(fast.log->str().find("Precise solution") != std::string::npos);
    EXPECT(slow.log->str().find("Precise solution") != std::string::npos);
}

TEST_CASE(simple_benchmark_regenerates_a_shared_argument_whose_shape_differs)
{
    migraphx::gpu::context ctx{};
    test_candidate small{100, 1};
    small.scratch = 16;
    test_candidate large{100, 2};
    large.scratch                                              = 32;
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {small, large};
    (void)migraphx::gpu::simple_benchmark{/* bundle */ 1, /* nruns */ 2}.run(ctx, candidates);
    EXPECT(small.generated->size() == 2);
    // large reuses x, but not the scratch, whose key matches and whose shape does not
    EXPECT(large.generated->size() == 1);
    EXPECT(large.generated->front().second == migraphx::shape{migraphx::shape::int8_type, {32}});
}

TEST_CASE(fill_map_argument_keys_differ_for_close_fill_values)
{
    migraphx::gpu::context ctx{};
    migraphx::shape s{migraphx::shape::double_type, {4}};
    migraphx::program p;
    {
        auto* mm = p.get_main_module();
        mm->add_return({mm->add_parameter("x", s)});
    }
    auto id = s.type_string() + migraphx::shape::to_sizes_string({s});
    std::unordered_map<std::string, double> first_fill_map  = {{id, 3.0000001}};
    std::unordered_map<std::string, double> second_fill_map = {{id, 3.0000002}};
    auto first  = migraphx::gpu::fill_map_argument_keys(p, first_fill_map);
    auto second = migraphx::gpu::fill_map_argument_keys(p, second_fill_map);
    EXPECT(first.front().first != second.front().first);
    auto arg =
        migraphx::gpu::generate_fill_map_argument(ctx, second_fill_map, second.front().first, s);
    EXPECT(migraphx::gpu::from_gpu(arg) == migraphx::fill_argument(s, 3.0000002));
}

TEST_CASE(fill_map_argument_keys_do_not_confuse_a_parameter_name_with_a_fill_value)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        mm->add_return({mm->add_parameter("x", s)});
    }
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        mm->add_return({mm->add_parameter("x:3.000000", s)});
    }
    std::unordered_map<std::string, double> fill_map = {
        {s.type_string() + migraphx::shape::to_sizes_string({s}), 3}};
    auto filled = migraphx::gpu::fill_map_argument_keys(p1, fill_map);
    auto random = migraphx::gpu::fill_map_argument_keys(p2, {});
    EXPECT(filled.front().first != random.front().first);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
