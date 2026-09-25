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
#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

// Identity op that sleeps on the host, so a candidate's measured time is deterministic
struct sleep_op
{
    std::size_t usec = 0;
    // The first slow_launches launches sleep for slow_usec instead, like a transiently slow GPU
    std::size_t slow_launches = 0;
    std::size_t slow_usec     = 0;
    std::shared_ptr<std::size_t> launches{};

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
        const bool slow = launches != nullptr and *launches <= slow_launches;
        std::this_thread::sleep_for(std::chrono::microseconds{slow ? slow_usec : usec});
        return args.front();
    }
};

// Buffers generated for the candidates sharing it, to observe how many are still resident when
// another candidate's arguments are generated
struct argument_buffers
{
    std::vector<std::weak_ptr<std::array<float, 4>>> generated;
    std::size_t most_resident = 0;
};

// Benchmark candidate that runs for `usec` microseconds and is identified by its solution id
struct test_candidate
{
    std::size_t usec = 0;
    int id           = 0;
    bool fail        = false;
    // Makes the first launches of the candidate's programs slower, see sleep_op
    std::size_t slow_launches = 0;
    std::size_t slow_usec     = 0;
    // Counts make_program calls to observe which candidates each benchmark pass builds
    std::shared_ptr<std::size_t> programs_built = std::make_shared<std::size_t>(0);
    std::shared_ptr<std::size_t> launches       = std::make_shared<std::size_t>(0);
    // Trace output, to observe which candidates the precise pass times
    std::shared_ptr<std::stringstream> log      = std::make_shared<std::stringstream>();
    std::shared_ptr<argument_buffers> arguments = std::make_shared<argument_buffers>();

    std::vector<migraphx::argument> generate_arguments(const migraphx::gpu::context&,
                                                       const migraphx::program&) const
    {
        const auto& generated = arguments->generated;
        std::size_t resident =
            std::count_if(generated.begin(), generated.end(), [](const auto& buffer) {
                return not buffer.expired();
            });
        arguments->most_resident = std::max(arguments->most_resident, resident);
        auto buffer              = std::make_shared<std::array<float, 4>>();
        arguments->generated.push_back(buffer);
        return {migraphx::argument{{migraphx::shape::float_type, {4}}, buffer}};
    }

    migraphx::program make_program() const
    {
        if(fail)
            MIGRAPHX_THROW("test candidate failure");
        ++(*programs_built);
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {4}});
        mm->add_return(
            {mm->add_instruction(sleep_op{usec, slow_launches, slow_usec, launches}, x)});
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
    EXPECT(*fast.programs_built == 2);
    EXPECT(*slow.programs_built == 2);
}

TEST_CASE(adaptive_benchmark_skips_second_measurement_when_estimate_exceeds_coarse_budget)
{
    migraphx::gpu::context ctx{};
    test_candidate slow{20000, 1};
    test_candidate fast{100, 2};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {slow, fast};
    const auto& winner = small_adaptive_benchmark(1).run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 2);
    // Only the warmup and the estimate; top_k = 1 keeps slow out of the precise pass
    EXPECT(*slow.launches == 2);
}

TEST_CASE(adaptive_benchmark_measures_a_candidate_whose_estimate_misses_the_top_k)
{
    migraphx::gpu::context ctx{};
    test_candidate steady{300, 1};
    test_candidate transient{100, 2};
    // The warmup and the estimate are slower than steady, the later runs are faster
    transient.slow_launches                                    = 2;
    transient.slow_usec                                        = 1000;
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {steady, transient};
    const auto& winner = small_adaptive_benchmark(1).run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 2);
}

TEST_CASE(adaptive_benchmark_top_k_zero_still_measures_candidates_under_the_coarse_budget)
{
    migraphx::gpu::context ctx{};
    test_candidate fast{100, 1};
    test_candidate mid{1000, 2};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {fast, mid};
    const auto& winner = small_adaptive_benchmark(0).run(ctx, candidates);
    EXPECT(winner.solution().to<int>() == 1);
    EXPECT(*fast.programs_built == 2);
    EXPECT(*mid.programs_built == 2);
    EXPECT(*mid.launches > 2);
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
    // slow enters the top_k first and is evicted by near_best
    EXPECT(slow.log->str().find("Precise solution") == std::string::npos);
    EXPECT(*fast.programs_built == 1);
    EXPECT(*near_best.programs_built == 1);
    EXPECT(*slow.programs_built == 1);
}

TEST_CASE(adaptive_benchmark_does_not_keep_the_arguments_of_the_top_k_resident)
{
    migraphx::gpu::context ctx{};
    test_candidate fast{100, 1};
    test_candidate near_best{150, 2};
    test_candidate mid{300, 3};
    near_best.arguments                                        = fast.arguments;
    mid.arguments                                              = fast.arguments;
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {fast, near_best, mid};
    auto bench                                                 = small_adaptive_benchmark(3);
    bench.coarse_cutoff_factor                                 = 0;
    (void)bench.run(ctx, candidates);
    EXPECT(fast.arguments->most_resident == 0);
    // Arguments are generated for each coarse and each precise timing, but programs only once
    EXPECT(fast.arguments->generated.size() == 6);
    EXPECT(*fast.programs_built == 1);
    EXPECT(*near_best.programs_built == 1);
    EXPECT(*mid.programs_built == 1);
}

TEST_CASE(adaptive_benchmark_precise_pass_skips_the_warmup_of_a_reused_program)
{
    migraphx::gpu::context ctx{};
    test_candidate only{1500, 1};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {only};
    auto bench                                                 = small_adaptive_benchmark(1);
    bench.precise_ms                                           = 1;
    bench.precise_min_bundle                                   = 2;
    (void)bench.run(ctx, candidates);
    EXPECT(*only.programs_built == 1);
    // Coarse: warmup, estimate, then a single run (coarse_ms / 1.5ms rounds down to 1).
    // Precise: one bundle of 2 (precise_ms / (1.5ms * 2) clamps to one run) and no warmup.
    EXPECT(*only.launches == 5);
}

TEST_CASE(adaptive_benchmark_top_k_zero_warms_up_the_rebuilt_program)
{
    migraphx::gpu::context ctx{};
    test_candidate only{1500, 1};
    std::vector<migraphx::gpu::benchmark_candidate> candidates = {only};
    auto bench                                                 = small_adaptive_benchmark(0);
    bench.precise_ms                                           = 1;
    bench.precise_min_bundle                                   = 2;
    (void)bench.run(ctx, candidates);
    EXPECT(*only.programs_built == 2);
    // Same runs as a reused program, plus the warmup of the rebuilt one
    EXPECT(*only.launches == 6);
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

TEST_CASE(generate_program_arguments_shares_parameters_with_the_same_name_and_shape)
{
    migraphx::gpu::context ctx{};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", {migraphx::shape::float_type, {4}});
        auto scratch = mm->add_parameter("scratch", {migraphx::shape::int8_type, {16}});
        mm->add_return({x, scratch});
    }
    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", {migraphx::shape::float_type, {4}});
        auto scratch = mm->add_parameter("scratch", {migraphx::shape::int8_type, {32}});
        mm->add_return({x, scratch});
    }
    std::unordered_map<std::string, migraphx::argument> generated;
    auto args1 = migraphx::gpu::make_parameter_map(
        p1.get_main_module(), migraphx::gpu::generate_program_arguments(ctx, p1, {}, generated));
    auto args2 = migraphx::gpu::make_parameter_map(
        p2.get_main_module(), migraphx::gpu::generate_program_arguments(ctx, p2, {}, generated));
    EXPECT(args1.at("x").data() == args2.at("x").data());
    EXPECT(args1.at("scratch").data() != args2.at("scratch").data());
    EXPECT(args2.at("scratch").get_shape() == migraphx::shape{migraphx::shape::int8_type, {32}});
    // The larger scratch replaced the smaller one
    EXPECT(generated.size() == 2);
}

TEST_CASE(generate_program_arguments_does_not_share_differently_filled_parameters)
{
    migraphx::gpu::context ctx{};
    migraphx::shape s{migraphx::shape::float_type, {4}};
    migraphx::program p;
    {
        auto* mm = p.get_main_module();
        mm->add_return({mm->add_parameter("x", s)});
    }
    std::unordered_map<std::string, double> fill_map = {
        {s.type_string() + migraphx::shape::to_sizes_string({s}), 3}};
    std::unordered_map<std::string, migraphx::argument> generated;
    auto random       = migraphx::gpu::generate_program_arguments(ctx, p, {}, generated);
    auto filled       = migraphx::gpu::generate_program_arguments(ctx, p, fill_map, generated);
    auto filled_again = migraphx::gpu::generate_program_arguments(ctx, p, fill_map, generated);
    auto random_again = migraphx::gpu::generate_program_arguments(ctx, p, {}, generated);
    EXPECT(random.front().data() != filled.front().data());
    EXPECT(filled.front().data() == filled_again.front().data());
    EXPECT(random.front().data() == random_again.front().data());
}

TEST_CASE(generate_program_arguments_does_not_share_parameters_filled_with_close_values)
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
    std::unordered_map<std::string, migraphx::argument> generated;
    auto first  = migraphx::gpu::generate_program_arguments(ctx, p, first_fill_map, generated);
    auto second = migraphx::gpu::generate_program_arguments(ctx, p, second_fill_map, generated);
    EXPECT(first.front().data() != second.front().data());
    EXPECT(migraphx::gpu::from_gpu(second.front()) == migraphx::fill_argument(s, 3.0000002));
}

TEST_CASE(generate_program_arguments_does_not_confuse_a_parameter_name_with_a_fill_value)
{
    migraphx::gpu::context ctx{};
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
    std::unordered_map<std::string, migraphx::argument> generated;
    auto filled = migraphx::gpu::generate_program_arguments(ctx, p1, fill_map, generated);
    auto random = migraphx::gpu::generate_program_arguments(ctx, p2, {}, generated);
    EXPECT(filled.front().data() != random.front().data());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
