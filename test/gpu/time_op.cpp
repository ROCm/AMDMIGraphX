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

#include <migraphx/gpu/compile_ops.hpp>
#include <migraphx/serialize.hpp>
#include <test.hpp>
#include <algorithm>
#include <limits>

TEST_CASE(adaptive_tuning_default_budgets)
{
    const migraphx::gpu::adaptive_tuning_options options;

    EXPECT(options.coarse.target_ms == 10);
    EXPECT(options.precise.target_ms == 20);
    EXPECT(options.coarse.warmup_ms == 0);
    EXPECT(options.precise.warmup_ms == 0);
    EXPECT(options.coarse.estimate_runs == 1);
    EXPECT(options.precise.estimate_runs == 1);
    EXPECT(options.top_k == 10);
}

TEST_CASE(adaptive_time_schedule_slow_candidate)
{
    migraphx::gpu::adaptive_time_options options;
    options.target_ms        = 100;
    options.preferred_bundle = 10;

    const auto schedule = migraphx::gpu::make_timing_schedule(1000.0, options);

    // A candidate slower than the whole budget still takes min_samples samples, otherwise a
    // single interfered run would decide the result.
    EXPECT(schedule.bundle == 1);
    EXPECT(schedule.samples == options.min_samples);
    EXPECT(schedule.executions == options.min_samples);
    EXPECT(schedule.warmup_runs == 1);
}

TEST_CASE(adaptive_time_schedule_coarse_stage_keeps_slow_candidate_cheap)
{
    const migraphx::gpu::adaptive_tuning_options tuning;

    const auto schedule = migraphx::gpu::make_timing_schedule(1000.0, tuning.coarse);

    // The coarse stage runs on every candidate and only has to rank them, so it does not pay the
    // min_samples floor.
    EXPECT(schedule.bundle == 1);
    EXPECT(schedule.samples == 1);
    EXPECT(schedule.executions == 1);
}

TEST_CASE(adaptive_time_schedule_clamps_max_samples_below_min)
{
    migraphx::gpu::adaptive_time_options options;
    options.target_ms   = 100;
    options.max_samples = 2;

    const auto schedule = migraphx::gpu::make_timing_schedule(1000.0, options);

    EXPECT(options.max_samples < options.min_samples);
    EXPECT(schedule.samples == options.max_samples);
}

TEST_CASE(adaptive_time_schedule_uses_preferred_bundle)
{
    migraphx::gpu::adaptive_time_options options;
    options.target_ms        = 100;
    options.preferred_bundle = 10;

    const auto schedule = migraphx::gpu::make_timing_schedule(1.0, options);

    EXPECT(schedule.bundle == 10);
    EXPECT(schedule.samples == 10);
    EXPECT(schedule.executions == 100);
    EXPECT(schedule.warmup_runs == options.warmup_ms);
}

TEST_CASE(adaptive_time_schedule_caps_fast_candidate)
{
    migraphx::gpu::adaptive_time_options options;
    options.target_ms        = 100;
    options.preferred_bundle = 10;

    const auto schedule = migraphx::gpu::make_timing_schedule(0.001, options);

    EXPECT(schedule.bundle == 500);
    EXPECT(schedule.samples == 20);
    EXPECT(schedule.executions == options.max_executions);
    EXPECT(schedule.warmup_runs == options.max_warmup_runs);
}

TEST_CASE(adaptive_time_schedule_caps_at_fixed_benchmark_work)
{
    migraphx::gpu::adaptive_time_options options;
    options.target_ms        = 20;
    options.preferred_bundle = 10;
    options.max_executions   = 20 * options.preferred_bundle;

    const auto schedule = migraphx::gpu::make_timing_schedule(0.001, options);

    EXPECT(schedule.bundle == options.preferred_bundle);
    EXPECT(schedule.samples == options.max_samples);
    EXPECT(schedule.executions == options.max_executions);
}

TEST_CASE(adaptive_time_schedule_hard_cap_includes_minimum_samples)
{
    migraphx::gpu::adaptive_time_options options;
    options.target_ms      = 20;
    options.min_samples    = 4;
    options.max_executions = 3;

    const auto schedule = migraphx::gpu::make_timing_schedule(1000.0, options);

    EXPECT(schedule.bundle == 1);
    EXPECT(schedule.samples == options.max_executions);
    EXPECT(schedule.executions == options.max_executions);
}

TEST_CASE(adaptive_time_loop_respects_precise_total_budget)
{
    migraphx::gpu::context ctx;
    migraphx::gpu::adaptive_time_options options;
    options.warmup_ms     = 5;
    options.estimate_runs = 1;
    migraphx::gpu::adaptive_time_budget budget;
    budget.max_executions  = 21;
    std::size_t executions = 0;

    migraphx::gpu::adaptive_time_loop(ctx, options, budget, [&] { executions++; });

    EXPECT(executions <= budget.max_executions);
}

TEST_CASE(adaptive_time_loop_shares_coarse_and_precise_total_budget)
{
    migraphx::gpu::context ctx;
    migraphx::gpu::adaptive_tuning_options tuning;
    constexpr std::size_t preferred_bundle = 10;
    constexpr std::size_t candidate_budget = 1 + 20 * preferred_bundle;
    std::size_t executions = 0;

    auto coarse             = tuning.coarse;
    coarse.preferred_bundle = preferred_bundle;
    migraphx::gpu::adaptive_time_budget coarse_budget;
    coarse_budget.max_executions = 2 + preferred_bundle;
    const auto coarse_time =
        migraphx::gpu::adaptive_time_loop(ctx, coarse, coarse_budget, [&] { executions++; });

    auto precise             = tuning.precise;
    precise.preferred_bundle = preferred_bundle;
    precise.estimated_ms     = coarse_time;
    migraphx::gpu::adaptive_time_budget precise_budget;
    precise_budget.skip_initialization = true;
    precise_budget.max_executions      = candidate_budget - executions;
    precise.max_executions = precise_budget.max_executions;
    migraphx::gpu::adaptive_time_loop(
        ctx, precise, precise_budget, [&] { executions++; });

    EXPECT(executions <= candidate_budget);
}

TEST_CASE(adaptive_time_program_reuses_prepared_setup_and_budget)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto data = mm->add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {1}});
    mm->add_return({data});

    migraphx::gpu::context ctx;
    auto prepared = migraphx::gpu::prepare_time_program(ctx, std::move(p), {});
    const auto params = prepared.params;
    constexpr std::size_t candidate_budget = 21;

    migraphx::gpu::adaptive_tuning_options tuning;
    migraphx::gpu::adaptive_time_budget coarse_budget;
    coarse_budget.max_executions = 3;
    const auto coarse_time =
        migraphx::gpu::adaptive_time_program(prepared, tuning.coarse, coarse_budget);
    const auto coarse_executions = prepared.executions;

    auto precise         = tuning.precise;
    precise.estimated_ms = coarse_time;
    migraphx::gpu::adaptive_time_budget precise_budget;
    precise_budget.max_executions = candidate_budget - coarse_executions;
    migraphx::gpu::adaptive_time_program(prepared, precise, precise_budget);

    EXPECT(prepared.params == params);
    EXPECT(coarse_executions > 0);
    EXPECT(prepared.executions <= candidate_budget);
}

TEST_CASE(adaptive_time_schedule_rejects_nonfinite_estimate)
{
    migraphx::gpu::adaptive_time_options options;
    EXPECT(test::throws([&] {
        migraphx::gpu::make_timing_schedule(std::numeric_limits<double>::quiet_NaN(), options);
    }));
}

TEST_CASE(compile_ops_tuning_overrides_resolve)
{
    migraphx::value values;
    values["tuning_top_k"]             = 0;
    values["tuning_coarse_target_ms"]  = 3;
    values["tuning_precise_target_ms"] = 30;
    values["tuning_max_samples"]       = 8;
    values["tuning_sleep_us"]          = 0;

    const auto overrides =
        migraphx::from_value<migraphx::gpu::compile_ops_tuning_overrides>(values);
    const auto options = overrides.resolve();
    EXPECT(options.top_k == 0);
    EXPECT(options.coarse.target_ms == 3);
    EXPECT(options.precise.target_ms == 30);
    EXPECT(options.coarse.max_samples == 8);
    EXPECT(options.precise.max_samples == 8);
    EXPECT(options.sleep_us == 0);
}

TEST_CASE(compile_ops_tuning_overrides_reject_invalid_values)
{
    migraphx::gpu::compile_ops_tuning_overrides negative_top_k;
    negative_top_k.top_k = -1;
    EXPECT(test::throws([&] { negative_top_k.resolve(); }));

    migraphx::gpu::compile_ops_tuning_overrides zero_target;
    zero_target.coarse_target_ms = 0;
    EXPECT(test::throws([&] { zero_target.resolve(); }));
}

TEST_CASE(adaptive_time_topk_shortlists_candidates)
{
    migraphx::gpu::adaptive_tuning_options options;
    options.top_k                           = 2;
    options.sleep_us                        = 0;
    const std::vector<double> coarse_times  = {5.0, 1.0, 3.0, 2.0, 4.0};
    const std::vector<double> precise_times = {5.0, 4.0, 3.0, 1.0, 2.0};
    std::vector<std::size_t> coarse_calls(coarse_times.size());
    std::vector<std::size_t> precise_calls(precise_times.size());

    const auto result = migraphx::gpu::adaptive_time_topk_staged(
        coarse_times.size(),
        options,
        [&](auto i, auto stage, const auto&) -> migraphx::optional<double> {
            if(stage == migraphx::gpu::adaptive_time_stage::coarse)
            {
                coarse_calls[i]++;
                return coarse_times[i];
            }
            precise_calls[i]++;
            return precise_times[i];
        });

    EXPECT(result == 3);
    EXPECT(std::all_of(coarse_calls.begin(), coarse_calls.end(), [](auto n) { return n == 1; }));
    EXPECT(precise_calls == std::vector<std::size_t>{0, 1, 0, 1, 0});
}

TEST_CASE(adaptive_time_topk_zero_measures_every_candidate_precisely)
{
    migraphx::gpu::adaptive_tuning_options options;
    options.top_k                           = 0;
    options.sleep_us                        = 0;
    const std::vector<double> precise_times = {3.0, 1.0, 2.0};
    std::size_t precise_calls               = 0;

    const auto result = migraphx::gpu::adaptive_time_topk_staged(
        precise_times.size(), options, [&](auto i, auto stage, const auto&)
                                          -> migraphx::optional<double> {
            EXPECT(stage == migraphx::gpu::adaptive_time_stage::precise);
            precise_calls++;
            return precise_times[i];
        });

    EXPECT(result == 1);
    EXPECT(precise_calls == precise_times.size());
}

TEST_CASE(adaptive_time_topk_allows_max_samples_below_min)
{
    migraphx::gpu::adaptive_tuning_options options;
    options.top_k               = 0;
    options.sleep_us            = 0;
    options.precise.max_samples = 2;
    std::vector<migraphx::gpu::adaptive_time_options> seen;

    const auto result = migraphx::gpu::adaptive_time_topk(
        2, options, [&](auto i, const auto& o) -> migraphx::optional<double> {
            seen.push_back(o);
            return i == 0 ? 2.0 : 1.0;
        });

    EXPECT(result == 1);
    EXPECT(seen.size() == 2);
    EXPECT(std::all_of(
        seen.begin(), seen.end(), [](const auto& o) { return o.min_samples <= o.max_samples; }));
}

TEST_CASE(adaptive_time_topk_promotes_after_precise_failure)
{
    migraphx::gpu::adaptive_tuning_options options;
    options.top_k                          = 2;
    options.sleep_us                       = 0;
    const std::vector<double> coarse_times = {1.0, 2.0, 3.0, 4.0};
    std::vector<std::size_t> precise_calls(coarse_times.size());

    const auto result = migraphx::gpu::adaptive_time_topk_staged(
        coarse_times.size(),
        options,
        [&](auto i, auto stage, const auto&) -> migraphx::optional<double> {
            if(stage == migraphx::gpu::adaptive_time_stage::coarse)
                return coarse_times[i];
            precise_calls[i]++;
            if(i == 0)
                return migraphx::nullopt;
            return i == 2 ? 1.0 : 2.0;
        });

    EXPECT(result == 2);
    EXPECT(precise_calls == std::vector<std::size_t>{1, 1, 1, 0});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }

