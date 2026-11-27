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
// No GPU/MLIR dependency: these are pure string/arithmetic helpers, tested on the host.
#include <migraphx/gpu/mlir_perf_config.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <test.hpp>

using namespace migraphx::gpu; // NOLINT

TEST_CASE(named_config_detection)
{
    EXPECT(is_named_perf_config("gemm:mPerBlock=128,nPerBlock=128,numCTAs=1"));
    EXPECT(is_named_perf_config("attn:mPerBlock=128,numCTAs=1"));
    // Legacy positional configs have no '=' in their colon-separated fields.
    EXPECT(not is_named_perf_config("gemm:v1:128:128:64:1"));
    // No ':' at all.
    EXPECT(not is_named_perf_config("notaconfig"));
    // Nothing after the ':'.
    EXPECT(not is_named_perf_config("gemm:"));
}

TEST_CASE(prefix)
{
    EXPECT(perf_config_prefix("gemm:mPerBlock=128") == "gemm");
    EXPECT(perf_config_prefix("attn:mPerBlock=128") == "attn");
    EXPECT(perf_config_prefix("noprefix") == "");
}

TEST_CASE(get_field_present)
{
    std::string c = "gemm:mPerBlock=128,nPerBlock=64,numCTAs=4";
    EXPECT(get_perf_config_field(c, "mPerBlock") == 128);
    EXPECT(get_perf_config_field(c, "nPerBlock") == 64);
    EXPECT(get_perf_config_field(c, "numCTAs") == 4);
}

TEST_CASE(get_field_absent_has_schema_default)
{
    std::string c = "gemm:mPerBlock=128,nPerBlock=64";
    EXPECT(get_perf_config_field(c, "numCTAs") == 1);
    EXPECT(get_perf_config_field(c, "numWaves") == 4);
    EXPECT(get_perf_config_field(c, "useWarpSpecialize") == -1);
}

TEST_CASE(get_field_absent_unknown_key)
{
    std::string c = "gemm:mPerBlock=128";
    EXPECT(not get_perf_config_field(c, "kPerBlock").has_value());
}

TEST_CASE(get_field_not_named)
{
    EXPECT(not get_perf_config_field("gemm:v1:128:128", "numCTAs").has_value());
}

TEST_CASE(set_field_replaces)
{
    std::string c        = "gemm:mPerBlock=128,numCTAs=1,numWaves=4";
    std::string replaced = set_perf_config_field(c, "numCTAs", 4);
    EXPECT(get_perf_config_field(replaced, "numCTAs") == 4);
    // Everything else is untouched.
    EXPECT(get_perf_config_field(replaced, "mPerBlock") == 128);
    EXPECT(get_perf_config_field(replaced, "numWaves") == 4);
}

TEST_CASE(set_field_appends)
{
    std::string c       = "gemm:mPerBlock=128,nPerBlock=128";
    std::string appended = set_perf_config_field(c, "useWarpSpecialize", 1);
    EXPECT(get_perf_config_field(appended, "useWarpSpecialize") == 1);
    EXPECT(get_perf_config_field(appended, "mPerBlock") == 128);
}

TEST_CASE(set_field_not_named_unchanged)
{
    std::string c = "gemm:v1:128:128";
    EXPECT(set_perf_config_field(c, "numCTAs", 4) == c);
}

TEST_CASE(tile_splittable)
{
    // num_ctas <= 1 never splits anything, so it always passes.
    EXPECT(is_tile_splittable(1, 1, 1));
    EXPECT(is_tile_splittable(1, 1, 0));
    // rocMLIR's own documented cases.
    EXPECT(not is_tile_splittable(16, 16, 4));
    EXPECT(is_tile_splittable(128, 128, 16));
    EXPECT(not is_tile_splittable(16, 64, 4));
    // A tile that splits exactly to the minimum on both dims.
    EXPECT(is_tile_splittable(32, 32, 4));
}

TEST_CASE(num_ctas_zero_unchanged)
{
    std::vector<std::string> sols = {"gemm:mPerBlock=128,nPerBlock=128,numCTAs=1",
                                     "gemm:mPerBlock=32,nPerBlock=32,numCTAs=4"};
    EXPECT(apply_num_ctas(sols, 0, 16) == sols);
}

TEST_CASE(num_ctas_one_filters)
{
    std::vector<std::string> sols = {"gemm:mPerBlock=128,nPerBlock=128,numCTAs=1",
                                     "gemm:mPerBlock=32,nPerBlock=32,numCTAs=4",
                                     "attn:mPerBlock=128,numCTAs=1"};
    auto result = apply_num_ctas(sols, 1, 16);
    EXPECT(result.size() == 2);
    EXPECT(migraphx::contains(result, sols[0]));
    EXPECT(migraphx::contains(result, sols[2]));
}

TEST_CASE(num_ctas_rewrites_splittable_gemm)
{
    std::vector<std::string> sols = {
        "gemm:mPerBlock=128,nPerBlock=128,numCTAs=1",  // splittable by 4 (32x128 per cta)
        "gemm:mPerBlock=16,nPerBlock=16,numCTAs=1",    // not splittable by 4
        "gemm:mPerBlock=64,nPerBlock=64,numCTAs=4",    // already at the requested size
        "attn:mPerBlock=128,numCTAs=1",                // not a gemm: config, left out
        "gemm:v1:128:128:64:1"};                       // positional, left out
    auto result = apply_num_ctas(sols, 4, 16);
    EXPECT(migraphx::contains(result, std::string{"gemm:mPerBlock=128,nPerBlock=128,numCTAs=4"}));
    EXPECT(migraphx::contains(result, sols[2]));
    EXPECT(not migraphx::contains(result, sols[1]));
    EXPECT(not migraphx::contains(result, sols[3]));
    EXPECT(not migraphx::contains(result, sols[4]));
    EXPECT(result.size() == 2);
}

TEST_CASE(num_ctas_dedupes)
{
    std::vector<std::string> sols = {"gemm:mPerBlock=128,nPerBlock=128,numCTAs=1",
                                     "gemm:mPerBlock=128,nPerBlock=128,numCTAs=4"};
    auto result = apply_num_ctas(sols, 4, 16);
    EXPECT(result.size() == 1);
}

TEST_CASE(num_ctas_empty_result_falls_back_to_one)
{
    // Every tile is too small to split by 4; the numCTAs=1 list is returned instead.
    std::vector<std::string> sols = {"gemm:mPerBlock=16,nPerBlock=16,numCTAs=1",
                                     "attn:mPerBlock=128,numCTAs=1"};
    auto result = apply_num_ctas(sols, 4, 16);
    EXPECT(result == apply_num_ctas(sols, 1, 16));
}

TEST_CASE(num_ctas_clamped_to_device_max)
{
    std::vector<std::string> sols = {"gemm:mPerBlock=128,nPerBlock=128,numCTAs=1"};
    // Requesting 16 on a device whose max is 8 behaves as if 8 was requested.
    EXPECT(apply_num_ctas(sols, 16, 8) == apply_num_ctas(sols, 8, 8));
}

TEST_CASE(warp_specialize_eligibility)
{
    EXPECT(is_warp_specialize_eligible("gemm:mPerBlock=128,numWaves=4,numCTAs=1"));
    EXPECT(not is_warp_specialize_eligible("attn:mPerBlock=128,numWaves=4,numCTAs=1"));
    EXPECT(not is_warp_specialize_eligible("gemm:mPerBlock=128,numWaves=8,numCTAs=1"));
    EXPECT(not is_warp_specialize_eligible("gemm:mPerBlock=128,numWaves=4,numCTAs=2"));
    EXPECT(not is_warp_specialize_eligible(
        "gemm:mPerBlock=128,numWaves=4,numCTAs=1,useWarpSpecialize=1"));
    EXPECT(not is_warp_specialize_eligible("gemm:v1:128:128"));
}

TEST_CASE(warp_specialize_off_unchanged)
{
    std::vector<std::string> sols = {"gemm:mPerBlock=128,numWaves=4,numCTAs=1"};
    EXPECT(apply_warp_specialize(sols, warp_specialize_mode::off) == sols);
}

TEST_CASE(warp_specialize_tune_appends)
{
    std::vector<std::string> sols = {"gemm:mPerBlock=128,numWaves=4,numCTAs=1",
                                     "attn:mPerBlock=128,numWaves=4,numCTAs=1"};
    auto result = apply_warp_specialize(sols, warp_specialize_mode::tune);
    EXPECT(result.size() == 3);
    // Originals come first, unmodified.
    EXPECT(result[0] == sols[0]);
    EXPECT(result[1] == sols[1]);
    EXPECT(get_perf_config_field(result[2], "useWarpSpecialize") == 1);
    EXPECT(get_perf_config_field(result[2], "mPerBlock") == 128);
}

TEST_CASE(warp_specialize_force_rewrites_in_place)
{
    std::vector<std::string> sols = {"gemm:mPerBlock=128,numWaves=4,numCTAs=1",
                                     "attn:mPerBlock=128,numWaves=4,numCTAs=1",
                                     "gemm:mPerBlock=64,numWaves=8,numCTAs=1"};
    auto result = apply_warp_specialize(sols, warp_specialize_mode::force);
    EXPECT(result.size() == sols.size());
    EXPECT(get_perf_config_field(result[0], "useWarpSpecialize") == 1);
    // Ineligible candidates (attn:, numWaves != 4) are untouched.
    EXPECT(result[1] == sols[1]);
    EXPECT(result[2] == sols[2]);
}

TEST_CASE(warp_specialize_mode_names)
{
    EXPECT(to_string(warp_specialize_mode::off) == "off");
    EXPECT(to_string(warp_specialize_mode::tune) == "tune");
    EXPECT(to_string(warp_specialize_mode::force) == "force");
    EXPECT(migraphx::from_string<warp_specialize_mode>("force") == warp_specialize_mode::force);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
