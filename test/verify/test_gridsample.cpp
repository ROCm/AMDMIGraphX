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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// Compares the GPU gridsample kernel against the reference operator across the
// supported modes, padding modes and align_corners settings. The ONNX verify
// tests only ever compile the reference target, so this is the only coverage
// the JIT kernel gets.

namespace {

constexpr std::array<const char*, 3> gridsample_modes{"linear", "nearest", "cubic"};
constexpr std::array<const char*, 3> gridsample_paddings{"zeros", "border", "reflection"};

// Grid coordinates deliberately run past [-1, 1] so every padding mode has
// out-of-range taps to resolve; a grid that stayed in range would exercise the
// same code path for all three.
std::vector<float> make_grid_data(std::size_t n)
{
    std::vector<float> data(n);
    for(std::size_t i = 0; i < n; ++i)
        data[i] = -1.4f + 2.8f * (static_cast<float>(i % 17) / 16.0f);
    return data;
}

} // namespace

template <migraphx::shape::type_t DType, int ModeIdx, int PadIdx, bool AlignCorners>
struct test_gridsample : verify_program<test_gridsample<DType, ModeIdx, PadIdx, AlignCorners>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape xs{DType, {2, 3, 4, 5}};
        migraphx::shape gs{DType, {2, 3, 4, 2}};

        auto x    = mm->add_parameter("x", xs);
        auto grid = mm->add_literal(migraphx::literal{gs, make_grid_data(gs.elements())});

        mm->add_instruction(migraphx::make_op("gridsample",
                                              {{"mode", gridsample_modes.at(ModeIdx)},
                                               {"padding_mode", gridsample_paddings.at(PadIdx)},
                                               {"align_corners", AlignCorners}}),
                            x,
                            grid);
        return p;
    }
};

// float: full mode x padding x align_corners matrix
template struct test_gridsample<migraphx::shape::float_type, 0, 0, false>;
template struct test_gridsample<migraphx::shape::float_type, 0, 0, true>;
template struct test_gridsample<migraphx::shape::float_type, 0, 1, false>;
template struct test_gridsample<migraphx::shape::float_type, 0, 1, true>;
template struct test_gridsample<migraphx::shape::float_type, 0, 2, false>;
template struct test_gridsample<migraphx::shape::float_type, 0, 2, true>;
template struct test_gridsample<migraphx::shape::float_type, 1, 0, false>;
template struct test_gridsample<migraphx::shape::float_type, 1, 0, true>;
template struct test_gridsample<migraphx::shape::float_type, 1, 1, false>;
template struct test_gridsample<migraphx::shape::float_type, 1, 1, true>;
template struct test_gridsample<migraphx::shape::float_type, 1, 2, false>;
template struct test_gridsample<migraphx::shape::float_type, 1, 2, true>;
template struct test_gridsample<migraphx::shape::float_type, 2, 0, false>;
template struct test_gridsample<migraphx::shape::float_type, 2, 0, true>;
template struct test_gridsample<migraphx::shape::float_type, 2, 1, false>;
template struct test_gridsample<migraphx::shape::float_type, 2, 1, true>;
template struct test_gridsample<migraphx::shape::float_type, 2, 2, false>;
template struct test_gridsample<migraphx::shape::float_type, 2, 2, true>;

// half: one case per mode and per padding mode
template struct test_gridsample<migraphx::shape::half_type, 0, 0, false>;
template struct test_gridsample<migraphx::shape::half_type, 1, 0, false>;
template struct test_gridsample<migraphx::shape::half_type, 2, 0, false>;
template struct test_gridsample<migraphx::shape::half_type, 0, 1, true>;
template struct test_gridsample<migraphx::shape::half_type, 0, 2, true>;

// A single-pixel spatial dimension gives reflection a zero-width span; this
// used to produce NaNs on both targets.
template <migraphx::shape::type_t DType, bool AlignCorners>
struct test_gridsample_singleton : verify_program<test_gridsample_singleton<DType, AlignCorners>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape xs{DType, {1, 1, 1, 1}};
        migraphx::shape gs{DType, {1, 2, 2, 2}};

        auto x    = mm->add_parameter("x", xs);
        auto grid = mm->add_literal(migraphx::literal{gs, make_grid_data(gs.elements())});

        mm->add_instruction(migraphx::make_op("gridsample",
                                              {{"mode", "linear"},
                                               {"padding_mode", "reflection"},
                                               {"align_corners", AlignCorners}}),
                            x,
                            grid);
        return p;
    }
};

template struct test_gridsample_singleton<migraphx::shape::float_type, true>;
template struct test_gridsample_singleton<migraphx::shape::float_type, false>;

// Non-standard (transposed) input. The operator dropped its require_std_shape
// attribute, so the kernel must read a permuted-stride view directly instead of
// relying on a contiguous copy being inserted ahead of it.
template <migraphx::shape::type_t DType>
struct test_gridsample_transposed : verify_program<test_gridsample_transposed<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape xs{DType, {2, 4, 5, 3}};
        migraphx::shape gs{DType, {2, 3, 4, 2}};

        auto x = mm->add_parameter("x", xs);
        auto xt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), x);
        auto grid = mm->add_literal(migraphx::literal{gs, make_grid_data(gs.elements())});

        mm->add_instruction(
            migraphx::make_op(
                "gridsample",
                {{"mode", "linear"}, {"padding_mode", "zeros"}, {"align_corners", false}}),
            xt,
            grid);
        return p;
    }
};

template struct test_gridsample_transposed<migraphx::shape::float_type>;
