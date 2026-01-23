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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// Nearest mode downsample with floor rounding (1-input mode with scales as attribute)
template <migraphx::shape::type_t DType>
struct test_resize_nearest_downsample_floor
    : verify_program<test_resize_nearest_downsample_floor<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 4, 8}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1.0f, 1.0f, 0.5f, 0.5f}},
                                               {"nearest_mode", "floor"},
                                               {"coordinate_transformation_mode", "asymmetric"}}),
                            x);
        return p;
    }
};

template struct test_resize_nearest_downsample_floor<migraphx::shape::float_type>;
template struct test_resize_nearest_downsample_floor<migraphx::shape::half_type>;

// Nearest mode upsample with round_prefer_floor rounding
template <migraphx::shape::type_t DType>
struct test_resize_nearest_upsample_pf : verify_program<test_resize_nearest_upsample_pf<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 2, 2}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1.0f, 1.0f, 2.0f, 3.0f}},
                                               {"nearest_mode", "round_prefer_floor"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            x);
        return p;
    }
};

template struct test_resize_nearest_upsample_pf<migraphx::shape::float_type>;
template struct test_resize_nearest_upsample_pf<migraphx::shape::half_type>;

// Nearest mode with ceil rounding
template <migraphx::shape::type_t DType>
struct test_resize_nearest_ceil : verify_program<test_resize_nearest_ceil<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 2, 4}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1.0f, 1.0f, 0.6f, 0.6f}},
                                               {"nearest_mode", "ceil"},
                                               {"coordinate_transformation_mode", "asymmetric"}}),
                            x);
        return p;
    }
};

template struct test_resize_nearest_ceil<migraphx::shape::float_type>;
template struct test_resize_nearest_ceil<migraphx::shape::half_type>;

// Nearest mode with round_prefer_ceil rounding
template <migraphx::shape::type_t DType>
struct test_resize_nearest_round_prefer_ceil
    : verify_program<test_resize_nearest_round_prefer_ceil<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 2, 4}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1.0f, 1.0f, 2.0f, 1.5f}},
                                               {"nearest_mode", "round_prefer_ceil"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            x);
        return p;
    }
};

template struct test_resize_nearest_round_prefer_ceil<migraphx::shape::float_type>;
template struct test_resize_nearest_round_prefer_ceil<migraphx::shape::half_type>;

// Linear mode downsample
template <migraphx::shape::type_t DType>
struct test_resize_linear_downsample : verify_program<test_resize_linear_downsample<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 4, 8}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1.0f, 1.0f, 0.5f, 0.5f}},
                                               {"mode", "linear"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            x);
        return p;
    }
};

template struct test_resize_linear_downsample<migraphx::shape::float_type>;
template struct test_resize_linear_downsample<migraphx::shape::half_type>;

// Linear mode upsample
template <migraphx::shape::type_t DType>
struct test_resize_linear_upsample : verify_program<test_resize_linear_upsample<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 2, 2}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1.0f, 1.0f, 2.0f, 2.0f}},
                                               {"mode", "linear"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            x);
        return p;
    }
};

template struct test_resize_linear_upsample<migraphx::shape::float_type>;
template struct test_resize_linear_upsample<migraphx::shape::half_type>;

// Linear mode with align_corners
template <migraphx::shape::type_t DType>
struct test_resize_linear_align_corners : verify_program<test_resize_linear_align_corners<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 2, 2}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(
            migraphx::make_op("resize",
                              {{"scales", {1.0f, 1.0f, 2.0f, 2.0f}},
                               {"mode", "linear"},
                               {"coordinate_transformation_mode", "align_corners"}}),
            x);
        return p;
    }
};

template struct test_resize_linear_align_corners<migraphx::shape::float_type>;
template struct test_resize_linear_align_corners<migraphx::shape::half_type>;

// Linear mode with asymmetric
template <migraphx::shape::type_t DType>
struct test_resize_linear_asymmetric : verify_program<test_resize_linear_asymmetric<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 2, 4}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1.0f, 1.0f, 2.0f, 2.0f}},
                                               {"mode", "linear"},
                                               {"coordinate_transformation_mode", "asymmetric"}}),
                            x);
        return p;
    }
};

template struct test_resize_linear_asymmetric<migraphx::shape::float_type>;
template struct test_resize_linear_asymmetric<migraphx::shape::half_type>;

// Using sizes attribute (nearest mode)
template <migraphx::shape::type_t DType>
struct test_resize_nearest_sizes : verify_program<test_resize_nearest_sizes<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 2, 2}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"sizes", {1, 1, 4, 6}},
                                               {"nearest_mode", "round_prefer_floor"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            x);
        return p;
    }
};

template struct test_resize_nearest_sizes<migraphx::shape::float_type>;
template struct test_resize_nearest_sizes<migraphx::shape::half_type>;

// 3D resize (nearest mode)
template <migraphx::shape::type_t DType>
struct test_resize_3d_nearest : verify_program<test_resize_3d_nearest<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 2, 2, 2}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1.0f, 1.0f, 2.0f, 2.0f, 2.0f}},
                                               {"nearest_mode", "floor"},
                                               {"coordinate_transformation_mode", "asymmetric"}}),
                            x);
        return p;
    }
};

template struct test_resize_3d_nearest<migraphx::shape::float_type>;
template struct test_resize_3d_nearest<migraphx::shape::half_type>;

// 3D resize (linear mode)
template <migraphx::shape::type_t DType>
struct test_resize_3d_linear : verify_program<test_resize_3d_linear<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 2, 2, 2}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"scales", {1.0f, 1.0f, 2.0f, 2.0f, 2.0f}},
                                               {"mode", "linear"},
                                               {"coordinate_transformation_mode", "half_pixel"}}),
                            x);
        return p;
    }
};

template struct test_resize_3d_linear<migraphx::shape::float_type>;
template struct test_resize_3d_linear<migraphx::shape::half_type>;

// Test with asymmetric instead of half_pixel (3->5x8, non-integer scales)
template <migraphx::shape::type_t DType>
struct test_resize_nonint_asymmetric : verify_program<test_resize_nonint_asymmetric<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{DType, {1, 1, 3, 3}};
        auto x = mm->add_parameter("X", sx);

        mm->add_instruction(migraphx::make_op("resize",
                                              {{"sizes", {1, 1, 5, 8}},
                                               {"nearest_mode", "floor"},
                                               {"coordinate_transformation_mode", "asymmetric"}}),
                            x);
        return p;
    }
};

template struct test_resize_nonint_asymmetric<migraphx::shape::float_type>;
template struct test_resize_nonint_asymmetric<migraphx::shape::half_type>;
