/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(roialign_out_of_bound_test)
{
    auto create_program = [](const std::string& trans_mode = "half_pixel") {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape x_s{migraphx::shape::float_type, {1, 1, 10, 10}};
        std::vector<float> x_vec = {
            0.2764, 0.7150, 0.1958, 0.3416, 0.4638, 0.0259, 0.2963, 0.6518, 0.4856, 0.7250,
            0.9637, 0.0895, 0.2919, 0.6753, 0.0234, 0.6132, 0.8085, 0.5324, 0.8992, 0.4467,
            0.3265, 0.8479, 0.9698, 0.2471, 0.9336, 0.1878, 0.4766, 0.4308, 0.3400, 0.2162,
            0.0206, 0.1720, 0.2155, 0.4394, 0.0653, 0.3406, 0.7724, 0.3921, 0.2541, 0.5799,
            0.4062, 0.2194, 0.4473, 0.4687, 0.7109, 0.9327, 0.9815, 0.6320, 0.1728, 0.6119,
            0.3097, 0.1283, 0.4984, 0.5068, 0.4279, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,
            0.1011, 0.8477, 0.4726, 0.1777, 0.9923, 0.4042, 0.1869, 0.7795, 0.9946, 0.9689,
            0.1366, 0.3671, 0.7011, 0.6234, 0.9867, 0.5585, 0.6985, 0.5609, 0.8788, 0.9928,
            0.5697, 0.8511, 0.6711, 0.9406, 0.8751, 0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
            0.7012, 0.4056, 0.7879, 0.3461, 0.0415, 0.2998, 0.5094, 0.3727, 0.5482, 0.0502};

        migraphx::shape roi_s{migraphx::shape::float_type, {3, 4}};
        std::vector<float> roi_vec = {0, 0, 9.99, 9.99, 0, 5, 4, 9, 5, 5, 9.9, 9.9};

        migraphx::shape ind_s{migraphx::shape::int64_type, {3}};
        std::vector<int64_t> ind_vec = {0, 0, 0};

        auto x   = mm->add_literal(migraphx::literal(x_s, x_vec));
        auto roi = mm->add_literal(migraphx::literal(roi_s, roi_vec));
        auto ind = mm->add_literal(migraphx::literal(ind_s, ind_vec));
        auto r =
            mm->add_instruction(migraphx::make_op("roialign",
                                                  {{"coordinate_transformation_mode", trans_mode},
                                                   {"spatial_scale", 5.0},
                                                   {"output_height", 1},
                                                   {"output_width", 1},
                                                   {"sampling_ratio", 1}}),
                                x,
                                roi,
                                ind);
        mm->add_return({r});
        return p;
    };

    {
        auto p = create_program("half_pixel");
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0.0f, 0.0f, 0.0f};

        EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    }
}

auto create_program(const std::string& trans_mode = "half_pixel",
                        const migraphx::op::pooling_mode pooling_mode =
                            migraphx::op::pooling_mode::average,
                        int64_t sampling_ratio = 2) {
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape x_s{migraphx::shape::float_type, {5, 2, 5, 4}};

    // first and second channels are almost but not quite identical
    std::vector<float> x_vec = {
        0.2764, 0.7150, 0.1958, 0.3416, 0.4638,    0.0259, 0.2963, 0.6518, 0.4856, 0.7250,
        0.9637, 0.0895, 0.2919, 0.6753, 0.0234,    0.6132, 0.8085, 0.5324, 0.8992, 0.4467,
        0.2764, 0.7150, 0.1958, 0.3416, 0.4638,    0.0259, 0.2963, 0.6518, 0.4856, 0.7250,
        0.9637, 0.0895, 0.2919, 0.6753, 0.0234,    0.6132, 0.8085, 0.5324, 0.8992, 0.4467,

        0.3265, 0.8479, 0.9698, 0.2471, 0.9336,    0.1878, 0.4766, 0.4308, 0.3400, 0.2162,
        0.0206, 0.1720, 0.2155, 0.4394, 0.0653,    0.3406, 0.7724, 0.3921, 0.2541, 0.5799,
        0.3265, 0.8479, 0.9698, 0.2471, 0.9336,    0.1878, 0.4766, 0.4308, 0.3400, 0.2162,
        0.0206, 0.1720, 0.2155, 0.4394, 0.0653,    0.3406, 0.7724, 0.3921, 0.2541, 0.5799,

        0.4062, 0.2194, 0.4473, 0.4687, 0.7109,    0.9327, 0.9815, 0.6320, 0.1728, 0.6119,
        0.3097, 0.1283, 0.4984, 0.5068, 0.4279,    0.0173, 0.4388, 0.0430, 0.4671, 0.7119,
        0.4062, 0.2194, 0.4473, 0.4687, 0.7109,    0.9327, 0.9815, 0.6320, 0.1728, 0.6119,
        0.3097, 0.1283, 0.4984, 0.5068, -11.70004, 0.0173, 0.4388, 0.0430, 0.4671, 0.7119,

        0.1011, 0.8477, 0.4726, 0.1777, 0.9923,    0.4042, 0.1869, 0.7795, 0.9946, 0.9689,
        0.1366, 0.3671, 0.7011, 0.6234, 0.9867,    0.5585, 0.6985, 0.5609, 0.8788, 0.9928,
        0.1011, 0.8477, 0.4726, 0.1777, 0.9923,    0.4042, 0.1869, 0.7795, 0.9946, 0.9689,
        0.1366, 0.3671, 0.7011, 0.6234, 0.9867,    0.5585, 0.6985, 0.5609, 0.8788, 0.9928,

        0.5697, 0.8511, 0.6711, 0.9406, 0.8751,    0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
        0.7012, 0.4056, 0.7879, 0.3461, 0.0415,    0.2998, 0.5094, 0.3727, 0.5482, 0.0502,
        0.5697, 0.8511, 0.6711, 0.9406, 0.8751,    0.7496, 0.1650, 0.1049, 0.1559, 0.2514,
        0.7012, 0.4056, 0.7879, 0.3461, 0.0415,    0.2998, 0.5094, 0.3727, 0.5482, 0.0502};

    migraphx::shape roi_s{migraphx::shape::float_type, {4, 4}};
    std::vector<float> roi_vec = {
        0, 0, 4, 3, 0, 2, 4, 5.7, 0.1, 0.15, 0.6, 0.35, 2.1, 1.73, 3.8, 2.13};

    // Indices include two unused layers and one layer with two ROI
    migraphx::shape ind_s{migraphx::shape::int64_type, {4}};
    std::vector<int64_t> ind_vec = {2, 2, 4, 0};

    auto x   = mm->add_literal(migraphx::literal(x_s, x_vec));
    auto roi = mm->add_literal(migraphx::literal(roi_s, roi_vec));
    auto ind = mm->add_literal(migraphx::literal(ind_s, ind_vec));
    auto r   = mm->add_instruction(migraphx::make_op("roialign",
                                                    {{"coordinate_transformation_mode", trans_mode},
                                                    {"spatial_scale", 0.9},
                                                    {"output_height", 3},
                                                    {"output_width", 2},
                                                    {"sampling_ratio", sampling_ratio},
                                                    {"mode", pooling_mode}}),
                                x,
                                roi,
                                ind);
    mm->add_return({r});
    return p;
}

TEST_CASE(roialign_test)
{
    auto p = create_program("output_half_pixel");
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        0.5669691,  0.59674937, 0.7255512,   0.5647045,  0.4513675,   0.1900625,
        0.5669691,  0.59674937, 0.7255512,   0.5647045,  -0.07922941, -0.9469313,

        0.45687163, 0.18743224, 0.36763424,  0.37997854, 0.22606248,  0.6201,
        -0.2965762, -1.4270992, -0.76784426, -2.0531905, 0.22606248,  0.6201,

        0.7157706,  0.7950966,  0.7714553,   0.78296447, 0.80618614,  0.75798905,
        0.7157706,  0.7950966,  0.7714553,   0.78296447, 0.80618614,  0.75798905,

        0.63365304, 0.28596374, 0.68311983,  0.19171247, 0.5141697,   0.31719163,
        0.63365304, 0.28596374, 0.68311983,  0.19171247, 0.5141697,   0.31719163};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(roialign_test_half_pixel)
{
    auto p = create_program("half_pixel");
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        0.36866605, 0.46392143,  0.6987579,  0.75855565, 0.4708544,  0.43184322,
        0.36866605, 0.46392143,  0.6987579,  0.75855565, 0.4708544,  0.43184322,

        0.46302575, 0.4106747,   0.45164075, 0.3248054,  0.29401278, 0.47447777,
        0.46302575, -0.03123695, 0.45164075, -4.494535,  0.29401278, -0.20089805,

        0.5697,     0.5697,      0.5697,     0.5697,     0.5697,     0.5697,
        0.5697,     0.5697,      0.5697,     0.5697,     0.5697,     0.5697,

        0.3137135,  0.48813424,  0.3946669,  0.48890662, 0.4756204,  0.48967898,
        0.3137135,  0.48813424,  0.3946669,  0.48890662, 0.4756204,  0.48967898};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(roialign_test_half_pixel_max)
{
    // TODO: max pooling mode currently fails testing
    auto p = create_program("half_pixel", migraphx::op::pooling_mode::max, 0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        0.4062,     0.335475,   0.73333544, 0.6809158,  0.5071119,  0.3496595,

        0.4062,     0.335475,   0.73333544, 0.6809158,  0.5071119,  0.3496595,

        0.4511997,  0.31101292, 0.37753808, 0.24310073, 0.4388,     0.4627349,

        0.4511997,  0.31101292, 0.37753808, 0.11221313, 0.4388,     0.4627349,

        0.5697,     0.5697,     0.5697,     0.5697,     0.5697,     0.5697,

        0.5697,     0.5697,     0.5697,     0.5697,     0.5697,     0.5697,

        0.26071548, 0.4336742,  0.24798042, 0.3766743,  0.35943243, 0.31967437,

        0.26071548, 0.4336742,  0.24798042, 0.3766743,  0.35943243, 0.31967437};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(roialign_test_output_half_max)
{
    // TODO: max pooling mode currently fails testing
    auto p = create_program("output_half_pixel", migraphx::op::pooling_mode::max, 0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        0.40922216, 0.49688435, 0.53047323, 0.6441095,  0.38779172, 0.22646816,

        0.40922216, 0.49688435, 0.53047323, 0.6441095,  0.38779172, 0.22646816,

        0.36691064, 0.21427372, 0.26765385, 0.5285856,  0.24134001, 0.7119,

        0.36691064, 0.21427372, 0.26765385, 0.5285856,  0.24134001, 0.7119,

        0.34957266, 0.6419918,  0.49346158, 0.5196164,  0.6514608,  0.6859901,

        0.34957266, 0.6419918,  0.49346158, 0.5196164,  0.6514608,  0.6859901,

        0.7145173,  0.23443075, 0.8620839,  0.14426728, 0.61358184, 0.2904524,

        0.7145173,  0.23443075, 0.8620839,  0.14426728, 0.61358184, 0.2904524};

    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
