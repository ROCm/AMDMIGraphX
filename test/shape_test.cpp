/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/shape.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/stringutils.hpp>
#include <array>
#include <algorithm>
#include <numeric>
#include "test.hpp"

TEST_CASE(test_shape_default)
{
    migraphx::shape s{};
    EXPECT(s.elements() == 0);
    EXPECT(s.bytes() == 0);
}
TEST_CASE(test_shape_assign)
{
    migraphx::shape s1{migraphx::shape::float_type, {100, 32, 8, 8}};
    migraphx::shape s2 = s1; // NOLINT
    EXPECT(s1 == s2);
    EXPECT(not(s1 != s2));
}

TEST_CASE(test_shape_packed_default)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_standard)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}, {6, 3, 1}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_min_max_opt)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}, {6, 3, 1}};
    EXPECT(s.min_lens() == s.lens());
    EXPECT(s.max_lens() == s.lens());
    EXPECT(s.opt_lens() == s.lens());
}

TEST_CASE(test_shape_dynamic_fixed)
{
    migraphx::shape s{migraphx::shape::float_type, {{2, 2, 0}, {2, 2, 0}, {3, 3, 0}}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.dynamic());
    EXPECT(s.dyn_dims().size() == 3);
    EXPECT(s.dyn_dims().at(0).is_fixed());
    EXPECT(not s.dyn_dims().at(0).has_optimal());
    EXPECT(s.min_lens() == std::vector<std::size_t>{2, 2, 3});
    EXPECT(s.max_lens() == std::vector<std::size_t>{2, 2, 3});
    EXPECT(s.opt_lens() == std::vector<std::size_t>{0, 0, 0});
    EXPECT(s.bytes() == 2 * 2 * 3 * sizeof(float));
}

TEST_CASE(test_shape_dynamic_not_fixed)
{
    using migraphx::shape;
    std::vector<shape::dynamic_dimension> dims = {};
    dims.push_back(shape::dynamic_dimension{2, 5, 2});
    dims.push_back(shape::dynamic_dimension{2, 8, 0});
    migraphx::shape s{migraphx::shape::float_type, dims};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.dynamic());
    EXPECT(s.dyn_dims().size() == 2);
    EXPECT(not s.dyn_dims().at(0).is_fixed());
    EXPECT(s.dyn_dims().at(0).has_optimal());
    EXPECT(s.min_lens() == std::vector<std::size_t>{2, 2});
    EXPECT(s.max_lens() == std::vector<std::size_t>{5, 8});
    EXPECT(s.opt_lens() == std::vector<std::size_t>{2, 0});
    EXPECT(s.bytes() == 5 * 8 * sizeof(float));
}

TEST_CASE(test_shape_dynamic_compares)
{
    using migraphx::shape;
    auto a = shape::dynamic_dimension{2, 5, 2};
    auto b = a;
    auto c = shape::dynamic_dimension{2, 5, 2};
    auto d = shape::dynamic_dimension{3, 8, 4};
    EXPECT(a == b);
    EXPECT(a == c);
    EXPECT(a != d);

    migraphx::shape s0{shape::float_type, {a, d}};
    migraphx::shape s1 = s0;
    migraphx::shape s2{shape::float_type, {a, d}};
    migraphx::shape s3{shape::int32_type, {a}};
    EXPECT(s0 == s1);
    EXPECT(s0 == s2);
    EXPECT(s0 != s3);

    std::stringstream ss0;
    std::stringstream ss1;
    std::stringstream ss3;
    ss0 << s0;
    ss1 << s1;
    ss3 << s3;
    EXPECT(ss0.str() == ss1.str());
    EXPECT(ss0.str() != ss3.str());
}

TEST_CASE(test_shape_dynamic_errors)
{
    using migraphx::shape;
    std::vector<shape::dynamic_dimension> dims = {};
    dims.push_back(shape::dynamic_dimension{2, 5, 2});
    dims.push_back(shape::dynamic_dimension{2, 8, 0});
    migraphx::shape s{shape::float_type, dims};
    EXPECT(test::throws([&] { s.elements(); }));
    EXPECT(test::throws([&] { s.index({0, 1}); }));
    EXPECT(test::throws([&] { s.index(1); }));
    EXPECT(test::throws([&] { s.index(std::vector<std::size_t>{0, 1}); }));
    EXPECT(test::throws([&] { s.with_lens({3, 5}); }));
    EXPECT(test::throws([&] { s.with_lens(shape::float_type, {3, 5}); }));
}

TEST_CASE(test_shape_dynamic_serialize)
{
    using migraphx::shape;
    std::vector<shape::dynamic_dimension> dims1 = {};
    dims1.push_back(shape::dynamic_dimension{2, 5, 2});
    dims1.push_back(shape::dynamic_dimension{2, 8, 0});
    migraphx::shape s1{shape::float_type, dims1};
    auto v1 = migraphx::to_value(s1);

    std::vector<shape::dynamic_dimension> dims2 = {};
    dims2.push_back(shape::dynamic_dimension{2, 5, 2});
    migraphx::shape s2{shape::uint64_type, dims2};
    auto v2 = migraphx::to_value(s2);
    EXPECT(v1 != v2);

    auto s3 = migraphx::from_value<shape>(v1);
    EXPECT(s3 == s1);
    auto s4 = migraphx::from_value<shape>(v2);
    EXPECT(s4 == s2);
    EXPECT(s3 != s4);
}

TEST_CASE(test_shape_packed)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}, {2, 1}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_non_packed_single_dim)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 64, 35, 35}, {156800, 1225, 35, 1}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_transposed1)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}, {1, 2}};
    EXPECT(not s.standard());
    EXPECT(s.packed());
    EXPECT(s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_transposed2)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 1, 1, 1, 2}, {2, 2, 2, 2, 1}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_overlap)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}, {6, 3, 2}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_overlap2)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}, {6, 2, 1}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_overlap3)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}, {4, 2, 1}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

TEST_CASE(test_shape_scalar1)
{
    migraphx::shape s{migraphx::shape::float_type};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_scalar2)
{
    migraphx::shape s{migraphx::shape::float_type, {1}, {0}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_scalar_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 2, 3, 3}, {0, 0, 0, 0}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}, {1, 0}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted2)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 2}, {0, 1}};
    EXPECT(not s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted3)
{
    migraphx::shape s{migraphx::shape::float_type, {3, 2}, {0, 1}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted4)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}, {6, 0, 1}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_broadcasted5)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}, {1, 0, 6}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_step_broadcasted)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}, {0, 3}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

TEST_CASE(test_shape_default_copy)
{
    migraphx::shape s1{};
    migraphx::shape s2{};
    EXPECT(s1 == s2);
    EXPECT(not(s1 != s2));
}

TEST_CASE(test_shape_normalize_standard1)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 3}, {6, 3, 1}};
    EXPECT(s.standard());
    auto n = s.normalize_standard();
    EXPECT(n == s);
}

TEST_CASE(test_shape_normalize_standard2)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 64, 35, 35}, {156800, 1225, 35, 1}};
    EXPECT(s.standard());
    auto n = s.normalize_standard();
    EXPECT(n.standard());
    EXPECT(n != s);
    EXPECT(n.lens() == s.lens());
    EXPECT(n.type() == s.type());
}

TEST_CASE(test_shape_normalize_standard3)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}, {1, 2}};
    EXPECT(not s.standard());
    auto n = s.normalize_standard();
    EXPECT(n == s);
}

TEST_CASE(test_shape_normalize_scalar1)
{
    migraphx::shape s{migraphx::shape::float_type};
    EXPECT(s.standard());
    EXPECT(s.scalar());
    auto n = s.normalize_standard();
    EXPECT(n != s);
    EXPECT(n.standard());
    EXPECT(not n.scalar());
}

TEST_CASE(test_shape_normalize_scalar2)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}, {0, 0}};
    EXPECT(not s.standard());
    EXPECT(s.scalar());
    auto n = s.normalize_standard();
    EXPECT(n == s);
}

TEST_CASE(test_shape4)
{
    migraphx::shape s{migraphx::shape::float_type, {100, 32, 8, 8}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.type() == migraphx::shape::float_type);
    EXPECT(s.lens()[0] == 100);
    EXPECT(s.lens()[1] == 32);
    EXPECT(s.lens()[2] == 8);
    EXPECT(s.lens()[3] == 8);
    EXPECT(s.strides()[0] == s.lens()[1] * s.strides()[1]);
    EXPECT(s.strides()[1] == s.lens()[2] * s.strides()[2]);
    EXPECT(s.strides()[2] == s.lens()[3] * s.strides()[3]);
    EXPECT(s.strides()[3] == 1);
    EXPECT(s.elements() == 100 * 32 * 8 * 8);
    EXPECT(s.bytes() == 100 * 32 * 8 * 8 * sizeof(float));
    EXPECT(s.index({0, 0, 0, 0}) == 0);
    EXPECT(s.index({0, 0, 0, 1}) == 1);
    EXPECT(s.index({0, 0, 0, 0}) == s.index(0));
    EXPECT(s.index({0, 0, 0, 1}) == s.index(1));
    EXPECT(s.index({0, 0, 1, 0}) == s.index(8));
    EXPECT(s.index({0, 1, 0, 0}) == s.index(8 * 8));
    EXPECT(s.index({1, 0, 0, 0}) == s.index(8 * 8 * 32));
    EXPECT(s.index(0) == 0);
    EXPECT(s.index(1) == 1);
    EXPECT(s.index(8) == 8);
    EXPECT(s.index(8 * 8) == 8 * 8);
    EXPECT(s.index(8 * 8 * 32) == 8 * 8 * 32);
    EXPECT(s.index(s.elements() - 1) == s.elements() - 1);
}

TEST_CASE(test_shape42)
{
    migraphx::shape s{migraphx::shape::float_type, {100, 32, 8, 8}, {2048, 64, 8, 1}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.type() == migraphx::shape::float_type);
    EXPECT(s.lens()[0] == 100);
    EXPECT(s.lens()[1] == 32);
    EXPECT(s.lens()[2] == 8);
    EXPECT(s.lens()[3] == 8);
    EXPECT(s.strides()[0] == s.lens()[1] * s.strides()[1]);
    EXPECT(s.strides()[1] == s.lens()[2] * s.strides()[2]);
    EXPECT(s.strides()[2] == s.lens()[3] * s.strides()[3]);
    EXPECT(s.strides()[3] == 1);
    EXPECT(s.elements() == 100 * 32 * 8 * 8);
    EXPECT(s.bytes() == 100 * 32 * 8 * 8 * sizeof(float));
    EXPECT(s.index({0, 0, 0, 0}) == 0);
    EXPECT(s.index({0, 0, 0, 1}) == 1);
    EXPECT(s.index({0, 0, 0, 0}) == s.index(0));
    EXPECT(s.index({0, 0, 0, 1}) == s.index(1));
    EXPECT(s.index({0, 0, 1, 0}) == s.index(8));
    EXPECT(s.index({0, 1, 0, 0}) == s.index(8 * 8));
    EXPECT(s.index({1, 0, 0, 0}) == s.index(8 * 8 * 32));
    EXPECT(s.index(0) == 0);
    EXPECT(s.index(1) == 1);
    EXPECT(s.index(8) == 8);
    EXPECT(s.index(8 * 8) == 8 * 8);
    EXPECT(s.index(8 * 8 * 32) == 8 * 8 * 32);
    EXPECT(s.index(s.elements() - 1) == s.elements() - 1);
}

TEST_CASE(test_shape4_transposed)
{
    migraphx::shape s{migraphx::shape::float_type, {32, 100, 8, 8}, {64, 2048, 8, 1}};
    EXPECT(s.transposed());
    EXPECT(s.packed());
    EXPECT(not s.standard());
    EXPECT(not s.broadcasted());
    EXPECT(s.type() == migraphx::shape::float_type);
    EXPECT(s.lens()[0] == 32);
    EXPECT(s.lens()[1] == 100);
    EXPECT(s.lens()[2] == 8);
    EXPECT(s.lens()[3] == 8);
    EXPECT(s.strides()[0] == 64);
    EXPECT(s.strides()[1] == 2048);
    EXPECT(s.strides()[2] == 8);
    EXPECT(s.strides()[3] == 1);
    EXPECT(s.elements() == 100 * 32 * 8 * 8);
    EXPECT(s.bytes() == 100 * 32 * 8 * 8 * sizeof(float));
    EXPECT(s.index({0, 0, 0, 0}) == 0);
    EXPECT(s.index({0, 0, 0, 1}) == 1);
    EXPECT(s.index({0, 0, 0, 0}) == s.index(0));
    EXPECT(s.index({0, 0, 0, 1}) == s.index(1));
    EXPECT(s.index({0, 0, 1, 0}) == s.index(8));
    EXPECT(s.index({0, 1, 0, 0}) == s.index(8 * 8));
    EXPECT(s.index({1, 0, 0, 0}) == s.index(8 * 8 * 100));
    EXPECT(s.index(0) == 0);
    EXPECT(s.index(1) == 1);
    EXPECT(s.index(8) == 8);
    EXPECT(s.index(8 * 8) == 2048);
    EXPECT(s.index(8 * 8 * 100) == 64);
    EXPECT(s.index(s.elements() - 1) == s.elements() - 1);
}

TEST_CASE(test_shape4_nonpacked)
{
    std::vector<std::size_t> lens       = {100, 32, 8, 8};
    std::array<std::size_t, 4> offsets  = {{5, 10, 0, 6}};
    std::array<std::size_t, 4> adj_lens = {{0, 0, 0, 0}};

    std::transform(
        lens.begin(), lens.end(), offsets.begin(), adj_lens.begin(), std::plus<size_t>());
    // adj_lens should be: { 105, 42, 8, 14 }
    std::vector<std::size_t> strides(4);
    strides.back() = 1;
    std::partial_sum(adj_lens.rbegin(),
                     adj_lens.rend() - 1,
                     strides.rbegin() + 1,
                     std::multiplies<std::size_t>());

    migraphx::shape s{migraphx::shape::float_type, lens, strides};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.type() == migraphx::shape::float_type);
    EXPECT(s.lens()[0] == 100);
    EXPECT(s.lens()[1] == 32);
    EXPECT(s.lens()[2] == 8);
    EXPECT(s.lens()[3] == 8);
    EXPECT(s.strides()[0] == 4704);
    EXPECT(s.strides()[1] == 112);
    EXPECT(s.strides()[2] == 14);
    EXPECT(s.strides()[3] == 1);
    EXPECT(s.elements() == 100 * 32 * 8 * 8);
    EXPECT(s.bytes() == sizeof(float) * 469274);

    EXPECT(s.index(0) == 0);
    EXPECT(s.index(1) == 1);
    EXPECT(s.index({0, 0, 0, 0}) == 0);
    EXPECT(s.index({0, 0, 0, 1}) == s.index(1));
    EXPECT(s.index({0, 0, 1, 0}) == s.index(8));
    EXPECT(s.index({0, 1, 0, 0}) == s.index(8 * 8));
    EXPECT(s.index({1, 0, 0, 0}) == s.index(8 * 8 * 32));
    EXPECT(s.index(s.elements() - 1) == 469273);
}

TEST_CASE(test_serialize)
{
    migraphx::shape s1{migraphx::shape::float_type, {100, 32, 8, 8}};
    auto v1 = migraphx::to_value(s1);
    migraphx::shape s2{migraphx::shape::uint64_type, {2, 2}};
    auto v2 = migraphx::to_value(s2);
    EXPECT(v1 != v2);

    auto s3 = migraphx::from_value<migraphx::shape>(v1);
    EXPECT(s3 == s1);
    auto s4 = migraphx::from_value<migraphx::shape>(v2);
    EXPECT(s4 == s2);
    EXPECT(s3 != s4);
}

TEST_CASE(tuple)
{
    migraphx::shape s{{migraphx::shape{migraphx::shape::float_type},
                       migraphx::shape{migraphx::shape::int8_type}}};
    EXPECT(s.type() == migraphx::shape::tuple_type);
    EXPECT(s.bytes() == 4 + 1);
    EXPECT(s.type_size() == 0);
    EXPECT(s.type_string() == "tuple_type");
    EXPECT(s.lens().empty());
    EXPECT(s.strides().empty());
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.broadcasted());
    EXPECT(not s.transposed());
    EXPECT(not s.scalar());
    EXPECT(s.sub_shapes().size() == 2);
    EXPECT(s.sub_shapes()[0].type() == migraphx::shape::float_type);
    EXPECT(s.sub_shapes()[0].elements() == 1);
    EXPECT(s.sub_shapes()[1].type() == migraphx::shape::int8_type);
    EXPECT(s.sub_shapes()[1].elements() == 1);
    EXPECT(test::throws([&] { s.visit_type([](auto) {}); }));
}

TEST_CASE(tuple_copy)
{
    migraphx::shape s1{{migraphx::shape{migraphx::shape::float_type},
                        migraphx::shape{migraphx::shape::int8_type}}};
    migraphx::shape s2{{migraphx::shape{migraphx::shape::float_type},
                        migraphx::shape{migraphx::shape::int8_type}}};
    EXPECT(s1 == s2);
    auto s3 = s1;
    EXPECT(s3 == s1);
    EXPECT(s3 == s2);
    migraphx::shape s4{{migraphx::shape{migraphx::shape::int8_type},
                        migraphx::shape{migraphx::shape::float_type}}};
    EXPECT(s4 != s1);
    EXPECT(s4 != s2);
    EXPECT(s4 != s3);
}

TEST_CASE(tuple_print)
{
    migraphx::shape s{{migraphx::shape{migraphx::shape::float_type},
                       migraphx::shape{migraphx::shape::int8_type}}};
    std::string x = migraphx::to_string(s);
    EXPECT(x.front() == '[');
    EXPECT(x.back() == ']');
    EXPECT(migraphx::contains(x, "float"));
    EXPECT(migraphx::contains(x, "int8"));
}

TEST_CASE(tuple_serialize)
{
    migraphx::shape s1{{migraphx::shape{migraphx::shape::float_type},
                        migraphx::shape{migraphx::shape::int8_type}}};
    migraphx::shape s2{{migraphx::shape{migraphx::shape::int8_type},
                        migraphx::shape{migraphx::shape::float_type}}};
    auto v1 = migraphx::to_value(s1);
    auto v2 = migraphx::to_value(s2);
    EXPECT(v1 != v2);

    auto s3 = migraphx::from_value<migraphx::shape>(v1);
    EXPECT(s3 == s1);
    auto s4 = migraphx::from_value<migraphx::shape>(v2);
    EXPECT(s4 == s2);
    EXPECT(s3 != s4);
}

TEST_CASE(test_with_lens1)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 2}, {1, 2}};
    auto s2 = s1.with_lens({4, 3});
    EXPECT(s2.transposed());
    migraphx::shape s3{migraphx::shape::float_type, {4, 3}, {1, 4}};
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens2)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 2}, {2, 1}};
    auto s2 = s1.with_lens({3, 4});
    EXPECT(s2.standard());
    migraphx::shape s3{migraphx::shape::float_type, {3, 4}};
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous1)
{
    migraphx::shape s1{migraphx::shape::float_type, {64, 1, 24, 24}};
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(not s2.transposed());
    migraphx::shape s3{migraphx::shape::float_type, {64, 3, 24, 24}};
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous2)
{
    auto s1 = migraphx::reorder_shape({migraphx::shape::float_type, {64, 24, 24, 1}}, {0, 3, 1, 2});
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(s2.transposed());
    migraphx::shape s3 =
        migraphx::reorder_shape({migraphx::shape::float_type, {64, 24, 24, 3}}, {0, 3, 1, 2});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous3)
{
    migraphx::shape s1{migraphx::shape::float_type, {64, 3, 1, 1}};
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(not s2.transposed());
    migraphx::shape s3{migraphx::shape::float_type, {64, 3, 24, 24}};
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous4)
{
    auto s1 = migraphx::reorder_shape({migraphx::shape::float_type, {64, 1, 1, 3}}, {0, 3, 1, 2});
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(s2.transposed());
    migraphx::shape s3 =
        migraphx::reorder_shape({migraphx::shape::float_type, {64, 24, 24, 3}}, {0, 3, 1, 2});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous5)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 5, 24, 24}};
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(not s2.transposed());
    migraphx::shape s3{migraphx::shape::float_type, {64, 3, 24, 24}};
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous6)
{
    auto s1 = migraphx::reorder_shape({migraphx::shape::float_type, {1, 24, 24, 5}}, {0, 3, 1, 2});
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(s2.transposed());
    migraphx::shape s3 =
        migraphx::reorder_shape({migraphx::shape::float_type, {64, 24, 24, 3}}, {0, 3, 1, 2});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous7)
{
    auto s1 = migraphx::reorder_shape({migraphx::shape::float_type, {1, 1, 1, 3}}, {0, 3, 1, 2});
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(s2.transposed());
    migraphx::shape s3 =
        migraphx::reorder_shape({migraphx::shape::float_type, {64, 24, 24, 3}}, {0, 3, 1, 2});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous8)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 1, 24, 24}};
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(not s2.transposed());
    migraphx::shape s3{migraphx::shape::float_type, {64, 3, 24, 24}};
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous9)
{
    auto s1 = migraphx::reorder_shape({migraphx::shape::float_type, {1, 24, 24, 1}}, {0, 3, 1, 2});
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(s2.transposed());
    migraphx::shape s3 =
        migraphx::reorder_shape({migraphx::shape::float_type, {64, 24, 24, 3}}, {0, 3, 1, 2});
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous10)
{
    migraphx::shape s1{migraphx::shape::float_type, {3, 2, 4, 1}};
    auto s2 = s1.with_lens({3, 2, 4, 1});
    EXPECT(not s2.transposed());
    migraphx::shape s3{migraphx::shape::float_type, {3, 2, 4, 1}};
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous11)
{
    migraphx::shape s1{migraphx::shape::float_type, {64, 1, 1, 1}};
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(s1.standard());
    EXPECT(s2.standard());
    migraphx::shape s3{migraphx::shape::float_type, {64, 3, 24, 24}};
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous12)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 64, 1, 1}};
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(s1.standard());
    EXPECT(s2.standard());
    migraphx::shape s3{migraphx::shape::float_type, {64, 3, 24, 24}};
    EXPECT(s2 == s3);
}

TEST_CASE(test_with_lens_ambigous13)
{
    auto s1 = migraphx::reorder_shape({migraphx::shape::float_type, {1, 1, 1, 3}}, {0, 3, 1, 2});
    auto s2 = s1.with_lens({64, 3, 24, 24});
    EXPECT(s2.transposed());
    migraphx::shape s3 =
        migraphx::reorder_shape({migraphx::shape::float_type, {64, 24, 24, 3}}, {0, 3, 1, 2});
    EXPECT(s2 == s3);
}

TEST_CASE(cpp_type_name)
{
    EXPECT(migraphx::shape::cpp_type(migraphx::shape::int8_type) == "int8_t");
    EXPECT(migraphx::shape::cpp_type(migraphx::shape::float_type) == "float");
    EXPECT(migraphx::shape::cpp_type(migraphx::shape::half_type) == "half");
    EXPECT(test::throws([&] { migraphx::shape::cpp_type(migraphx::shape::tuple_type); }));
}

TEST_CASE(test_with_type)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 2}, {1, 0}};
    EXPECT(s.type() == migraphx::shape::float_type);
    auto new_s = s.with_type(migraphx::shape::half_type);
    EXPECT(s.type() == migraphx::shape::float_type);
    EXPECT(s.type() != new_s.type());
    EXPECT(s.lens() == new_s.lens());
    EXPECT(s.strides() == new_s.strides());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
