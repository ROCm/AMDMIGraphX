
#include <migraphx/shape.hpp>
#include <migraphx/serialize.hpp>
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
    EXPECT(!(s1 != s2));
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

TEST_CASE(test_shape_default_copy)
{
    migraphx::shape s1{};
    migraphx::shape s2{};
    EXPECT(s1 == s2);
    EXPECT(!(s1 != s2));
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
