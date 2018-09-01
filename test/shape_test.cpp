
#include <migraph/shape.hpp>
#include <array>
#include <algorithm>
#include <numeric>
#include "test.hpp"

void test_shape_default()
{
    migraph::shape s{};
    EXPECT(s.elements() == 0);
    EXPECT(s.bytes() == 0);
}

void test_shape_assign()
{
    migraph::shape s1{migraph::shape::float_type, {100, 32, 8, 8}};
    migraph::shape s2 = s1; // NOLINT
    EXPECT(s1 == s2);
    EXPECT(!(s1 != s2));
}

void test_shape_packed_default()
{
    migraph::shape s{migraph::shape::float_type, {2, 2}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

void test_shape_packed()
{
    migraph::shape s{migraph::shape::float_type, {2, 2}, {2, 1}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
}

void test_shape_transposed()
{
    migraph::shape s{migraph::shape::float_type, {2, 2}, {1, 2}};
    EXPECT(not s.standard());
    EXPECT(s.packed());
    EXPECT(s.transposed());
    EXPECT(not s.broadcasted());
}

void test_shape_broadcasted()
{
    migraph::shape s{migraph::shape::float_type, {2, 2}, {1, 0}};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(s.broadcasted());
}

void test_shape_default_copy()
{
    migraph::shape s1{};
    migraph::shape s2{};
    EXPECT(s1 == s2);
    EXPECT(!(s1 != s2));
}

void test_shape4()
{
    migraph::shape s{migraph::shape::float_type, {100, 32, 8, 8}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.type() == migraph::shape::float_type);
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

void test_shape42()
{
    migraph::shape s{migraph::shape::float_type, {100, 32, 8, 8}, {2048, 64, 8, 1}};
    EXPECT(s.standard());
    EXPECT(s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.type() == migraph::shape::float_type);
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

void test_shape4_transposed()
{
    migraph::shape s{migraph::shape::float_type, {32, 100, 8, 8}, {64, 2048, 8, 1}};
    EXPECT(s.transposed());
    EXPECT(s.packed());
    EXPECT(not s.standard());
    EXPECT(not s.broadcasted());
    EXPECT(s.type() == migraph::shape::float_type);
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

void test_shape4_nonpacked()
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

    migraph::shape s{migraph::shape::float_type, lens, strides};
    EXPECT(not s.standard());
    EXPECT(not s.packed());
    EXPECT(not s.transposed());
    EXPECT(not s.broadcasted());
    EXPECT(s.type() == migraph::shape::float_type);
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

int main()
{
    test_shape_default();
    test_shape_assign();
    test_shape_packed_default();
    test_shape_packed();
    test_shape_transposed();
    test_shape_broadcasted();
    test_shape_default_copy();
    test_shape4();
    test_shape42();
    test_shape4_transposed();
    test_shape4_nonpacked();
}
