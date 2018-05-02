
#include <rtg/shape.hpp>
#include <array>
#include <algorithm>
#include <numeric>
#include "test.hpp"

void test_shape_assign()
{
    rtg::shape s1{rtg::shape::float_type, {100, 32, 8, 8}};
    rtg::shape s2 = s1; // NOLINT
    EXPECT(s1 == s2);
    EXPECT(!(s1 != s2));
}

void test_shape_default()
{
    rtg::shape s1{};
    rtg::shape s2{};
    EXPECT(s1 == s2);
    EXPECT(!(s1 != s2));
}

void test_shape4()
{
    rtg::shape s{rtg::shape::float_type, {100, 32, 8, 8}};
    EXPECT(s.packed());
    EXPECT(s.type() == rtg::shape::float_type);
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

    rtg::shape s{rtg::shape::float_type, lens, strides};
    EXPECT(!s.packed());
    EXPECT(s.type() == rtg::shape::float_type);
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
    // TODO: Fix these tests
    // EXPECT(s.index({0, 0, 1, 0}) == s.index(8));
    // EXPECT(s.index({0, 1, 0, 0}) == s.index(8 * 8));
    // EXPECT(s.index({1, 0, 0, 0}) == s.index(8 * 8 * 32));
    // EXPECT(s.index(s.elements() - 1) == 469273);
}

int main()
{
    test_shape_assign();
    test_shape_default();
    test_shape4();
    test_shape4_nonpacked();
}
