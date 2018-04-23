
#include <rtg/shape.hpp>
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
}

int main()
{
    test_shape_assign();
    test_shape_default();
    test_shape4();
}
