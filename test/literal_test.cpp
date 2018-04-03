
#include <rtg/literal.hpp>
#include <sstream>
#include <string>
#include "test.hpp"


void literal_test()
{
    EXPECT(rtg::literal{1} == rtg::literal{1});
    EXPECT(rtg::literal{1} != rtg::literal{2});
    EXPECT(rtg::literal{} == rtg::literal{});
    EXPECT(rtg::literal{} != rtg::literal{2});

    rtg::literal l1{1};
    rtg::literal l2 = l1;
    EXPECT(l1 == l2);
    EXPECT(l1.at<int>(0) == 1);
    EXPECT(!l1.empty());
    EXPECT(!l2.empty());

    rtg::literal l3{};
    rtg::literal l4{};
    EXPECT(l3 == l4);
    EXPECT(l3.empty());
    EXPECT(l4.empty());   
}

void literal_os1()
{
    rtg::literal l{1};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str() == "1");
}

void literal_os2()
{
    rtg::literal l{};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str() == "");
}

void literal_os3()
{
    rtg::shape s{rtg::shape::int_type, {3}};
    rtg::literal l{s, {1, 2, 3}};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str() == "1, 2, 3");
}

int main() {
    literal_test();
    literal_os1();
    literal_os2();

}

