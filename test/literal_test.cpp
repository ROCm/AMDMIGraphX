
#include <migraph/literal.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

void literal_test()
{
    EXPECT(migraph::literal{1} == migraph::literal{1});
    EXPECT(migraph::literal{1} != migraph::literal{2});
    EXPECT(migraph::literal{} == migraph::literal{});
    EXPECT(migraph::literal{} != migraph::literal{2});

    migraph::literal l1{1};
    migraph::literal l2 = l1; // NOLINT
    EXPECT(l1 == l2);
    EXPECT(l1.at<int>(0) == 1);
    EXPECT(!l1.empty());
    EXPECT(!l2.empty());

    migraph::literal l3{};
    migraph::literal l4{};
    EXPECT(l3 == l4);
    EXPECT(l3.empty());
    EXPECT(l4.empty());
}

void literal_os1()
{
    migraph::literal l{1};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str() == "1");
}

void literal_os2()
{
    migraph::literal l{};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str().empty());
}

void literal_os3()
{
    migraph::shape s{migraph::shape::int64_type, {3}};
    migraph::literal l{s, {1, 2, 3}};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str() == "1, 2, 3");
}

int main()
{
    literal_test();
    literal_os1();
    literal_os2();
}
