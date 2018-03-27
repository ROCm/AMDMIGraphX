
#include <rtg/literal.hpp>
#include "test.hpp"

int main() {
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

