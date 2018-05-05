
#include <rtg/operation.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

struct simple_operation
{
    int data = 1;
    std::string name() const { return "simple"; }
    rtg::shape compute_shape(std::vector<rtg::shape>) const { RTG_THROW("not computable"); }
    rtg::argument compute(std::vector<rtg::argument>) const { RTG_THROW("not computable"); }
};

void operation_copy_test()
{
    simple_operation s{};
    rtg::operation op1 = s;   // NOLINT
    rtg::operation op2 = op1; // NOLINT
    EXPECT(s.name() == op1.name());
    EXPECT(op2.name() == op1.name());
}

struct not_operation
{
};

void operation_any_cast()
{
    rtg::operation op1 = simple_operation{};
    EXPECT(rtg::any_cast<simple_operation>(op1).data == 1);
    EXPECT(rtg::any_cast<not_operation*>(&op1) == nullptr);
    EXPECT(test::throws([&] { rtg::any_cast<not_operation&>(op1); }));
    rtg::operation op2 = simple_operation{2};
    EXPECT(rtg::any_cast<simple_operation>(op2).data == 2);
    EXPECT(rtg::any_cast<not_operation*>(&op2) == nullptr);
}

int main()
{
    operation_copy_test();
    operation_any_cast();
}
