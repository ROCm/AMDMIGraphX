
#include <rtg/operation.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

struct simple_operation
{
    std::string name() const { return "simple"; }
    rtg::shape compute_shape(std::vector<rtg::shape>) const { RTG_THROW("not computable"); }
    rtg::argument compute(std::vector<rtg::argument>) const { RTG_THROW("not computable"); }
};

void operation_copy_test()
{
    simple_operation s{};
    rtg::operation op1 = s; // NOLINT
    rtg::operation op2 = op1; // NOLINT
    EXPECT(s.name() == op1.name());
    EXPECT(op2.name() == op1.name());
}

int main()
{
    operation_copy_test();
}
