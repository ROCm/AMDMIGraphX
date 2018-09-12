
#include <migraph/operation.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

struct simple_operation
{
    int data = 1;
    std::string name() const { return "simple"; }
    migraph::shape compute_shape(const std::vector<migraph::shape>&) const
    {
        MIGRAPH_THROW("not computable");
    }
    migraph::argument
    compute(migraph::context&, const migraph::shape&, const std::vector<migraph::argument>&) const
    {
        MIGRAPH_THROW("not computable");
    }
    friend std::ostream& operator<<(std::ostream& os, const simple_operation& op)
    {
        os << "[" << op.name() << "]";
        return os;
    }
};

struct simple_operation_no_print
{
    std::string name() const { return "simple"; }
    migraph::shape compute_shape(const std::vector<migraph::shape>&) const
    {
        MIGRAPH_THROW("not computable");
    }
    migraph::argument
    compute(migraph::context&, const migraph::shape&, const std::vector<migraph::argument>&) const
    {
        MIGRAPH_THROW("not computable");
    }
};

void operation_copy_test()
{
    simple_operation s{};
    migraph::operation op1 = s;   // NOLINT
    migraph::operation op2 = op1; // NOLINT
    // cppcheck-suppress duplicateExpression
    EXPECT(s.name() == op1.name());
    // cppcheck-suppress duplicateExpression
    EXPECT(op2.name() == op1.name());
}

struct not_operation
{
};

void operation_any_cast()
{
    migraph::operation op1 = simple_operation{};
    EXPECT(migraph::any_cast<simple_operation>(op1).data == 1);
    EXPECT(migraph::any_cast<not_operation*>(&op1) == nullptr);
    EXPECT(test::throws([&] { migraph::any_cast<not_operation&>(op1); }));
    migraph::operation op2 = simple_operation{2};
    EXPECT(migraph::any_cast<simple_operation>(op2).data == 2);
    EXPECT(migraph::any_cast<not_operation*>(&op2) == nullptr);
}

void operation_print()
{
    migraph::operation op = simple_operation{};
    std::stringstream ss;
    ss << op;
    std::string s = ss.str();
    EXPECT(s == "[simple]");
}

void operation_default_print()
{
    migraph::operation op = simple_operation_no_print{};
    std::stringstream ss;
    ss << op;
    std::string s = ss.str();
    EXPECT(s == "simple");
}

int main()
{
    operation_copy_test();
    operation_any_cast();
    operation_print();
    operation_default_print();
}
