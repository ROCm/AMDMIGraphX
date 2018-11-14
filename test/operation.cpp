
#include <migraphx/operation.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

struct simple_operation
{
    template <class T, class F>
    static auto reflect(T& x, F f)
    {
        return migraphx::pack(f(x.data, "data"));
    }
    int data = 1;
    std::string name() const { return "simple"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        MIGRAPH_THROW("not computable");
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        MIGRAPH_THROW("not computable");
    }
    friend std::ostream& operator<<(std::ostream& os, const simple_operation& op)
    {
        os << op.name() << "[" << op.data << "]";
        return os;
    }
};

struct simple_operation_no_print
{
    std::string name() const { return "simple"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        MIGRAPH_THROW("not computable");
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        MIGRAPH_THROW("not computable");
    }
};

TEST_CASE(operation_copy_test)
{
    simple_operation s{};
    migraphx::operation op1 = s;   // NOLINT
    migraphx::operation op2 = op1; // NOLINT
    // cppcheck-suppress duplicateExpression
    EXPECT(s == op1);
    // cppcheck-suppress duplicateExpression
    EXPECT(op2 == op1);
}

TEST_CASE(operation_equal_test)
{
    simple_operation s{};
    migraphx::operation op1 = s;
    s.data                  = 2;
    migraphx::operation op2 = op1; // NOLINT
    migraphx::operation op3 = s;   // NOLINT

    EXPECT(s != op1);
    EXPECT(op2 == op1);
    EXPECT(op3 != op2);
    EXPECT(op3 != op1);
}

struct not_operation
{
};

TEST_CASE(operation_any_cast)
{
    migraphx::operation op1 = simple_operation{};
    EXPECT(migraphx::any_cast<simple_operation>(op1).data == 1);
    EXPECT(migraphx::any_cast<not_operation*>(&op1) == nullptr);
    EXPECT(test::throws([&] { migraphx::any_cast<not_operation&>(op1); }));
    migraphx::operation op2 = simple_operation{2};
    EXPECT(migraphx::any_cast<simple_operation>(op2).data == 2);
    EXPECT(migraphx::any_cast<not_operation*>(&op2) == nullptr);
}

TEST_CASE(operation_print)
{
    migraphx::operation op = simple_operation{};
    std::stringstream ss;
    ss << op;
    std::string s = ss.str();
    EXPECT(s == "simple[1]");
}

TEST_CASE(operation_default_print)
{
    migraphx::operation op = simple_operation_no_print{};
    std::stringstream ss;
    ss << op;
    std::string s = ss.str();
    EXPECT(s == "simple");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
