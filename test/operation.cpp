
#include <migraphx/operation.hpp>
#include <migraphx/context.hpp>
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
        MIGRAPHX_THROW("not computable");
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        MIGRAPHX_THROW("not computable");
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
        MIGRAPHX_THROW("not computable");
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        MIGRAPHX_THROW("not computable");
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

TEST_CASE(operation_copy_assign_test)
{
    simple_operation s{};
    migraphx::operation op;
    op = s;
    // cppcheck-suppress duplicateExpression
    EXPECT(s == op);
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

struct final_operation
{
    std::string name() const { return "final"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
    void
    finalize(migraphx::context&, const migraphx::shape&, const std::vector<migraphx::shape>&) const
    {
    }
};

struct final_operation_throw
{
    std::string name() const { return "final"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
    [[gnu::noreturn]] void
    finalize(migraphx::context&, const migraphx::shape&, const std::vector<migraphx::shape>&) const
    {
        MIGRAPHX_THROW("finalize");
    }
};

TEST_CASE(check_has_finalize_simple)
{
    migraphx::operation op = simple_operation{};
    EXPECT(not migraphx::has_finalize(op));
}

TEST_CASE(check_has_finalize)
{
    migraphx::operation op = final_operation{};
    EXPECT(migraphx::has_finalize(op));
}

TEST_CASE(check_run_finalize)
{
    migraphx::operation op = final_operation{};
    migraphx::context ctx{};
    op.finalize(ctx, {}, {});
}

TEST_CASE(check_run_finalize_simple)
{
    migraphx::operation op = simple_operation{};
    migraphx::context ctx{};
    op.finalize(ctx, {}, {});
}

TEST_CASE(check_run_finalize_throw)
{
    migraphx::operation op = final_operation_throw{};
    migraphx::context ctx{};
    EXPECT(test::throws([&] { op.finalize(ctx, {}, {}); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
