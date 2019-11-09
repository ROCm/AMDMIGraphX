
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/compile_options.hpp>
#include <sstream>
#include "test.hpp"
#include <basic_ops.hpp>

struct id_target
{
    struct context
    {
        void finish() const {}
    };
    migraphx::context ctx = context{};
    std::string name() const { return "id"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&, const migraphx::compile_options&) const { return {}; }
    migraphx::context get_context() const { return ctx; }
};

struct id_ctx_op
{
    std::string name() const { return "id_ctx_op"; }
    migraphx::argument
    compute(id_target::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct id_ctx_final_op
{
    std::string name() const { return "id_ctx_final_op"; }
    migraphx::argument compute(const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    void finalize(id_target::context&, const migraphx::shape&, const std::vector<migraphx::shape>&)
    {
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct reverse_pass
{
    std::string name() const { return "reverse_pass"; }

    void apply(migraphx::program& p) const { std::reverse(p.begin(), p.end()); }
};

struct reverse_target
{
    std::string name() const { return "reverse"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&, const migraphx::compile_options&) const { return {reverse_pass{}}; }
    migraphx::context get_context() const { return {}; }
};

struct invert_pass
{
    std::string name() const { return "invert_pass"; }

    void apply(migraphx::program& p) const
    {
        for(auto ins : migraphx::iterator_for(p))
        {
            if(ins->name() == "sum")
            {
                p.replace_instruction(ins, minus_op{}, ins->inputs());
            }
            else if(ins->name() == "minus")
            {
                p.replace_instruction(ins, sum_op{}, ins->inputs());
            }
        }
    }
};

struct invert_target
{
    std::string name() const { return "invert"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&, const migraphx::compile_options&) const { return {invert_pass{}}; }
    migraphx::context get_context() const { return {}; }
};

struct double_invert_target
{
    std::string name() const { return "double_invert"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&, const migraphx::compile_options&) const
    {
        return {invert_pass{}, invert_pass{}};
    }
    migraphx::context get_context() const { return {}; }
};

TEST_CASE(literal_test1)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(literal_test2)
{
    migraphx::program p;

    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, sum1, two);

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{5});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(print_test)
{
    migraphx::program p;

    auto x   = p.add_parameter("x", {migraphx::shape::int32_type});
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, x, two);

    std::stringstream ss;
    ss << p;
    std::string s = ss.str();
    EXPECT(!s.empty());
}

TEST_CASE(param_test)
{
    migraphx::program p;

    auto x = p.add_parameter("x", {migraphx::shape::int32_type});
    auto y = p.add_parameter("y", {migraphx::shape::int32_type});

    p.add_instruction(sum_op{}, x, y);
    auto result = p.eval(
        {{"x", migraphx::literal{1}.get_argument()}, {"y", migraphx::literal{2}.get_argument()}});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(param_error_test)
{
    migraphx::program p;

    auto x = p.add_parameter("x", {migraphx::shape::int32_type});
    auto y = p.add_parameter("y", {migraphx::shape::int32_type});

    p.add_instruction(sum_op{}, x, y);
    EXPECT(test::throws<migraphx::exception>(
        [&] {
            p.eval({{"x", migraphx::literal{1}.get_argument()}});
        },
        "Parameter not found: y"));
}

TEST_CASE(param_error_shape_test)
{
    migraphx::program p;

    auto x = p.add_parameter("x", {migraphx::shape::int32_type, {1, 1}});
    auto y = p.add_parameter("y", {migraphx::shape::int32_type, {1, 1}});

    p.add_instruction(sum_op{}, x, y);
    EXPECT(test::throws<migraphx::exception>(
        [&] {
            p.eval({
                {"x", migraphx::literal{1}.get_argument()},
                {"y", migraphx::literal{{migraphx::shape::int32_type, {1, 1}}, {2}}.get_argument()},
            });
        },
        "Incorrect shape {int32_type, {1}, {0}} for parameter: x"));
}

TEST_CASE(get_param1)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::int32_type, {1, 2}};
    auto x = p.add_parameter("x", s);
    auto y = p.add_parameter("y", s);
    p.add_instruction(sum_op{}, x, y);
    EXPECT(bool{p.get_parameter("x") == x});
    EXPECT(bool{p.get_parameter("y") == y});
    EXPECT(bool{p.get_parameter("nonexistent") == p.end()});
}

TEST_CASE(get_param2)
{
    migraphx::program p;
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    EXPECT(bool{p.get_parameter("nonexistent") == p.end()});
}

TEST_CASE(get_param_shapes)
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::int32_type, {1, 2}};
    auto x = p.add_parameter("x", s);
    auto y = p.add_parameter("y", s);
    p.add_instruction(sum_op{}, x, y);
    auto m = p.get_parameter_shapes();
    EXPECT(m.count("nonexistent") == 0);
    EXPECT(m.at("x") == s);
    EXPECT(m.at("y") == s);
}

TEST_CASE(replace_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    p.replace_instruction(sum, minus_op{}, two, one);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_ins_test)
{
    migraphx::program p;

    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto sum   = p.add_instruction(sum_op{}, one, two);
    auto minus = p.add_instruction(minus_op{}, two, one);
    p.replace_instruction(sum, minus);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_ins_test2)
{
    migraphx::program p;

    auto one   = p.add_literal(1);
    auto two   = p.add_literal(2);
    auto sum   = p.add_instruction(sum_op{}, one, two);
    auto minus = p.add_instruction(minus_op{}, two, one);
    p.add_instruction(pass_op{}, minus);
    p.replace_instruction(two, sum);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{2});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_op_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, two, one);
    sum->replace(minus_op{});
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_op_recompute_shape_throw)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    auto sum = p.add_instruction(sum_op{}, one, two);
    EXPECT(test::throws<migraphx::exception>([&] { sum->replace(unary_pass_op{}); }));
}

TEST_CASE(insert_replace_test)
{
    migraphx::program p;

    auto one  = p.add_literal(1);
    auto two  = p.add_literal(2);
    auto sum1 = p.add_instruction(sum_op{}, one, two);
    p.add_instruction(sum_op{}, sum1, two);

    auto sum0 = p.insert_instruction(sum1, sum_op{}, two, two);
    p.replace_instruction(sum1, minus_op{}, sum0, two);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{4});
    EXPECT(result != migraphx::literal{5});
}

TEST_CASE(remove_test1)
{
    migraphx::program p;

    auto one     = p.add_literal(1);
    auto two     = p.add_literal(2);
    auto sum     = p.add_instruction(sum_op{}, one, two);
    auto removed = p.add_instruction(minus_op{}, sum, one);
    p.remove_instruction(removed);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{1});
}

TEST_CASE(remove_test2)
{
    migraphx::program p;

    auto one     = p.add_literal(1);
    auto two     = p.add_literal(2);
    auto removed = p.add_instruction(minus_op{}, two, one);
    p.add_instruction(sum_op{}, one, two);
    p.remove_instruction(removed);
    EXPECT(bool{p.validate() == p.end()});

    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{1});
}

TEST_CASE(target_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    p.compile(id_target{});
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(invert_target_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, two, one);
    p.compile(invert_target{});
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(double_invert_target_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, two, one);
    p.compile(double_invert_target{});
    auto result = p.eval({});
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(reverse_target_test)
{
    migraphx::program p;

    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    EXPECT(test::throws<migraphx::exception>([&] { p.compile(reverse_target{}); }));
}

// Check that the program doesnt modify the context directly, and only the operators modify the
// context
TEST_CASE(eval_context1)
{
    migraphx::program p;
    id_target t{};
    EXPECT(is_shared(t.ctx, t.get_context()));
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(sum_op{}, one, two);
    p.compile(t);
    EXPECT(is_shared(t.ctx, p.get_context()));
    p.eval({});
    EXPECT(is_shared(t.ctx, p.get_context()));
}

TEST_CASE(eval_context2)
{
    migraphx::program p;
    id_target t{};
    EXPECT(is_shared(t.ctx, t.get_context()));
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(id_ctx_op{}, one, two);
    p.compile(t);
    EXPECT(is_shared(t.ctx, p.get_context()));
    p.eval({});
    // id_ctx_op will modify the context
    EXPECT(not is_shared(t.ctx, p.get_context()));
}

TEST_CASE(eval_context3)
{
    migraphx::program p;
    id_target t{};
    EXPECT(is_shared(t.ctx, t.get_context()));
    auto one = p.add_literal(1);
    auto two = p.add_literal(2);
    p.add_instruction(id_ctx_final_op{}, one, two);
    p.compile(t);
    // Finalizer will modify the context
    EXPECT(not is_shared(t.ctx, p.get_context()));
    auto ctx = p.get_context();
    p.eval({});
    EXPECT(is_shared(ctx, p.get_context()));
    EXPECT(not is_shared(t.ctx, p.get_context()));
}

struct cout_redirect
{
    cout_redirect()                     = delete;
    cout_redirect(const cout_redirect&) = delete;
    template <class T>
    cout_redirect(T& stream) : old(std::cout.rdbuf(stream.rdbuf()))
    {
    }
    ~cout_redirect() { std::cout.rdbuf(old); }

    private:
    std::streambuf* old;
};

template <class F>
std::string capture_output(F f)
{
    std::stringstream ss;
    cout_redirect cr{ss};
    f();
    return ss.str();
}

TEST_CASE(debug_print_test)
{
    migraphx::program p;
    auto one                                    = p.add_literal(1);
    std::vector<migraphx::instruction_ref> onev = {one};

    migraphx::program p2;
    auto one2 = p2.add_literal(1);

    auto program_out = migraphx::trim(capture_output([&] { p.debug_print(); }));
    auto ins_out     = migraphx::trim(capture_output([&] { p.debug_print(one); }));
    auto inss_out    = migraphx::trim(capture_output([&] { p.debug_print(onev); }));
    auto end_out     = migraphx::trim(capture_output([&] { p.debug_print(p.end()); }));
    auto p2_ins_out  = migraphx::trim(capture_output([&] { p.debug_print(one2); }));

    EXPECT(program_out == ins_out);
    EXPECT(inss_out == ins_out);
    EXPECT(end_out == "End instruction");
    EXPECT(p2_ins_out == "Instruction not part of program");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
