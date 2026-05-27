/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/execution_environment.hpp>
#include <migraphx/any_ptr.hpp>
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
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {};
    }
    migraphx::context get_context() const { return ctx; }
};

struct id_ctx_op
{
    std::string name() const { return ""; }
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
    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>&) const { return {0}; }
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
    std::vector<std::size_t> output_alias(const std::vector<migraphx::shape>&) const { return {0}; }
};

struct reverse_pass
{
    std::string name() const { return "reverse_pass"; }

    void apply(migraphx::module& m) const { std::reverse(m.begin(), m.end()); }
};

struct reverse_target
{
    std::string name() const { return "reverse"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {reverse_pass{}};
    }
    migraphx::context get_context() const { return {}; }
};

struct invert_pass
{
    std::string name() const { return "invert_pass"; }

    void apply(migraphx::module& m) const
    {
        for(auto ins : migraphx::iterator_for(m))
        {
            if(ins->name() == "sum")
            {
                m.replace_instruction(ins, minus_op{}, ins->inputs());
            }
            else if(ins->name() == "minus")
            {
                m.replace_instruction(ins, sum_op{}, ins->inputs());
            }
        }
    }
};

struct invert_target
{
    std::string name() const { return "invert"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {invert_pass{}};
    }
    migraphx::context get_context() const { return {}; }
};

struct double_invert_target
{
    std::string name() const { return "double_invert"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {invert_pass{}, invert_pass{}};
    }
    migraphx::context get_context() const { return {}; }
};

// Minimal context that implements the optional set_queue/restore_queue members
// of the context concept
// Each call bumps a counter so a test can verify the dispatch routed through.
struct tracked_ctx
{
    int set_calls     = 0;
    int restore_calls = 0;
    migraphx::any_ptr last_queue{};

    void finish() const {}
    void set_queue(migraphx::any_ptr q)
    {
        ++set_calls;
        last_queue = q;
    }
    void restore_queue() { ++restore_calls; }
};

TEST_CASE(literal_test1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(literal_test2)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(migraphx::make_op("add"), one, two);
    mm->add_instruction(migraphx::make_op("add"), sum1, two);

    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{5});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(print_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::int32_type});
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), x, two);

    std::stringstream ss;
    ss << p;
    std::string s = ss.str();
    EXPECT(not s.empty());
}

TEST_CASE(param_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::int32_type});
    auto y   = mm->add_parameter("y", {migraphx::shape::int32_type});

    mm->add_instruction(migraphx::make_op("add"), x, y);
    auto result = p.eval({{"x", migraphx::literal{1}.get_argument()},
                          {"y", migraphx::literal{2}.get_argument()}})
                      .back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(param_error_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::int32_type});
    auto y   = mm->add_parameter("y", {migraphx::shape::int32_type});

    mm->add_instruction(sum_op{}, x, y);
    EXPECT(test::throws<migraphx::exception>(
        [&] {
            p.eval({{"x", migraphx::literal{1}.get_argument()}});
        },
        "Parameter not found: y"));
}

TEST_CASE(param_error_shape_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", {migraphx::shape::int32_type, {1, 1}});
    auto y   = mm->add_parameter("y", {migraphx::shape::int32_type, {1, 1}});

    mm->add_instruction(migraphx::make_op("add"), x, y);
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
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {1, 2}};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    mm->add_instruction(migraphx::make_op("add"), x, y);
    EXPECT(p.get_parameter("x") == x);
    EXPECT(p.get_parameter("y") == y);
    EXPECT(p.get_parameter("nonexistent") == mm->end());
}

TEST_CASE(get_param2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    EXPECT(p.get_parameter("nonexistent") == mm->end());
}

TEST_CASE(get_param_shapes)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::int32_type, {1, 2}};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);
    mm->add_instruction(migraphx::make_op("add"), x, y);
    auto m = p.get_parameter_shapes();
    EXPECT(m.count("nonexistent") == 0);
    EXPECT(m.at("x") == s);
    EXPECT(m.at("y") == s);
}

TEST_CASE(replace_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(migraphx::make_op("add"), one, two);
    mm->replace_instruction(sum, migraphx::make_op("sub"), two, one);
    EXPECT(p.validate() == mm->end());

    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_ins_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto sum   = mm->add_instruction(migraphx::make_op("add"), one, two);
    auto minus = mm->add_instruction(migraphx::make_op("sub"), two, one);
    mm->replace_instruction(sum, minus);
    EXPECT(p.validate() == mm->end());

    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_ins_test2)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto one   = mm->add_literal(1);
    auto two   = mm->add_literal(2);
    auto sum   = mm->add_instruction(migraphx::make_op("add"), one, two);
    auto minus = mm->add_instruction(migraphx::make_op("sub"), two, one);
    mm->add_instruction(pass_op{}, minus);
    mm->replace_instruction(two, sum);
    EXPECT(p.validate() == mm->end());

    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{2});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_op_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(migraphx::make_op("add"), two, one);
    sum->replace(migraphx::make_op("sub"));
    EXPECT(p.validate() == mm->end());

    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{3});
}

TEST_CASE(replace_op_recompute_shape_throw)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(migraphx::make_op("add"), one, two);
    EXPECT(test::throws<migraphx::exception>([&] { sum->replace(unary_pass_op{}); }));
}

TEST_CASE(insert_replace_test)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto one  = mm->add_literal(1);
    auto two  = mm->add_literal(2);
    auto sum1 = mm->add_instruction(migraphx::make_op("add"), one, two);
    mm->add_instruction(migraphx::make_op("add"), sum1, two);

    auto sum0 = mm->insert_instruction(sum1, migraphx::make_op("add"), two, two);
    mm->replace_instruction(sum1, migraphx::make_op("sub"), sum0, two);
    EXPECT(p.validate() == mm->end());

    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{4});
    EXPECT(result != migraphx::literal{5});
}

TEST_CASE(remove_test1)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto one     = mm->add_literal(1);
    auto two     = mm->add_literal(2);
    auto sum     = mm->add_instruction(migraphx::make_op("add"), one, two);
    auto removed = mm->add_instruction(migraphx::make_op("sub"), sum, one);
    mm->remove_instruction(removed);
    EXPECT(p.validate() == mm->end());

    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{1});
}

TEST_CASE(remove_test2)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto one     = mm->add_literal(1);
    auto two     = mm->add_literal(2);
    auto removed = mm->add_instruction(migraphx::make_op("sub"), two, one);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    mm->remove_instruction(removed);
    EXPECT(p.validate() == mm->end());

    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{1});
}

TEST_CASE(target_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    p.compile(id_target{});
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(invert_target_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, two, one);
    p.compile(invert_target{});
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{1});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(double_invert_target_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, two, one);
    p.compile(double_invert_target{});
    auto result = p.eval({}).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(result != migraphx::literal{4});
}

TEST_CASE(reverse_target_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, one, two);
    EXPECT(test::throws<migraphx::exception>([&] { p.compile(reverse_target{}); }));
}

// Check that the program doesnt modify the context directly, and only the operators modify the
// context
TEST_CASE(eval_context1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    id_target t{};
    EXPECT(is_shared(t.ctx, t.get_context()));
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(sum_op{}, one, two);
    p.compile(t);
    EXPECT(is_shared(t.ctx, p.get_context()));
    std::ignore = p.eval({}).back();
    EXPECT(is_shared(t.ctx, p.get_context()));
}

TEST_CASE(eval_context2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    id_target t{};
    EXPECT(is_shared(t.ctx, t.get_context()));
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(id_ctx_op{}, one, two);
    p.compile(t);
    EXPECT(is_shared(t.ctx, p.get_context()));
    std::ignore = p.eval({}).back();
    // id_ctx_op will modify the context
    EXPECT(not is_shared(t.ctx, p.get_context()));
}

TEST_CASE(eval_context3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    id_target t{};
    EXPECT(is_shared(t.ctx, t.get_context()));
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(id_ctx_final_op{}, one, two);
    p.compile(t);
    // Finalizer will modify the context
    EXPECT(not is_shared(t.ctx, p.get_context()));
    auto ctx    = p.get_context();
    std::ignore = p.eval({}).back();
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
static std::string capture_output(F f)
{
    std::stringstream ss;
    cout_redirect cr{ss};
    f();
    return ss.str();
}

TEST_CASE(debug_print_test)
{
    migraphx::program p;
    auto* mm                                    = p.get_main_module();
    auto one                                    = mm->add_literal(1);
    std::vector<migraphx::instruction_ref> onev = {one};

    migraphx::program p2;
    auto* mm2 = p2.get_main_module();
    auto one2 = mm2->add_literal(1);

    auto program_out = migraphx::trim(capture_output([&] { mm->debug_print(); }));
    auto ins_out     = migraphx::trim(capture_output([&] { mm->debug_print(one); }));
    auto inss_out    = migraphx::trim(capture_output([&] { mm->debug_print(onev); }));
    auto end_out     = migraphx::trim(capture_output([&] { mm->debug_print(mm->end()); }));
    auto p2_ins_out  = migraphx::trim(capture_output([&] { mm->debug_print(one2); }));

    EXPECT(program_out == ins_out);
    EXPECT(inss_out == ins_out);
    EXPECT(end_out == "End instruction");
    EXPECT(p2_ins_out == "Instruction not part of module");
}

TEST_CASE(eval_trace_fires_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    p.compile(id_target{});

    std::vector<std::string> fired_ops;
    migraphx::execution_environment exec_env;
    exec_env.trace = [&](migraphx::instruction_ref ins, const migraphx::argument&) {
        fired_ops.push_back(ins->name());
    };

    auto result = p.eval({}, exec_env).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(not fired_ops.empty());
    EXPECT(fired_ops.back() == "add");
}

TEST_CASE(eval_trace_disabled_falls_through)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);

    migraphx::execution_environment exec_env;
    auto result = p.eval({}, exec_env).back();
    EXPECT(result == migraphx::literal{3});
}

TEST_CASE(eval_trace_filter_by_name_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(migraphx::make_op("add"), one, two);
    mm->add_instruction(migraphx::make_op("sub"), sum, one);
    p.compile(id_target{});

    std::vector<std::string> fired_ops;
    migraphx::execution_environment exec_env;
    exec_env.trace = [&](migraphx::instruction_ref ins, const migraphx::argument&) {
        if(ins->name() == "sub")
            fired_ops.push_back(ins->name());
    };

    auto result = p.eval({}, exec_env).back();
    EXPECT(result == migraphx::literal{2});
    EXPECT(fired_ops.size() == 1);
    EXPECT(fired_ops.front() == "sub");
}

TEST_CASE(eval_trace_filter_by_ins_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    auto sum = mm->add_instruction(migraphx::make_op("add"), one, two);
    mm->add_instruction(migraphx::make_op("sub"), sum, one);
    p.compile(id_target{});

    std::vector<std::string> fired_ops;
    migraphx::execution_environment exec_env;
    exec_env.trace = [&](migraphx::instruction_ref ins, const migraphx::argument&) {
        if(ins == sum)
            fired_ops.push_back(ins->name());
    };

    auto result = p.eval({}, exec_env).back();
    EXPECT(result == migraphx::literal{2});
    EXPECT(fired_ops.size() == 1);
    EXPECT(fired_ops.front() == "add");
}

TEST_CASE(eval_trace_with_target_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    p.compile(id_target{});

    std::vector<std::string> fired_ops;
    migraphx::execution_environment exec_env;
    exec_env.trace = [&](migraphx::instruction_ref ins, const migraphx::argument&) {
        fired_ops.push_back(ins->name());
    };

    auto result = p.eval({}, exec_env).back();
    EXPECT(result == migraphx::literal{3});
    EXPECT(not fired_ops.empty());
}

TEST_CASE(async_eval_on_cpu_target_invokes_set_and_restore_queue)
{
    // The async branches of program::eval() call contexts.front().set_queue()
    // before generic_eval and contexts.front().restore_queue() after.  id_target
    // wraps an id_target::context that has no set_queue/restore_queue members,
    // so this exercises the type-erased facade end-to-end on a non-GPU build:
    //   - program::eval async prologue + epilogue (set_queue / restore_queue)
    //   - context::set_queue / context::restore_queue facade bodies
    //   - the float-overload (no-member) dispatchers
    //   - the default set_queue_context / restore_queue_context free-function
    //     fallbacks
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    p.compile(id_target{});

    int dummy = 0;
    migraphx::execution_environment exec_env;
    exec_env.queue = migraphx::any_ptr{&dummy};
    exec_env.async = true;

    auto result = p.eval({}, exec_env).back();
    EXPECT(result == migraphx::literal{3});

    // A default-constructed any_ptr is a legal exec_env.queue and must also
    // round-trip through set_queue / restore_queue without throwing.
    migraphx::execution_environment exec_env_null;
    exec_env_null.async = true;
    auto result2        = p.eval({}, exec_env_null).back();
    EXPECT(result2 == migraphx::literal{3});
}

TEST_CASE(context_facade_dispatches_to_member_set_and_restore_queue)
{
    // Sister test of the one above: tracked_ctx *does* implement set_queue and
    // restore_queue.
    migraphx::context ctx{tracked_ctx{}};

    int dummy = 0;
    migraphx::any_ptr q{&dummy};
    ctx.set_queue(q);
    ctx.set_queue(q);
    ctx.restore_queue();

    auto* held = ctx.any_cast<tracked_ctx>();
    EXPECT(held != nullptr);
    EXPECT(held->set_calls == 2);
    EXPECT(held->restore_calls == 1);
    EXPECT(held->last_queue.unsafe_get() == &dummy);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
