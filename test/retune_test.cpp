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
#include <migraphx/compile_options.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/value.hpp>
#include <test.hpp>

#include <algorithm>
#include <string>
#include <vector>

// A target whose context records the device it was compiled for, like the gpu target, so
// program::retune can be tested without a GPU
static std::string& current_device()
{
    static std::string device = "dev_a";
    return device;
}

struct retune_record
{
    std::string from;
    std::string to;
    bool force = false;
    int runs   = 0;
};

static retune_record& last_retune()
{
    static retune_record record;
    return record;
}

struct retune_test_context
{
    std::string compiled_for = current_device();

    migraphx::value to_value() const { return migraphx::value{{"compiled_for", compiled_for}}; }
    void from_value(const migraphx::value& v)
    {
        compiled_for = v.at("compiled_for").to<std::string>();
    }
    void finish() const {}
};

struct record_retune_pass
{
    std::string from;
    bool force = false;

    std::string name() const { return "record_retune"; }
    void apply(migraphx::module&) const
    {
        last_retune().from  = from;
        last_retune().to    = current_device();
        last_retune().force = force;
        last_retune().runs++;
    }
};

struct retune_test_target
{
    std::string name() const { return "retune_test"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {};
    }
    std::vector<migraphx::pass> get_retune_passes(migraphx::context&,
                                                  const migraphx::value& from_ctx,
                                                  const migraphx::compile_options&,
                                                  const migraphx::retune_options& retune) const
    {
        return {record_retune_pass{from_ctx.at("compiled_for").to<std::string>(), retune.force}};
    }
    migraphx::context get_context() const { return retune_test_context{}; }
};

static migraphx::target make_retune_target()
{
    migraphx::register_target<retune_test_target>();
    return migraphx::make_target("retune_test");
}

static migraphx::program make_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    return p;
}

static migraphx::program compile_for(const std::string& device)
{
    current_device() = device;
    auto p           = make_program();
    p.compile(make_retune_target());
    return p;
}

static migraphx::program reload(const migraphx::program& p, bool finalize)
{
    migraphx::file_options options;
    options.finalize = finalize;
    return migraphx::load_buffer(migraphx::save_buffer(p), options);
}

TEST_CASE(compiled_for_recorded)
{
    auto p = compile_for("dev_a");
    EXPECT(p.is_compiled());
    EXPECT(p.get_compiled_for().to<std::string>() == "dev_a");
}

TEST_CASE(compiled_for_null_when_not_recorded)
{
    auto p = make_program();
    EXPECT(not p.is_compiled());
    EXPECT(p.get_compiled_for().is_null());
}

TEST_CASE(retune_requires_compiled_program)
{
    auto p = make_program();
    EXPECT(
        test::throws<migraphx::exception>([&] { p.retune(make_retune_target()); }, "not compiled"));
}

TEST_CASE(retune_requires_same_target)
{
    auto p = compile_for("dev_a");
    EXPECT(test::throws<migraphx::exception>([&] { p.retune(migraphx::make_target("ref")); },
                                             "not compiled for the ref target"));
}

TEST_CASE(retune_unsupported_target_refused)
{
    auto p = make_program();
    p.compile(migraphx::make_target("ref"));
    EXPECT(test::throws<migraphx::exception>([&] { p.retune(migraphx::make_target("ref")); },
                                             "does not support retune"));
}

TEST_CASE(retune_runs_retune_passes)
{
    auto p           = compile_for("dev_a");
    last_retune()    = {};
    current_device() = "dev_b";
    migraphx::retune_options options;
    options.force = true;
    p.retune(make_retune_target(), {}, options);
    EXPECT(last_retune().runs == 1);
    EXPECT(last_retune().from == "dev_a");
    EXPECT(last_retune().to == "dev_b");
    EXPECT(last_retune().force);
    EXPECT(p.get_compiled_for().to<std::string>() == "dev_b");
}

TEST_CASE(load_without_finalize)
{
    auto p = reload(compile_for("dev_a"), false);
    EXPECT(p.is_compiled());
    EXPECT(p.get_compiled_for().to<std::string>() == "dev_a");
    EXPECT(test::throws<migraphx::exception>([&] { p.eval({}); }, "without finalizing"));

    // Saving again keeps the contexts that were never created
    auto q = reload(p, false);
    EXPECT(q.get_compiled_for().to<std::string>() == "dev_a");

    current_device() = "dev_b";
    p.finalize();
    EXPECT(p.get_compiled_for().to<std::string>() == "dev_a");
    EXPECT(p.get_context().to_value().at("compiled_for").to<std::string>() == "dev_a");
}

TEST_CASE(retune_without_finalize)
{
    auto p           = reload(compile_for("dev_a"), false);
    last_retune()    = {};
    current_device() = "dev_b";
    p.retune(make_retune_target());
    EXPECT(last_retune().runs == 1);
    EXPECT(last_retune().from == "dev_a");
    EXPECT(not last_retune().force);
    EXPECT(p.get_compiled_for().to<std::string>() == "dev_b");
}

TEST_CASE(load_with_finalize)
{
    auto p = reload(compile_for("dev_a"), true);
    EXPECT(p.get_compiled_for().to<std::string>() == "dev_a");
}

TEST_CASE(clear_context_drops_pending)
{
    auto p = reload(compile_for("dev_a"), false);
    p.clear_context();
    EXPECT(p.get_compiled_for().is_null());
}

static migraphx::program make_bypass_program()
{
    migraphx::program p;
    auto* sub = p.create_module("sub");
    auto sx   = sub->add_parameter("sx", migraphx::shape{migraphx::shape::float_type, {4}});
    sub->add_return({sub->add_instruction(migraphx::make_op("neg"), sx)});
    sub->set_bypass();
    auto* other = p.create_module("other");
    auto ox     = other->add_parameter("ox", migraphx::shape{migraphx::shape::float_type, {4}});
    other->add_return({other->add_instruction(migraphx::make_op("abs"), ox)});
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("relu"), x)});
    return p;
}

TEST_CASE(bypass_survives_serialization)
{
    auto p = make_bypass_program();
    migraphx::program q;
    q.from_value(p.to_value());
    EXPECT(q.get_module("sub")->bypass());
    EXPECT(not q.get_module("other")->bypass());
    EXPECT(not q.get_main_module()->bypass());
}

TEST_CASE(print_skips_bypassed_modules)
{
    auto p = make_bypass_program();
    std::vector<std::string> printed;
    p.print([&](auto ins, const auto&) { printed.push_back(ins->name()); });
    EXPECT(std::count(printed.begin(), printed.end(), "neg") == 0);
    EXPECT(std::count(printed.begin(), printed.end(), "abs") == 1);
    EXPECT(std::count(printed.begin(), printed.end(), "relu") == 1);
}

TEST_CASE(version_8_still_loads)
{
    auto v       = make_program().to_value();
    v["version"] = 8;
    migraphx::program p;
    p.from_value(v);
    EXPECT(p.get_main_module()->size() == make_program().get_main_module()->size());

    v["version"] = 7;
    migraphx::program q;
    EXPECT(test::throws<migraphx::exception>([&] { q.from_value(v); }, "version mismatch"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
