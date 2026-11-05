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
#include <migraphx/builtin.hpp>
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

// A target that records the device it compiled for in a comment, like the gpu target, so
// program::recompile can be tested without a GPU
static std::string& current_device()
{
    static std::string device = "dev_a";
    return device;
}

static const std::string& device_tag()
{
    static const std::string tag = "test_device";
    return tag;
}

static migraphx::instruction_ref find_device_comment(migraphx::module& m)
{
    return std::find_if(m.begin(), m.end(), [](const auto& ins) {
        return ins.name() == "@comment" and
               migraphx::any_cast<migraphx::builtin::comment>(ins.get_operator()).tag ==
                   device_tag();
    });
}

static std::string recorded_device(migraphx::program& p)
{
    auto* mm = p.get_main_module();
    auto it  = find_device_comment(*mm);
    if(it == mm->end())
        return "";
    return migraphx::any_cast<migraphx::builtin::comment>(it->get_operator()).text;
}

static void record_device(migraphx::module& m, const std::string& device)
{
    migraphx::builtin::comment op{device, device_tag()};
    auto it = find_device_comment(m);
    m.insert_instruction(it == m.end() ? m.begin() : it, op);
    if(it != m.end())
        m.remove_instruction(it);
}

struct recompile_record
{
    std::string from;
    std::string to;
    bool exhaustive_tune = false;
    int runs             = 0;
};

static recompile_record& last_recompile()
{
    static recompile_record record;
    return record;
}

// Makes the recompile pass leave an instruction before the input it uses
static bool& break_recompile()
{
    static bool b = false;
    return b;
}

struct recompile_test_context
{
    std::string device = current_device();

    migraphx::value to_value() const { return migraphx::value{{"device", device}}; }
    void from_value(const migraphx::value& v) { device = v.at("device").to<std::string>(); }
    void finish() const {}
};

struct record_device_pass
{
    std::string name() const { return "record_device"; }
    void apply(migraphx::module& m) const { record_device(m, current_device()); }
};

struct record_recompile_pass
{
    bool exhaustive_tune = false;

    std::string name() const { return "record_recompile"; }
    void apply(migraphx::module& m) const
    {
        auto it = find_device_comment(m);
        if(it != m.end())
            last_recompile().from =
                migraphx::any_cast<migraphx::builtin::comment>(it->get_operator()).text;
        last_recompile().to              = current_device();
        last_recompile().exhaustive_tune = exhaustive_tune;
        last_recompile().runs++;
        record_device(m, current_device());
        if(break_recompile())
        {
            auto x = std::find_if(
                m.begin(), m.end(), [](const auto& ins) { return ins.name() == "@param"; });
            m.insert_instruction(m.begin(), migraphx::make_op("neg"), x);
        }
    }
};

struct recompile_test_target
{
    std::string name() const { return "recompile_test"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options&) const
    {
        return {record_device_pass{}};
    }
    std::vector<migraphx::pass> get_recompile_passes(migraphx::context&,
                                                     const migraphx::compile_options& options) const
    {
        return {record_recompile_pass{options.exhaustive_tune}};
    }
    migraphx::context get_context() const { return recompile_test_context{}; }
};

static migraphx::target make_recompile_target()
{
    migraphx::register_target<recompile_test_target>();
    return migraphx::make_target("recompile_test");
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
    p.compile(make_recompile_target());
    return p;
}

static migraphx::program reload(const migraphx::program& p, bool finalize)
{
    migraphx::file_options options;
    options.finalize = finalize;
    return migraphx::load_buffer(migraphx::save_buffer(p), options);
}

static std::string context_device(const migraphx::program& p)
{
    return p.get_context().to_value().at("device").to<std::string>();
}

TEST_CASE(recompile_requires_compiled_program)
{
    auto p = make_program();
    EXPECT(not p.is_compiled());
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(make_recompile_target()); },
                                             "not compiled"));
}

TEST_CASE(recompile_requires_same_target)
{
    auto p = compile_for("dev_a");
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(migraphx::make_target("ref")); },
                                             "not compiled for the ref target"));
}

TEST_CASE(recompile_unsupported_target_refused)
{
    auto p = make_program();
    p.compile(migraphx::make_target("ref"));
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(migraphx::make_target("ref")); },
                                             "does not support recompile"));
}

TEST_CASE(recompile_runs_recompile_passes)
{
    auto p = compile_for("dev_a");
    EXPECT(recorded_device(p) == "dev_a");
    last_recompile() = {};
    current_device() = "dev_b";
    migraphx::compile_options options;
    options.exhaustive_tune = true;
    p.recompile(make_recompile_target(), options);
    EXPECT(last_recompile().runs == 1);
    EXPECT(last_recompile().from == "dev_a");
    EXPECT(last_recompile().to == "dev_b");
    EXPECT(last_recompile().exhaustive_tune);
    EXPECT(recorded_device(p) == "dev_b");
    EXPECT(context_device(p) == "dev_b");
}

TEST_CASE(recompile_invalid_module_refused)
{
    auto p            = compile_for("dev_a");
    current_device()  = "dev_b";
    break_recompile() = true;
    // Debug builds already reject the module after the pass, with their own message
    std::string error;
    try
    {
        p.recompile(make_recompile_target());
    }
    catch(const migraphx::exception& e)
    {
        error = e.what();
    }
    break_recompile() = false;
    EXPECT(error.find("Invalid module") != std::string::npos or
           error.find("produces invalid program") != std::string::npos);
    // The program is unchanged when recompiling fails
    EXPECT(recorded_device(p) == "dev_a");
    EXPECT(context_device(p) == "dev_a");
}

TEST_CASE(load_without_finalize)
{
    auto p = reload(compile_for("dev_a"), false);
    EXPECT(p.is_compiled());
    EXPECT(recorded_device(p) == "dev_a");
    EXPECT(test::throws<migraphx::exception>([&] { p.eval({}); }, "without finalizing"));

    // The context is created for the current device when the program is finalized
    current_device() = "dev_b";
    p.finalize();
    EXPECT(context_device(p) == "dev_b");
}

TEST_CASE(load_with_finalize_restores_context)
{
    auto p           = compile_for("dev_a");
    current_device() = "dev_b";
    auto q           = reload(p, true);
    EXPECT(context_device(q) == "dev_a");
}

TEST_CASE(save_without_contexts)
{
    // A program loaded without finalizing is saved without contexts, which are created on load
    auto p           = reload(compile_for("dev_a"), false);
    current_device() = "dev_b";
    auto q           = reload(p, true);
    EXPECT(q.is_compiled());
    EXPECT(recorded_device(q) == "dev_a");
    EXPECT(context_device(q) == "dev_b");
}

TEST_CASE(get_context_creates_context)
{
    auto p           = reload(compile_for("dev_a"), false);
    current_device() = "dev_b";
    EXPECT(context_device(p) == "dev_b");
}

TEST_CASE(recompile_without_finalize)
{
    auto p           = reload(compile_for("dev_a"), false);
    last_recompile() = {};
    current_device() = "dev_b";
    p.recompile(make_recompile_target());
    EXPECT(last_recompile().runs == 1);
    EXPECT(last_recompile().from == "dev_a");
    EXPECT(not last_recompile().exhaustive_tune);
    EXPECT(recorded_device(p) == "dev_b");
}

TEST_CASE(clear_context_uncompiles)
{
    auto p = reload(compile_for("dev_a"), false);
    p.clear_context();
    EXPECT(not p.is_compiled());
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

TEST_CASE(older_program_version_refused)
{
    auto v       = make_program().to_value();
    v["version"] = 8;
    migraphx::program p;
    EXPECT(test::throws<migraphx::exception>([&] { p.from_value(v); }, "version mismatch"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
