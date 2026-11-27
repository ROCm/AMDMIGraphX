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
#include <migraphx/algorithm.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/external_data.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/tmp_dir.hpp>
#include <migraphx/value.hpp>
#include <test.hpp>

#include <algorithm>
#include <iterator>
#include <sstream>
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

static migraphx::instruction_ref find_device_comment(const migraphx::module& m)
{
    return std::find_if(m.begin(), m.end(), [](const auto& ins) {
        return ins.name() == "@comment" and
               migraphx::any_cast<migraphx::builtin::comment>(ins.get_operator()).tag ==
                   device_tag();
    });
}

static std::string recorded_device(const migraphx::module& m)
{
    auto it = find_device_comment(m);
    if(it == m.end())
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
    // Leave an instruction before the input it uses, which makes the module invalid
    bool break_module = false;

    std::string name() const { return "record_recompile"; }
    void apply(migraphx::module& m) const
    {
        last_recompile().from            = recorded_device(m);
        last_recompile().to              = current_device();
        last_recompile().exhaustive_tune = exhaustive_tune;
        last_recompile().runs++;
        record_device(m, current_device());
        if(break_module)
            m.insert_instruction(m.begin(), migraphx::make_op("neg"), m.get_parameter("x"));
    }
};

// Return a view of memory borrowed with load, which is a dangling reference once the program
// returns
struct dangling_return_pass
{
    std::string name() const { return "dangling_return"; }
    void apply(migraphx::module& m) const
    {
        migraphx::shape s{migraphx::shape::float_type, {4}};
        auto ret   = std::prev(m.end());
        auto alloc = m.insert_instruction(
            ret, migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}));
        auto view = m.insert_instruction(
            ret,
            migraphx::make_op("load", {{"shape", migraphx::to_value(s)}, {"offset", 0}}),
            alloc);
        m.replace_return({m.insert_instruction(ret, migraphx::make_op("identity"), view)});
    }
};

static std::vector<migraphx::pass> with_dangling(std::vector<migraphx::pass> passes,
                                                 const migraphx::compile_options& options)
{
    if(migraphx::contains(options.backend_options, "dangling_return"))
        passes.push_back(dangling_return_pass{});
    return passes;
}

struct recompile_test_target
{
    std::string name() const { return "recompile_test"; }
    std::vector<migraphx::pass> get_passes(migraphx::context&,
                                           const migraphx::compile_options& options) const
    {
        return with_dangling({record_device_pass{}}, options);
    }
    std::vector<migraphx::pass> get_recompile_passes(migraphx::context&,
                                                     const migraphx::compile_options& options) const
    {
        return with_dangling(
            {record_recompile_pass{options.exhaustive_tune,
                                   migraphx::contains(options.backend_options, "break_recompile")}},
            options);
    }
    migraphx::context get_context() const { return recompile_test_context{}; }
};

static migraphx::target make_recompile_target()
{
    migraphx::register_target<recompile_test_target>();
    return migraphx::make_target("recompile_test");
}

static migraphx::program reload(const migraphx::program& p, bool finalize)
{
    return migraphx::load_buffer(migraphx::save_buffer(p), {.finalize = finalize});
}

static std::string context_device(const migraphx::program& p)
{
    return p.get_context().to_value().at("device").to<std::string>();
}

// The external references of the main module's literals in a saved msgpack program
static std::vector<migraphx::external_data> saved_external_refs(const std::string& file)
{
    auto v            = migraphx::from_msgpack(migraphx::read_buffer(file));
    const auto& nodes = v.at("modules").at("main").at("nodes");
    std::vector<migraphx::external_data> result;
    migraphx::transform_if(
        nodes.begin(),
        nodes.end(),
        std::back_inserter(result),
        [](const migraphx::value& node) {
            return node.contains("literal") and node.at("literal").contains("external");
        },
        [](const migraphx::value& node) {
            return migraphx::from_value<migraphx::external_data>(node.at("literal").at("external"));
        });
    return result;
}

static std::vector<migraphx::literal> main_literals(const migraphx::program& p)
{
    const auto* mm = p.get_main_module();
    std::vector<migraphx::literal> result;
    migraphx::transform_if(
        mm->begin(),
        mm->end(),
        std::back_inserter(result),
        [](const migraphx::instruction& ins) { return ins.name() == "@literal"; },
        [](const migraphx::instruction& ins) { return ins.get_literal(); });
    return result;
}

TEST_CASE(recompile_requires_compiled_program)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    EXPECT(not p.is_compiled());
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(make_recompile_target()); },
                                             "not compiled"));
}

TEST_CASE(recompile_requires_same_target)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    p.compile(make_recompile_target());
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(migraphx::make_target("ref")); },
                                             "not compiled for the ref target"));
}

TEST_CASE(recompile_unsupported_target_refused)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    p.compile(migraphx::make_target("ref"));
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(migraphx::make_target("ref")); },
                                             "does not support recompile"));
}

TEST_CASE(recompile_runs_recompile_passes)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    current_device() = "dev_a";
    p.compile(make_recompile_target());
    EXPECT(recorded_device(*mm) == "dev_a");

    last_recompile() = {};
    current_device() = "dev_b";
    migraphx::compile_options options;
    options.exhaustive_tune = true;
    p.recompile(make_recompile_target(), options);
    EXPECT(last_recompile().runs == 1);
    EXPECT(last_recompile().from == "dev_a");
    EXPECT(last_recompile().to == "dev_b");
    EXPECT(last_recompile().exhaustive_tune);
    EXPECT(recorded_device(*p.get_main_module()) == "dev_b");
    EXPECT(context_device(p) == "dev_b");
}

TEST_CASE(recompile_invalid_module_refused)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    current_device() = "dev_a";
    p.compile(make_recompile_target());

    current_device() = "dev_b";
    migraphx::compile_options options;
    options.backend_options["break_recompile"] = true;
    // Debug builds already reject the module after the pass, with their own message
    std::string error;
    try
    {
        p.recompile(make_recompile_target(), options);
    }
    catch(const migraphx::exception& e)
    {
        error = e.what();
    }
    EXPECT(error.find("Invalid module") != std::string::npos or
           error.find("produces invalid program") != std::string::npos);
    // The program is unchanged when recompiling fails
    EXPECT(recorded_device(*p.get_main_module()) == "dev_a");
    EXPECT(context_device(p) == "dev_a");
}

TEST_CASE(compile_dangling_reference_refused)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    migraphx::compile_options options;
    options.backend_options["dangling_return"] = true;
    EXPECT(test::throws<migraphx::exception>([&] { p.compile(make_recompile_target(), options); },
                                             "Dangling reference"));
}

TEST_CASE(recompile_dangling_reference_refused)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    current_device() = "dev_a";
    p.compile(make_recompile_target());

    current_device() = "dev_b";
    migraphx::compile_options options;
    options.backend_options["dangling_return"] = true;
    EXPECT(test::throws<migraphx::exception>([&] { p.recompile(make_recompile_target(), options); },
                                             "Dangling reference"));
    // The program is unchanged when recompiling fails
    EXPECT(recorded_device(*p.get_main_module()) == "dev_a");
}

TEST_CASE(load_without_finalize)
{
    migraphx::program c;
    auto* mm = c.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    current_device() = "dev_a";
    c.compile(make_recompile_target());

    auto p = reload(c, false);
    EXPECT(p.is_compiled());
    EXPECT(recorded_device(*p.get_main_module()) == "dev_a");
    EXPECT(test::throws<migraphx::exception>([&] { p.eval({}); }, "without finalizing"));

    // The context is created for the current device when the program is finalized
    current_device() = "dev_b";
    p.finalize();
    EXPECT(context_device(p) == "dev_b");
}

TEST_CASE(load_with_finalize_restores_context)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    current_device() = "dev_a";
    p.compile(make_recompile_target());

    current_device() = "dev_b";
    auto q           = reload(p, true);
    EXPECT(context_device(q) == "dev_a");
}

TEST_CASE(save_without_contexts)
{
    migraphx::program c;
    auto* mm = c.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    current_device() = "dev_a";
    c.compile(make_recompile_target());

    // A program loaded without finalizing is saved without contexts, which are created on load
    auto p           = reload(c, false);
    current_device() = "dev_b";
    auto q           = reload(p, true);
    EXPECT(q.is_compiled());
    EXPECT(recorded_device(*q.get_main_module()) == "dev_a");
    EXPECT(context_device(q) == "dev_b");
}

TEST_CASE(get_context_creates_context)
{
    migraphx::program c;
    auto* mm = c.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    current_device() = "dev_a";
    c.compile(make_recompile_target());

    auto p           = reload(c, false);
    current_device() = "dev_b";
    EXPECT(context_device(p) == "dev_b");
}

TEST_CASE(recompile_without_finalize)
{
    migraphx::program c;
    auto* mm = c.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    current_device() = "dev_a";
    c.compile(make_recompile_target());

    auto p           = reload(c, false);
    last_recompile() = {};
    current_device() = "dev_b";
    p.recompile(make_recompile_target());
    EXPECT(last_recompile().runs == 1);
    EXPECT(last_recompile().from == "dev_a");
    EXPECT(not last_recompile().exhaustive_tune);
    EXPECT(recorded_device(*p.get_main_module()) == "dev_b");
}

TEST_CASE(clear_context_uncompiles)
{
    migraphx::program c;
    auto* mm = c.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    c.compile(make_recompile_target());

    auto p = reload(c, false);
    p.clear_context();
    EXPECT(not p.is_compiled());
}

TEST_CASE(bypass_survives_serialization)
{
    migraphx::shape s{migraphx::shape::float_type, {4}};
    migraphx::program p;
    auto* sub = p.create_module("sub");
    sub->add_return({sub->add_instruction(migraphx::make_op("neg"), sub->add_parameter("sx", s))});
    sub->set_bypass();
    auto* other = p.create_module("other");
    other->add_return(
        {other->add_instruction(migraphx::make_op("abs"), other->add_parameter("ox", s))});
    auto* mm = p.get_main_module();
    mm->add_return({mm->add_instruction(migraphx::make_op("relu"), mm->add_parameter("x", s))});

    migraphx::program q;
    q.from_value(p.to_value());
    EXPECT(q.get_module("sub")->bypass());
    EXPECT(not q.get_module("other")->bypass());
    EXPECT(not q.get_main_module()->bypass());
}

TEST_CASE(perf_report_without_timing_for_bypassed_modules)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4}};
    mm->add_return({mm->add_instruction(migraphx::make_op("relu"), mm->add_parameter("x", s))});
    p.compile(migraphx::make_target("ref"));
    auto* sub = p.create_module("sub");
    sub->add_return({sub->add_instruction(migraphx::make_op("neg"), sub->add_parameter("sx", s))});
    sub->set_bypass();

    std::stringstream ss;
    p.perf_report(ss, 2, {{"x", migraphx::generate_argument(s)}});
    auto report = ss.str();
    // The bypassed module is printed, but it is never run, so it has no timing
    EXPECT(migraphx::contains(report, "neg"));
    EXPECT(not migraphx::contains(report, "nan"));
}

TEST_CASE(older_program_version_refused)
{
    migraphx::program c;
    auto* mm = c.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4}});
    mm->add_return({mm->add_instruction(migraphx::make_op("neg"), x)});
    auto v       = c.to_value();
    v["version"] = 7;
    migraphx::program p;
    EXPECT(test::throws<migraphx::exception>([&] { p.from_value(v); }, "version mismatch"));
}

TEST_CASE(recompile_external_weights)
{
    migraphx::tmp_dir td{"recompile-external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::shape s{migraphx::shape::float_type, {64}};
    auto weights = migraphx::generate_literal(s, 1);
    migraphx::program c;
    auto* mm = c.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto w   = mm->add_literal(weights);
    mm->add_return({mm->add_instruction(migraphx::make_op("add"), x, w)});
    current_device() = "dev_a";
    c.compile(make_recompile_target());

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(c, filename, options);
    auto refs = saved_external_refs(filename);
    EXPECT(refs.size() == 1);
    EXPECT(migraphx::is_weights_generation(td.path / "model.mxr.data",
                                           td.path / refs.front().file_path));
    EXPECT(migraphx::fs::exists(td.path / refs.front().file_path));

    // Opening the program without its kernels still maps the weights
    auto p      = migraphx::load(filename, {.finalize = false});
    auto loaded = main_literals(p);
    EXPECT(loaded.size() == 1);
    EXPECT(loaded.front().is_external());
    EXPECT(loaded.front().get_mapped_file() != nullptr);

    last_recompile() = {};
    current_device() = "dev_b";
    p.recompile(make_recompile_target());
    EXPECT(last_recompile().runs == 1);
    EXPECT(last_recompile().from == "dev_a");
    EXPECT(recorded_device(*p.get_main_module()) == "dev_b");
    auto recompiled = main_literals(p);
    EXPECT(recompiled.size() == 1);
    EXPECT(recompiled.front().is_external());
    const auto& ext = *recompiled.front().get_external_data();
    EXPECT(ext.offset == refs.front().offset);
    EXPECT(ext.size == refs.front().size);
    EXPECT(ext.checksum == refs.front().checksum);
    EXPECT(recompiled.front() == weights);
}

TEST_CASE(recompile_shares_external_weights)
{
    migraphx::tmp_dir td{"recompile-external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::shape s{migraphx::shape::float_type, {64}};
    auto weights = migraphx::generate_literal(s, 1);
    migraphx::program c;
    auto* mm = c.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto w   = mm->add_literal(weights);
    mm->add_return({mm->add_instruction(migraphx::make_op("add"), x, w)});
    current_device() = "dev_a";
    c.compile(make_recompile_target());

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(c, filename, options);
    auto original_refs = saved_external_refs(filename);
    auto weights_file  = td.path / original_refs.front().file_path;
    auto weights_bytes = migraphx::read_buffer(weights_file);

    auto p           = migraphx::load(filename, {.finalize = false});
    auto original    = p;
    current_device() = "dev_b";
    p.recompile(make_recompile_target());

    // The recompiled program reads the weights from the original's mapping, not from a copy
    auto before = main_literals(original);
    auto after  = main_literals(p);
    EXPECT(before.size() == 1);
    EXPECT(after.size() == 1);
    EXPECT(after.front().get_mapped_file() != nullptr);
    EXPECT(after.front().get_mapped_file() == before.front().get_mapped_file());
    EXPECT(after.front().data() == before.front().data());

    // Saved with reuse, it references the original weights file instead of writing its own
    auto recompiled = (td.path / "recompiled.mxr").string();
    migraphx::file_options reuse;
    reuse.reuse_external_data = true;
    migraphx::save(p, recompiled, reuse);
    EXPECT(not migraphx::fs::exists(recompiled + ".data"));
    auto recompiled_refs = saved_external_refs(recompiled);
    EXPECT(recompiled_refs.size() == 1);
    EXPECT(recompiled_refs.front().file_path == original_refs.front().file_path);
    EXPECT(migraphx::to_value(recompiled_refs) == migraphx::to_value(original_refs));
    EXPECT(migraphx::read_buffer(weights_file) == weights_bytes);
    auto q = migraphx::load(recompiled, {.finalize = false});
    EXPECT(recorded_device(*q.get_main_module()) == "dev_b");
    EXPECT(main_literals(q).front().get_external_data()->offset == original_refs.front().offset);
    EXPECT(main_literals(q).front() == weights);

    // Without reuse the weights are stored inline
    auto inline_file = (td.path / "inline.mxr").string();
    migraphx::save(p, inline_file);
    EXPECT(not migraphx::fs::exists(inline_file + ".data"));
    EXPECT(saved_external_refs(inline_file).empty());
    auto r = migraphx::load(inline_file, {.finalize = false});
    EXPECT(not main_literals(r).front().is_external());
    EXPECT(main_literals(r).front() == weights);
}

TEST_CASE(recompile_in_place_shares_external_weights)
{
    migraphx::tmp_dir td{"recompile-external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::shape s{migraphx::shape::float_type, {64}};
    auto weights = migraphx::generate_literal(s, 1);
    migraphx::program c;
    auto* mm = c.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto w   = mm->add_literal(weights);
    mm->add_return({mm->add_instruction(migraphx::make_op("add"), x, w)});
    current_device() = "dev_a";
    c.compile(make_recompile_target());

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(c, filename, options);
    auto original_refs = saved_external_refs(filename);
    auto weights_file  = td.path / original_refs.front().file_path;
    auto weights_bytes = migraphx::read_buffer(weights_file);
    {
        auto p           = migraphx::load(filename, {.finalize = false});
        current_device() = "dev_b";
        p.recompile(make_recompile_target());

        // New weights go to a new generation, and the one the program maps is kept
        migraphx::save(p, filename, options);
        auto written = td.path / saved_external_refs(filename).front().file_path;
        EXPECT(written != weights_file);
        EXPECT(migraphx::read_buffer(weights_file) == weights_bytes);

        // Saved with reuse, the unreferenced new generation is removed again
        migraphx::file_options reuse;
        reuse.reuse_external_data = true;
        migraphx::save(p, filename, reuse);
        EXPECT(not migraphx::fs::exists(written));
    }
    EXPECT(migraphx::read_buffer(weights_file) == weights_bytes);
    EXPECT(migraphx::to_value(saved_external_refs(filename)) == migraphx::to_value(original_refs));
    auto q = migraphx::load(filename, {.finalize = false});
    EXPECT(recorded_device(*q.get_main_module()) == "dev_b");
    EXPECT(main_literals(q).front() == weights);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
