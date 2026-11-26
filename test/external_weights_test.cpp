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
#include <migraphx/external_data.hpp>
#include <migraphx/mapped_file.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/md5.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/program.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/tmp_dir.hpp>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include "test.hpp"

// The external references of the main module's literals in a saved msgpack program, in the
// order they were written
static std::vector<migraphx::external_data> saved_external_refs(const migraphx::fs::path& file)
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

// Each literal sits in the weights file at an aligned offset, in order, with the fewest zero
// padding bytes before it and none after the last
static void check_weights_layout(const std::vector<migraphx::external_data>& refs,
                                 const std::vector<migraphx::literal>& literals,
                                 const std::vector<char>& weights,
                                 std::size_t alignment)
{
    EXPECT(refs.size() == literals.size());
    EXPECT(not refs.empty());
    EXPECT(refs.front().offset == 0);
    std::size_t end = 0;
    migraphx::for_each(
        refs.begin(), refs.end(), literals.begin(), [&](const auto& ref, const auto& l) {
            EXPECT(ref.offset % alignment == 0);
            EXPECT(ref.offset >= end);
            EXPECT(ref.offset - end < alignment);
            EXPECT(ref.offset + ref.size <= weights.size());
            EXPECT(std::all_of(weights.begin() + end, weights.begin() + ref.offset, [](char c) {
                return c == 0;
            }));
            EXPECT(ref.size == l.get_shape().bytes());
            EXPECT(std::equal(l.data(), l.data() + ref.size, weights.begin() + ref.offset));
            EXPECT(ref.checksum == migraphx::md5(std::string_view{l.data(), ref.size}));
            end = ref.offset + ref.size;
        });
    EXPECT(weights.size() == end);
}

TEST_CASE(round_trip_with_threshold)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm   = p1.get_main_module();
    auto x     = mm->add_parameter("x", {migraphx::shape::float_type, {8, 8}});
    auto big   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 8}}));
    auto small = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8}}));
    auto add   = mm->add_instruction(migraphx::make_op("add"), x, big);
    auto mul   = mm->add_instruction(
        migraphx::make_op("mul"),
        add,
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 8}}}), small));
    mm->add_return({mul});

    migraphx::file_options options;
    options.external_weights_threshold = big->get_shape().bytes();
    migraphx::save(p1, filename, options);

    EXPECT(not migraphx::fs::exists(filename + ".data"));
    auto v = migraphx::from_msgpack(migraphx::read_buffer(filename));
    EXPECT(v.at("version").to<int>() == p1.get_program_file_version() + 1);
    auto refs = saved_external_refs(filename);
    EXPECT(refs.size() == 1);
    EXPECT(migraphx::is_weights_generation(td.path / "model.mxr.data",
                                           td.path / refs.front().file_path));
    EXPECT(migraphx::fs::exists(td.path / refs.front().file_path));
    EXPECT(refs.front().size == big->get_shape().bytes());

    migraphx::program p2 = migraphx::load(filename);
    EXPECT(p1.sort() == p2.sort());
    auto* mm2 = p2.get_main_module();
    EXPECT(std::count_if(mm2->begin(), mm2->end(), [](const migraphx::instruction& ins) {
               return ins.name() == "@literal" and ins.get_literal().is_external();
           }) == 1);
    auto ext = std::find_if(mm2->begin(), mm2->end(), [](const migraphx::instruction& ins) {
        return ins.name() == "@literal" and ins.get_literal().is_external();
    });
    EXPECT(ext->get_literal() == big->get_literal());
}

TEST_CASE(weights_file_override)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {64}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    options.external_weights_file      = (td.path / "custom.data").string();
    migraphx::save(p1, filename, options);

    EXPECT(not migraphx::fs::exists(td.path / "custom.data"));
    EXPECT(not migraphx::fs::exists(filename + ".data"));
    auto refs = saved_external_refs(filename);
    EXPECT(refs.size() == 1);
    EXPECT(
        migraphx::is_weights_generation(td.path / "custom.data", td.path / refs.front().file_path));
    EXPECT(migraphx::fs::exists(td.path / refs.front().file_path));
    migraphx::program p2 = migraphx::load(filename);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(default_save_unchanged)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l =
        mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {64 * 1024}}));
    mm->add_return({l});

    migraphx::save(p1, filename);

    EXPECT(not migraphx::fs::exists(filename + ".data"));
    auto v = migraphx::from_msgpack(migraphx::read_buffer(filename));
    EXPECT(v.at("version").to<int>() == p1.get_program_file_version());
    EXPECT(saved_external_refs(filename).empty());
    migraphx::program p2 = migraphx::load(filename);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(json_round_trip)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.json").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::half_type, {65}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.format                     = "json";
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);

    EXPECT(std::any_of(migraphx::fs::directory_iterator{td.path},
                       migraphx::fs::directory_iterator{},
                       [&](const migraphx::fs::directory_entry& e) {
                           return migraphx::is_weights_generation(td.path / "model.json.data",
                                                                  e.path());
                       }));
    EXPECT(migraphx::read_string(filename).find("\"external\"") != std::string::npos);
    migraphx::program p2 = migraphx::load(filename, options);
    EXPECT(p1.sort() == p2.sort());
    EXPECT(std::prev(p2.get_main_module()->end(), 2)->get_literal().is_external());
}

TEST_CASE(checksum_verification)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);

    auto weights_file = td.path / saved_external_refs(filename).front().file_path;
    auto weights      = migraphx::read_buffer(weights_file);
    weights.at(5)++;
    migraphx::write_buffer(weights_file, weights);

    migraphx::file_options verify;
    verify.verify_external_weights = true;
    EXPECT(test::throws([&] { migraphx::load(filename, verify); }));

    migraphx::program p2 = migraphx::load(filename);
    auto loaded          = std::prev(p2.get_main_module()->end(), 2)->get_literal();
    EXPECT(loaded.is_external());
    EXPECT(loaded != l->get_literal());
}

TEST_CASE(truncated_weights_file_throws)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);

    auto weights_file = td.path / saved_external_refs(filename).front().file_path;
    auto weights      = migraphx::read_buffer(weights_file);
    weights.resize(weights.size() / 2);
    migraphx::write_buffer(weights_file, weights);
    EXPECT(test::throws([&] { migraphx::load(filename); }));

    migraphx::fs::remove(weights_file);
    EXPECT(test::throws([&] { migraphx::load(filename); }));
}

TEST_CASE(reuse_keeps_original_reference)
{
    migraphx::tmp_dir td{"external-weights"};
    migraphx::fs::create_directories(td.path / "a");
    migraphx::fs::create_directories(td.path / "b");
    auto original = (td.path / "a" / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, original, options);
    auto original_refs = saved_external_refs(original);
    EXPECT(original_refs.size() == 1);

    migraphx::program p2 = migraphx::load(original);
    migraphx::file_options reuse;
    reuse.reuse_external_data = true;

    // Outside the original's directory the reference becomes absolute
    auto other = (td.path / "b" / "other.mxr").string();
    migraphx::save(p2, other, reuse);
    EXPECT(not migraphx::fs::exists(other + ".data"));
    auto other_refs = saved_external_refs(other);
    EXPECT(other_refs.size() == 1);
    EXPECT(migraphx::fs::path{other_refs.front().file_path}.is_absolute());
    EXPECT(migraphx::fs::equivalent(other_refs.front().file_path,
                                    td.path / "a" / original_refs.front().file_path));
    EXPECT(other_refs.front().offset == original_refs.front().offset);
    EXPECT(other_refs.front().size == original_refs.front().size);
    EXPECT(other_refs.front().checksum == original_refs.front().checksum);

    // Beside the original the reference stays relative
    auto sibling = (td.path / "a" / "sibling.mxr").string();
    migraphx::save(p2, sibling, reuse);
    auto sibling_refs = saved_external_refs(sibling);
    EXPECT(sibling_refs.size() == 1);
    EXPECT(sibling_refs.front().file_path == original_refs.front().file_path);

    // The absolute reference leaves other.mxr's directory, so loading it needs a/ allowed
    EXPECT(test::throws([&] { migraphx::load(other); }));
    migraphx::file_options allow;
    allow.external_data_allowed_paths = {(td.path / "a").string()};
    migraphx::program p3              = migraphx::load(other, allow);
    EXPECT(p1.sort() == p3.sort());
    migraphx::program p4 = migraphx::load(sibling);
    EXPECT(p1.sort() == p4.sort());
}

TEST_CASE(save_over_mapped_weights_writes_new_generation)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);
    auto first = td.path / saved_external_refs(filename).front().file_path;
    migraphx::fs::path second;
    {
        migraphx::program p2 = migraphx::load(filename);
        EXPECT(migraphx::mapped_file::is_mapped(first));
        migraphx::save(p2, filename, options);
        second = td.path / saved_external_refs(filename).front().file_path;
        EXPECT(second != first);
        // The replaced generation is still mapped by p2, so it is kept
        EXPECT(migraphx::fs::exists(first));
        EXPECT(p1.sort() == p2.sort());
    }
    EXPECT(not migraphx::mapped_file::is_mapped(first));
    migraphx::save(p1, filename, options);
    auto third = td.path / saved_external_refs(filename).front().file_path;
    EXPECT(third != second);
    EXPECT(not migraphx::fs::exists(second));
    EXPECT(migraphx::fs::exists(third));
    migraphx::program p3 = migraphx::load(filename);
    EXPECT(p1.sort() == p3.sort());
}

TEST_CASE(save_removes_replaced_generation)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);
    auto first = td.path / saved_external_refs(filename).front().file_path;
    migraphx::save(p1, filename, options);
    auto second = td.path / saved_external_refs(filename).front().file_path;
    EXPECT(first != second);
    EXPECT(not migraphx::fs::exists(first));
    EXPECT(std::count_if(migraphx::fs::directory_iterator{td.path},
                         migraphx::fs::directory_iterator{},
                         [&](const migraphx::fs::directory_entry& e) {
                             return migraphx::is_weights_generation(td.path / "model.mxr.data",
                                                                    e.path());
                         }) == 1);
    migraphx::program p2 = migraphx::load(filename);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(weights_file_same_as_program_throws)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    options.external_weights_file      = (td.path / "." / "model.mxr").string();
    EXPECT(test::throws([&] { migraphx::save(p1, filename, options); }));
    EXPECT(migraphx::fs::is_empty(td.path));
}

TEST_CASE(failed_publish_removes_new_generation)
{
    migraphx::tmp_dir td{"external-weights"};
    // A directory cannot be replaced by the program file
    auto filename = td.path / "model.mxr";
    migraphx::fs::create_directories(filename / "keep");
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    EXPECT(test::throws([&] { migraphx::save(p1, filename.string(), options); }));
    EXPECT(migraphx::fs::is_directory(filename / "keep"));
    EXPECT(std::distance(migraphx::fs::directory_iterator{td.path},
                         migraphx::fs::directory_iterator{}) == 1);
}

TEST_CASE(save_over_mapped_file_throws)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);
    auto weights_file  = td.path / saved_external_refs(filename).front().file_path;
    auto weights_bytes = migraphx::read_buffer(weights_file);

    migraphx::program p2 = migraphx::load(filename);
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::save(p2, weights_file.string()); },
                                             "while a loaded program maps it"));
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::save(p2, weights_file.string(), options); },
        "while a loaded program maps it"));
    EXPECT(migraphx::read_buffer(weights_file) == weights_bytes);

    // A save that only reuses the mapped data is still fine
    migraphx::file_options reuse;
    reuse.reuse_external_data = true;
    auto other                = (td.path / "other.mxr").string();
    migraphx::save(p2, other, reuse);
    EXPECT(saved_external_refs(other).front().file_path ==
           saved_external_refs(filename).front().file_path);
    migraphx::program p3 = migraphx::load(other);
    EXPECT(p1.sort() == p3.sort());
}

TEST_CASE(load_confines_external_data)
{
    migraphx::tmp_dir td{"external-weights"};
    migraphx::fs::create_directories(td.path / "a");
    migraphx::fs::create_directories(td.path / "b");
    auto original = (td.path / "a" / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, original, options);
    auto generation = saved_external_refs(original).front().file_path;

    auto v      = migraphx::from_msgpack(migraphx::read_buffer(original));
    auto& nodes = v.at("modules").at("main").at("nodes");
    auto* node  = std::find_if(nodes.begin(), nodes.end(), [](const migraphx::value& n) {
        return n.contains("literal") and n.at("literal").contains("external");
    });
    EXPECT(node != nodes.end());
    auto& file_path = (*node)["literal"]["external"]["file_path"];

    migraphx::file_options allow_a;
    allow_a.external_data_allowed_paths = {(td.path / "a").string()};

    auto escaped = (td.path / "b" / "escaped.mxr").string();
    file_path    = "../a/" + generation;
    migraphx::write_buffer(escaped, migraphx::to_msgpack(v));
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::load(escaped); }, "is outside"));
    EXPECT(p1.sort() == migraphx::load(escaped, allow_a).sort());

    auto absolute = (td.path / "b" / "absolute.mxr").string();
    file_path     = (td.path / "a" / generation).string();
    migraphx::write_buffer(absolute, migraphx::to_msgpack(v));
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::load(absolute); }, "is outside"));
    EXPECT(p1.sort() == migraphx::load(absolute, allow_a).sort());

#ifndef _WIN32
    auto linked = (td.path / "b" / "linked.mxr").string();
    migraphx::fs::create_symlink(td.path / "a" / generation, td.path / "b" / "link.data");
    file_path = "link.data";
    migraphx::write_buffer(linked, migraphx::to_msgpack(v));
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::load(linked); }, "is outside"));
    EXPECT(p1.sort() == migraphx::load(linked, allow_a).sort());
#endif

    migraphx::file_options missing_dir;
    missing_dir.external_data_allowed_paths = {(td.path / "missing").string()};
    EXPECT(test::throws([&] { migraphx::load(original, missing_dir); }));
}

TEST_CASE(verify_rejects_missing_checksum)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);
    auto v      = migraphx::from_msgpack(migraphx::read_buffer(filename));
    auto& nodes = v.at("modules").at("main").at("nodes");
    auto* node  = std::find_if(nodes.begin(), nodes.end(), [](const migraphx::value& n) {
        return n.contains("literal") and n.at("literal").contains("external");
    });
    EXPECT(node != nodes.end());
    (*node)["literal"]["external"]["checksum"] = "";
    migraphx::write_buffer(filename, migraphx::to_msgpack(v));

    EXPECT(p1.sort() == migraphx::load(filename).sort());
    migraphx::file_options verify;
    verify.verify_external_weights = true;
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::load(filename, verify); },
                                             "has no checksum"));
}

TEST_CASE(malformed_references_throw)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::shape s{migraphx::shape::float_type, {32}};
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal(s));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);
    const auto v = migraphx::from_msgpack(migraphx::read_buffer(filename));

    const std::size_t max = std::numeric_limits<std::size_t>::max();
    const std::vector<std::function<void(migraphx::value&)>> corruptions = {
        [&](migraphx::value& lit) {
            lit["shape"] = migraphx::to_value(migraphx::shape{std::vector<migraphx::shape>{s}});
        },
        [&](migraphx::value& lit) { lit["external"]["offset"] = max; },
        [&](migraphx::value& lit) { lit["external"]["size"] = max; },
        [&](migraphx::value& lit) { lit["external"]["size"] = s.bytes() - 4; },
        [&](migraphx::value& lit) { lit["external"]["file_path"] = ""; },
    };
    auto bad = (td.path / "bad.mxr").string();
    EXPECT(std::all_of(corruptions.begin(), corruptions.end(), [&](const auto& corrupt) {
        auto c      = v;
        auto& nodes = c.at("modules").at("main").at("nodes");
        auto* node  = std::find_if(nodes.begin(), nodes.end(), [](const migraphx::value& n) {
            return n.contains("literal") and n.at("literal").contains("external");
        });
        corrupt((*node)["literal"]);
        migraphx::write_buffer(bad, migraphx::to_msgpack(c));
        return test::throws([&] { migraphx::load(bad); });
    }));
}

TEST_CASE(program_file_versions)
{
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    auto version = p1.get_program_file_version();
    auto v       = p1.to_value();
    EXPECT(v.at("version").to<int>() == version);
    migraphx::program p2;
    p2.from_value(v);
    EXPECT(p1.sort() == p2.sort());

    v["version"] = version + 1;
    migraphx::program p3;
    p3.from_value(v);
    EXPECT(p1.sort() == p3.sort());

    v["version"] = version + 2;
    migraphx::program p4;
    EXPECT(test::throws([&] { p4.from_value(v); }));
}

TEST_CASE(external_version_only_when_referenced)
{
    migraphx::tmp_dir td{"external-weights"};
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    // Reuse without any external literal writes nothing external
    auto reused = (td.path / "reused.mxr").string();
    migraphx::file_options reuse;
    reuse.reuse_external_data = true;
    migraphx::save(p1, reused, reuse);
    EXPECT(not migraphx::fs::exists(reused + ".data"));
    EXPECT(saved_external_refs(reused).empty());
    auto v1 = migraphx::from_msgpack(migraphx::read_buffer(reused));
    EXPECT(v1.at("version").to<int>() == p1.get_program_file_version());

    // A threshold above every literal keeps them all inline
    auto inline_file = (td.path / "inline.mxr").string();
    migraphx::file_options options;
    options.external_weights_threshold = l->get_shape().bytes() + 1;
    migraphx::save(p1, inline_file, options);
    EXPECT(not migraphx::fs::exists(inline_file + ".data"));
    EXPECT(saved_external_refs(inline_file).empty());
    auto v2 = migraphx::from_msgpack(migraphx::read_buffer(inline_file));
    EXPECT(v2.at("version").to<int>() == p1.get_program_file_version());

    migraphx::program p2 = migraphx::load(inline_file);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(save_buffer_requires_weights_file)
{
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    EXPECT(test::throws([&] { migraphx::save_buffer(p1, options); }));
}

TEST_CASE(save_buffer_with_weights_file)
{
    migraphx::tmp_dir td{"external-weights"};
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    options.external_weights_file      = (td.path / "weights.data").string();
    options.external_data_path         = td.path.string();
    auto buffer                        = migraphx::save_buffer(p1, options);
    EXPECT(not migraphx::fs::exists(td.path / "weights.data"));
    EXPECT(std::any_of(migraphx::fs::directory_iterator{td.path},
                       migraphx::fs::directory_iterator{},
                       [&](const migraphx::fs::directory_entry& e) {
                           return migraphx::is_weights_generation(td.path / "weights.data",
                                                                  e.path());
                       }));
    migraphx::program p2 = migraphx::load_buffer(buffer, options);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(alignment_int8_padding)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l1  = mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {129}}, 1));
    auto l2  = mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {129}}, 2));
    auto l3  = mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {129}}, 3));
    mm->add_return({l1, l2, l3});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);

    // add_literal inserts at the front, so the module holds them in reverse
    auto refs = saved_external_refs(filename);
    check_weights_layout(refs,
                         {l3->get_literal(), l2->get_literal(), l1->get_literal()},
                         migraphx::read_buffer(td.path / refs.front().file_path),
                         migraphx::external_data_alignment);
}

TEST_CASE(alignment_half_odd_elements)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l1  = mm->add_literal(migraphx::generate_literal({migraphx::shape::half_type, {3, 5}}, 1));
    auto l2  = mm->add_literal(migraphx::generate_literal({migraphx::shape::half_type, {7}}, 2));
    auto l3  = mm->add_literal(migraphx::generate_literal({migraphx::shape::half_type, {1}}, 3));
    mm->add_return({l1, l2, l3});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);

    auto refs = saved_external_refs(filename);
    check_weights_layout(refs,
                         {l3->get_literal(), l2->get_literal(), l1->get_literal()},
                         migraphx::read_buffer(td.path / refs.front().file_path),
                         migraphx::external_data_alignment);
}

TEST_CASE(alignment_mixed_types)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l1  = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5}}, 1));
    auto l2  = mm->add_literal(migraphx::generate_literal({migraphx::shape::half_type, {3}}, 2));
    auto l3  = mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {7}}, 3));
    auto l4  = mm->add_literal(migraphx::generate_literal({migraphx::shape::double_type, {9}}, 4));
    mm->add_return({l1, l2, l3, l4});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);

    auto refs = saved_external_refs(filename);
    check_weights_layout(
        refs,
        {l4->get_literal(), l3->get_literal(), l2->get_literal(), l1->get_literal()},
        migraphx::read_buffer(td.path / refs.front().file_path),
        migraphx::external_data_alignment);
}

TEST_CASE(alignment_already_aligned_sizes)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l1  = mm->add_literal(migraphx::generate_literal(
        {migraphx::shape::int8_type, {migraphx::external_data_alignment}}, 1));
    auto l2  = mm->add_literal(migraphx::generate_literal(
        {migraphx::shape::int8_type, {2 * migraphx::external_data_alignment}}, 2));
    mm->add_return({l1, l2});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);

    auto refs    = saved_external_refs(filename);
    auto weights = migraphx::read_buffer(td.path / refs.front().file_path);
    check_weights_layout(
        refs, {l2->get_literal(), l1->get_literal()}, weights, migraphx::external_data_alignment);
    EXPECT(refs.at(1).offset == refs.at(0).offset + refs.at(0).size);
    EXPECT(weights.size() == 3 * migraphx::external_data_alignment);
}

TEST_CASE(alignment_custom)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l1  = mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {129}}, 1));
    auto l2  = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5}}, 2));
    auto l3  = mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {129}}, 3));
    mm->add_return({l1, l2, l3});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    options.external_weights_alignment = 4096;
    migraphx::save(p1, filename, options);

    auto refs = saved_external_refs(filename);
    check_weights_layout(refs,
                         {l3->get_literal(), l2->get_literal(), l1->get_literal()},
                         migraphx::read_buffer(td.path / refs.front().file_path),
                         4096);
    migraphx::program p2 = migraphx::load(filename);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(alignment_invalid_throws)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {32}}));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    options.external_weights_alignment = 0;
    EXPECT(test::throws([&] { migraphx::save(p1, filename, options); }));
    options.external_weights_alignment = 48;
    EXPECT(test::throws([&] { migraphx::save(p1, filename, options); }));
    EXPECT(not migraphx::fs::exists(filename));
    EXPECT(not migraphx::fs::exists(filename + ".data"));
}

TEST_CASE(loaded_literals_alias_aligned_mapping)
{
    migraphx::tmp_dir td{"external-weights"};
    auto filename = (td.path / "model.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l1  = mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {129}}, 1));
    auto l2  = mm->add_literal(migraphx::generate_literal({migraphx::shape::half_type, {3}}, 2));
    auto l3  = mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5}}, 3));
    auto l4  = mm->add_literal(migraphx::generate_literal({migraphx::shape::double_type, {3}}, 4));
    mm->add_return({l1, l2, l3, l4});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, filename, options);

    migraphx::program p2 = migraphx::load(filename);
    auto* mm2            = p2.get_main_module();
    EXPECT(std::count_if(mm2->begin(), mm2->end(), [](const migraphx::instruction& ins) {
               return ins.name() == "@literal";
           }) == 4);
    EXPECT(std::all_of(mm2->begin(), mm2->end(), [](const migraphx::instruction& ins) {
        if(ins.name() != "@literal")
            return true;
        const auto& l    = ins.get_literal();
        const auto& file = l.get_mapped_file();
        if(not l.is_external() or file == nullptr)
            return false;
        auto addr = reinterpret_cast<std::uintptr_t>(l.data());
        return addr % migraphx::external_data_alignment == 0 and l.data() >= file->data() and
               l.data() + l.get_shape().bytes() <= file->data() + file->size();
    }));
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(alignment_with_reuse)
{
    migraphx::tmp_dir td{"external-weights"};
    auto first = (td.path / "first.mxr").string();
    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto l   = mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {129}}, 1));
    mm->add_return({l});

    migraphx::file_options options;
    options.external_weights_threshold = 1;
    migraphx::save(p1, first, options);

    migraphx::program p2 = migraphx::load(first);
    auto* mm2            = p2.get_main_module();
    auto n1 = mm2->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {70}}, 2));
    auto n2 = mm2->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {5}}, 3));
    auto reused = std::prev(mm2->end(), 2);
    mm2->replace_return({reused, n1, n2});

    auto second                 = (td.path / "second.mxr").string();
    options.reuse_external_data = true;
    migraphx::save(p2, second, options);

    auto first_file = saved_external_refs(first).front().file_path;
    auto refs       = saved_external_refs(second);
    EXPECT(refs.size() == 3);
    std::vector<migraphx::external_data> written;
    std::copy_if(refs.begin(), refs.end(), std::back_inserter(written), [&](const auto& ref) {
        return migraphx::is_weights_generation(td.path / "second.mxr.data",
                                               td.path / ref.file_path);
    });
    EXPECT(std::count_if(refs.begin(), refs.end(), [&](const auto& ref) {
               return ref.file_path == first_file;
           }) == 1);
    check_weights_layout(written,
                         {n2->get_literal(), n1->get_literal()},
                         migraphx::read_buffer(td.path / written.front().file_path),
                         migraphx::external_data_alignment);

    migraphx::program p3 = migraphx::load(second);
    EXPECT(p2.sort() == p3.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
