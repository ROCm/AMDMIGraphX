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
#include <migraphx/register_target.hpp>
#include <migraphx/load_save.hpp>
#include "test.hpp"
#include <migraphx/make_op.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/json.hpp>
#include <migraphx/file_buffer.hpp>

#include <cstdint>
#include <cstdio>
#include <limits>

static migraphx::program create_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::int32_type});
    auto two = mm->add_literal(2);
    auto add = mm->add_instruction(migraphx::make_op("add"), x, two);
    mm->add_return({add});
    return p;
}

TEST_CASE(as_value)
{
    migraphx::program p1 = create_program();
    migraphx::program p2;
    p2.from_value(p1.to_value());
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(as_msgpack)
{
    migraphx::file_options options;
    options.format           = "msgpack";
    migraphx::program p1     = create_program();
    std::vector<char> buffer = migraphx::save_buffer(p1, options);
    migraphx::program p2     = migraphx::load_buffer(buffer, options);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(as_json)
{
    migraphx::file_options options;
    options.format           = "json";
    migraphx::program p1     = create_program();
    std::vector<char> buffer = migraphx::save_buffer(p1, options);
    migraphx::program p2     = migraphx::load_buffer(buffer, options);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(as_file)
{
    std::string filename = "migraphx_program.mxr";
    migraphx::program p1 = create_program();
    migraphx::save(p1, filename);
    migraphx::program p2 = migraphx::load(filename);
    std::remove(filename.c_str());
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(compiled)
{
    migraphx::program p1 = create_program();
    p1.compile(migraphx::make_target("ref"));
    std::vector<char> buffer = migraphx::save_buffer(p1);
    migraphx::program p2     = migraphx::load_buffer(buffer);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unknown_format)
{
    migraphx::file_options options;
    options.format = "???";

    EXPECT(test::throws([&] { migraphx::save_buffer(create_program(), options); }));
    EXPECT(test::throws([&] { migraphx::load_buffer(std::vector<char>{}, options); }));
}

TEST_CASE(program_with_module)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {2, 3}};
    auto x = mm->add_parameter("x", sd);

    std::vector<float> one(sd.elements(), 1);
    std::vector<float> two(sd.elements(), 2);

    auto* then_smod = p.create_module("then_smod");
    auto l1         = then_smod->add_literal(migraphx::literal{sd, one});
    auto r1         = then_smod->add_instruction(migraphx::make_op("add"), x, l1);
    then_smod->add_return({r1});

    auto* else_smod = p.create_module("else_smod");
    auto l2         = else_smod->add_literal(migraphx::literal{sd, two});
    auto r2         = else_smod->add_instruction(migraphx::make_op("mul"), x, l2);
    else_smod->add_return({r2});

    migraphx::shape s_cond{migraphx::shape::bool_type, {1}};
    auto cond = mm->add_parameter("cond", s_cond);
    auto ret  = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_smod, else_smod});
    mm->add_return({ret});

    migraphx::program p1 = p;
    auto v               = p.to_value();
    auto v1              = p1.to_value();
    EXPECT(v == v1);

    std::stringstream ss;
    p.print_cpp(ss);
    std::stringstream ss1;
    p1.print_cpp(ss1);
    EXPECT(ss.str() == ss1.str());

    migraphx::program p2;
    p2.from_value(v);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(symbolic_shape_msgpack_roundtrip)
{
    using migraphx::shape;
    using dd = shape::dynamic_dimension;
    using migraphx::sym::lit;
    auto n = migraphx::sym::var("n", {1, 8});

    migraphx::program p;
    auto* mm = p.get_main_module();
    shape s{shape::float_type, {dd{n}, dd{lit(3)}, dd{lit(4)}}};
    auto x = mm->add_parameter("x", s);
    auto r = mm->add_instruction(migraphx::make_op("relu"), x);
    mm->add_return({r});

    migraphx::file_options options;
    options.format           = "msgpack";
    std::vector<char> buffer = migraphx::save_buffer(p, options);
    migraphx::program p2     = migraphx::load_buffer(buffer, options);
    EXPECT(p.sort() == p2.sort());
}

static migraphx::program create_program_with_debug_symbols()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3}});
    auto y   = mm->add_parameter("y", {migraphx::shape::float_type, {2, 3}});
    auto add = mm->add_instruction(migraphx::make_op("add"), x, y);
    mm->add_debug_symbols(add, {"onnx:Add_0", "onnx:Add_1"});
    auto relu = mm->add_instruction(migraphx::make_op("relu"), add);
    mm->add_debug_symbols(relu, {"onnx:Relu_0"});
    mm->add_return({relu});
    return p;
}

using symbol_map = std::map<std::string, std::set<std::string>>;

static symbol_map collect_debug_symbols(const migraphx::module& m)
{
    symbol_map result;
    for(auto ins : migraphx::iterator_for(m))
    {
        auto syms = ins->get_debug_symbols();
        if(not syms.empty())
            result[ins->name()] = std::move(syms);
    }
    return result;
}

TEST_CASE(debug_symbols_as_value)
{
    migraphx::program p1 = create_program_with_debug_symbols();
    migraphx::program p2;
    p2.from_value(p1.to_value());
    EXPECT(p1.sort() == p2.sort());

    auto syms1 = collect_debug_symbols(*p1.get_main_module());
    auto syms2 = collect_debug_symbols(*p2.get_main_module());
    EXPECT(syms1 == syms2);
}

TEST_CASE(debug_symbols_as_msgpack)
{
    migraphx::file_options options;
    options.format       = "msgpack";
    migraphx::program p1 = create_program_with_debug_symbols();
    auto buffer          = migraphx::save_buffer(p1, options);
    migraphx::program p2 = migraphx::load_buffer(buffer, options);
    EXPECT(p1.sort() == p2.sort());

    auto syms1 = collect_debug_symbols(*p1.get_main_module());
    auto syms2 = collect_debug_symbols(*p2.get_main_module());
    EXPECT(syms1 == syms2);
}

TEST_CASE(debug_symbols_as_json)
{
    migraphx::file_options options;
    options.format       = "json";
    migraphx::program p1 = create_program_with_debug_symbols();
    auto buffer          = migraphx::save_buffer(p1, options);
    migraphx::program p2 = migraphx::load_buffer(buffer, options);
    EXPECT(p1.sort() == p2.sort());

    auto syms1 = collect_debug_symbols(*p1.get_main_module());
    auto syms2 = collect_debug_symbols(*p2.get_main_module());
    EXPECT(syms1 == syms2);
}

TEST_CASE(debug_symbols_as_file)
{
    std::string filename = "migraphx_program_debug_symbols.mxr";
    migraphx::program p1 = create_program_with_debug_symbols();
    migraphx::save(p1, filename);
    migraphx::program p2 = migraphx::load(filename);
    std::remove(filename.c_str());
    EXPECT(p1.sort() == p2.sort());

    auto syms1 = collect_debug_symbols(*p1.get_main_module());
    auto syms2 = collect_debug_symbols(*p2.get_main_module());
    EXPECT(syms1 == syms2);
}

TEST_CASE(debug_symbols_compiled)
{
    migraphx::program p1 = create_program_with_debug_symbols();
    p1.compile(migraphx::make_target("ref"));
    auto buffer          = migraphx::save_buffer(p1);
    migraphx::program p2 = migraphx::load_buffer(buffer);
    EXPECT(p1.sort() == p2.sort());

    auto syms1 = collect_debug_symbols(*p1.get_main_module());
    auto syms2 = collect_debug_symbols(*p2.get_main_module());
    EXPECT(syms1 == syms2);
}

TEST_CASE(no_debug_symbols_roundtrip)
{
    migraphx::program p1 = create_program();
    migraphx::program p2;
    p2.from_value(p1.to_value());
    EXPECT(p1.sort() == p2.sort());

    auto syms2 = collect_debug_symbols(*p2.get_main_module());
    EXPECT(syms2.empty());
}

// The serialized program with its `version` field replaced, keeping the key's position.
static migraphx::value program_value_with_version(const migraphx::value& version)
{
    migraphx::value v = create_program().to_value();
    v.at("version")   = version;
    return v;
}

// The serialized program with `version` as the last top-level key instead of the first.
static migraphx::value program_value_version_last(const migraphx::value& version)
{
    migraphx::value v = create_program().to_value();
    migraphx::value result;
    for(const auto& entry : v)
    {
        if(entry.get_key() != "version")
            result[entry.get_key()] = entry.without_key();
    }
    result["version"] = version;
    return result;
}

static std::vector<char> raw_bytes(std::initializer_list<int> bytes)
{
    std::vector<char> result;
    for(int b : bytes)
        result.push_back(static_cast<char>(b));
    return result;
}

static void append(std::vector<char>& buffer, const std::vector<char>& tail)
{
    buffer.insert(buffer.end(), tail.begin(), tail.end());
}

// msgpack for the map entry "version" (a fixstr of 7 characters), without a value
static std::vector<char> version_key()
{
    return raw_bytes({0xa7, 'v', 'e', 'r', 's', 'i', 'o', 'n'});
}

static bool load_throws(const migraphx::value& v, const std::string& format, const std::string& msg)
{
    migraphx::file_options options;
    options.format = format;
    std::vector<char> buffer;
    if(format == "json")
    {
        auto json = migraphx::to_json_string(v);
        buffer.assign(json.begin(), json.end());
    }
    else
    {
        buffer = migraphx::to_msgpack(v);
    }
    return test::throws<migraphx::exception>([&] { migraphx::load_buffer(buffer, options); }, msg);
}

static bool from_value_throws(const migraphx::value& v, const std::string& msg)
{
    return test::throws<migraphx::exception>(
        [&] {
            migraphx::program p;
            p.from_value(v);
        },
        msg);
}

TEST_CASE(file_version_range)
{
    migraphx::program p;
    EXPECT(p.get_min_supported_program_file_version() >= 1);
    EXPECT(p.get_min_supported_program_file_version() <= p.get_program_file_version());
}

TEST_CASE(file_version_supported_loads)
{
    migraphx::program p1 = create_program();
    for(int version : {p1.get_min_supported_program_file_version(), p1.get_program_file_version()})
    {
        auto v = program_value_with_version(migraphx::value(version));
        migraphx::program p2;
        p2.from_value(v);
        EXPECT(p1.sort() == p2.sort());
        for(const std::string format : {"msgpack", "json"})
        {
            migraphx::file_options options;
            options.format           = format;
            std::vector<char> buffer = migraphx::save_buffer(p1, options);
            migraphx::program p3     = migraphx::load_buffer(buffer, options);
            EXPECT(p1.sort() == p3.sort());
        }
    }
}

TEST_CASE(file_version_too_old)
{
    migraphx::program p;
    int min = p.get_min_supported_program_file_version();
    int cur = p.get_program_file_version();
    // A version below 1 is invalid rather than old, covered by file_version_invalid_value.
    if(min > 1)
    {
        std::string file_version = "program file version: " + std::to_string(min - 1) + ",";
        std::string range =
            "supports program file versions " + std::to_string(min) + " to " + std::to_string(cur);
        for(const auto& version : {migraphx::value(static_cast<std::int64_t>(min - 1)),
                                   migraphx::value(static_cast<std::uint64_t>(min - 1))})
        {
            auto v = program_value_with_version(version);
            EXPECT(from_value_throws(v, "Program version mismatch"));
            EXPECT(from_value_throws(v, file_version));
            EXPECT(from_value_throws(v, range));
            EXPECT(from_value_throws(v, "installed MIGraphX ("));
            EXPECT(from_value_throws(v, "Regenerate the MXR file"));
            EXPECT(load_throws(v, "msgpack", "Regenerate the MXR file"));
            EXPECT(load_throws(v, "json", "Regenerate the MXR file"));
        }
    }
}

TEST_CASE(file_version_too_new)
{
    migraphx::program p;
    auto min     = static_cast<std::uint64_t>(p.get_min_supported_program_file_version());
    auto cur     = static_cast<std::uint64_t>(p.get_program_file_version());
    auto int_max = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    // The last three would wrap or truncate if the version were narrowed to an int first.
    std::vector<std::uint64_t> versions = {cur + 1,
                                           cur + 1000,
                                           int_max,
                                           int_max + 1,
                                           (std::uint64_t{1} << 32U) + cur,
                                           (std::uint64_t{1} << 32U) + min,
                                           std::numeric_limits<std::uint64_t>::max()};
    std::string range =
        "supports program file versions " + std::to_string(min) + " to " + std::to_string(cur);
    for(std::uint64_t version : versions)
    {
        std::vector<migraphx::value> values = {migraphx::value(version)};
        if(version <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            values.emplace_back(static_cast<std::int64_t>(version));
        for(const auto& value : values)
        {
            auto v = program_value_with_version(value);
            EXPECT(from_value_throws(v, "Program version mismatch"));
            EXPECT(from_value_throws(v, "program file version: " + std::to_string(version) + ","));
            EXPECT(from_value_throws(v, range));
            EXPECT(from_value_throws(v, "installed MIGraphX ("));
            EXPECT(from_value_throws(v, "Upgrade MIGraphX"));
            EXPECT(load_throws(v, "msgpack", "Upgrade MIGraphX"));
            EXPECT(load_throws(v, "json", "Upgrade MIGraphX"));
        }
    }
}

TEST_CASE(file_version_invalid_value)
{
    for(std::int64_t version :
        {std::int64_t{0}, std::int64_t{-1}, std::numeric_limits<std::int64_t>::min()})
    {
        auto v = program_value_with_version(migraphx::value(version));
        EXPECT(from_value_throws(v, "invalid `version` field"));
        EXPECT(load_throws(v, "msgpack", "invalid `version` field"));
        EXPECT(load_throws(v, "json", "invalid `version` field"));
    }
    auto v = program_value_with_version(migraphx::value(std::uint64_t{0}));
    EXPECT(from_value_throws(v, "invalid `version` field"));
}

TEST_CASE(file_version_invalid_type)
{
    std::vector<migraphx::value> versions = {migraphx::value("8"),
                                             migraphx::value(8.0),
                                             migraphx::value(8.9),
                                             migraphx::value(true),
                                             migraphx::value(false),
                                             migraphx::value(nullptr),
                                             migraphx::value(migraphx::value::array{}),
                                             migraphx::value(migraphx::value::object{})};
    for(const auto& version : versions)
    {
        auto v = program_value_with_version(version);
        EXPECT(from_value_throws(v, "invalid `version` field"));
        EXPECT(load_throws(v, "msgpack", "invalid `version` field"));
        EXPECT(load_throws(v, "json", "invalid `version` field"));
    }
}

TEST_CASE(file_version_missing)
{
    migraphx::value v = create_program().to_value();
    migraphx::value without_version;
    for(const auto& entry : v)
    {
        if(entry.get_key() != "version")
            without_version[entry.get_key()] = entry.without_key();
    }
    EXPECT(from_value_throws(without_version, "no `version` field"));
    EXPECT(load_throws(without_version, "msgpack", "no `version` field"));
    EXPECT(load_throws(without_version, "json", "no `version` field"));
}

TEST_CASE(file_version_top_level_not_an_object)
{
    std::vector<migraphx::value> tops = {
        migraphx::value(8),
        migraphx::value("version"),
        migraphx::value(nullptr),
        migraphx::value(migraphx::value::array{migraphx::value(8)})};
    for(const auto& top : tops)
    {
        EXPECT(from_value_throws(top, "no `version` field"));
        EXPECT(load_throws(top, "msgpack", "no `version` field"));
        EXPECT(load_throws(top, "json", "no `version` field"));
    }
}

TEST_CASE(file_version_failed_load_leaves_program_unchanged)
{
    migraphx::program p;
    auto v = program_value_with_version(migraphx::value(std::uint64_t{1000}));
    EXPECT(test::throws<migraphx::exception>([&] { p.from_value(v); }));
    EXPECT(p.get_main_module()->size() == 0);
    EXPECT(not p.is_compiled());
}

TEST_CASE(file_version_unsupported_from_file)
{
    std::string filename = "migraphx_program_unsupported_version.mxr";
    auto buffer          = migraphx::to_msgpack(program_value_with_version(migraphx::value(1000)));
    migraphx::write_buffer(filename, buffer.data(), buffer.size());
    bool failed = test::throws<migraphx::exception>([&] { migraphx::load(filename); },
                                                    "Program version mismatch");
    std::remove(filename.c_str());
    EXPECT(failed);
}

TEST_CASE(file_version_message_keeps_matching_phrase)
{
    auto v = program_value_with_version(migraphx::value(1000));
    EXPECT(from_value_throws(v, "Error: Program version mismatch. MXR file was created using"));
}

TEST_CASE(writer_emits_version_first)
{
    migraphx::value v = create_program().to_value();
    EXPECT(v.begin()->get_key() == "version");
    auto buffer = migraphx::to_msgpack(v);
    // A fixmap header (at most 15 top-level keys), then the 7 character key "version".
    EXPECT((buffer.at(0) & 0xf0) == 0x80);
    EXPECT(std::string(buffer.begin() + 1, buffer.begin() + 9) == "\xa7version");
}

TEST_CASE(msgpack_peek_version_position)
{
    int cur                             = migraphx::program{}.get_program_file_version();
    std::vector<migraphx::value> values = {program_value_with_version(migraphx::value(cur)),
                                           program_value_version_last(migraphx::value(cur))};
    for(const auto& v : values)
    {
        auto buffer = migraphx::to_msgpack(v);
        auto result = migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "version");
        EXPECT(bool{result});
        if(result)
            EXPECT(result->to<int>() == cur);
        EXPECT(not migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "no_such_key"));
    }
}

TEST_CASE(msgpack_peek_version_middle)
{
    // 0x83: map of 3 entries. Skips a string, an array of maps, then finds the key.
    std::vector<char> buffer = raw_bytes({0x83,
                                          0xa1,
                                          'a',
                                          0xa2,
                                          'h',
                                          'i',
                                          0xa1,
                                          'b',
                                          0x92,
                                          0x81,
                                          0xa1,
                                          'k',
                                          0xc4,
                                          0x02,
                                          0x00,
                                          0x01,
                                          0x90});
    append(buffer, version_key());
    append(buffer, raw_bytes({0x08}));
    auto result = migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "version");
    EXPECT(bool{result});
    if(result)
        EXPECT(result->to<int>() == 8);
}

TEST_CASE(msgpack_peek_scalar_kinds_match_decoder)
{
    // The scan must classify every scalar the way the full decoder does, since both feed the
    // same version check.
    std::vector<std::vector<char>> encodings = {
        raw_bytes({0x05}),                                                 // positive fixint
        raw_bytes({0xff}),                                                 // negative fixint
        raw_bytes({0xcc, 0x80}),                                           // uint8
        raw_bytes({0xcd, 0x01, 0x00}),                                     // uint16
        raw_bytes({0xce, 0x00, 0x01, 0x00, 0x00}),                         // uint32
        raw_bytes({0xcf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}), // uint64 max
        raw_bytes({0xd0, 0x05}),                                           // int8, positive
        raw_bytes({0xd0, 0xff}),                                           // int8, negative
        raw_bytes({0xd1, 0x00, 0x08}),                                     // int16, positive
        raw_bytes({0xd1, 0xff, 0xf8}),                                     // int16, negative
        raw_bytes({0xd2, 0x00, 0x00, 0x00, 0x08}),                         // int32, positive
        raw_bytes({0xd3, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}), // int64 min
        raw_bytes({0xca, 0x41, 0x00, 0x00, 0x00}),                         // float32 8.0
        raw_bytes({0xcb, 0x40, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}), // float64 8.0
        raw_bytes({0xa1, '8'}),                                            // fixstr "8"
        raw_bytes({0xd9, 0x01, '8'}),                                      // str8 "8"
        raw_bytes({0xc3}),                                                 // true
        raw_bytes({0xc2}),                                                 // false
        raw_bytes({0xc0})};                                                // nil
    for(const auto& encoding : encodings)
    {
        std::vector<char> buffer = raw_bytes({0x81});
        append(buffer, version_key());
        append(buffer, encoding);
        auto decoded = migraphx::from_msgpack(buffer);
        auto peeked  = migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "version");
        EXPECT(bool{peeked});
        if(peeked)
            EXPECT(peeked->without_key() == decoded.at("version").without_key());
    }
}

TEST_CASE(msgpack_peek_non_scalar_is_an_empty_array)
{
    std::vector<std::vector<char>> encodings = {raw_bytes({0x90}),
                                                raw_bytes({0x92, 0x01, 0x02}),
                                                raw_bytes({0x80}),
                                                raw_bytes({0x81, 0xa1, 'k', 0x01}),
                                                raw_bytes({0xc4, 0x01, 0x08}),
                                                raw_bytes({0x91, 0xc4, 0x01, 0x08})};
    for(const auto& encoding : encodings)
    {
        std::vector<char> buffer = raw_bytes({0x81});
        append(buffer, version_key());
        append(buffer, encoding);
        auto peeked = migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "version");
        EXPECT(bool{peeked});
        if(peeked)
        {
            EXPECT(peeked->is_array());
            EXPECT(peeked->empty());
        }
    }
}

TEST_CASE(msgpack_version_checked_before_decode)
{
    // {"version": 255, "x": <ext>}: the ext type makes the full decode throw.
    std::vector<char> buffer = raw_bytes({0x82});
    append(buffer, version_key());
    append(buffer, raw_bytes({0xcc, 0xff, 0xa1, 'x', 0xd4, 0x01, 0x00}));
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::from_msgpack(buffer); }, "EXT"));
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::load_buffer(buffer); },
                                             "Program version mismatch"));
}

TEST_CASE(msgpack_peek_gives_up_safely)
{
    std::vector<std::vector<char>> buffers;
    buffers.push_back({});
    buffers.push_back(raw_bytes({0x85}));
    buffers.push_back(raw_bytes({0xde, 0x00}));
    // a map that claims 2^32 - 1 entries
    buffers.push_back(raw_bytes({0xdf, 0xff, 0xff, 0xff, 0xff}));
    // a key string whose length runs past the end
    buffers.push_back(raw_bytes({0x81, 0xdb, 0xff, 0xff, 0xff, 0xff, 'v'}));
    // a value that is a bin whose length runs past the end
    buffers.push_back(raw_bytes({0x82, 0xa1, 'x', 0xc6, 0xff, 0xff, 0xff, 0xff, 0x00}));
    // top level is an array, a scalar, or a map with a non-string key
    buffers.push_back(raw_bytes({0x91, 0x01}));
    buffers.push_back(raw_bytes({0x08}));
    buffers.push_back(raw_bytes({0x81, 0x01, 0x01}));
    // 0xc1 is never used by msgpack
    buffers.push_back(raw_bytes({0x81, 0xa1, 'x', 0xc1}));
    // an ext value before the version
    buffers.push_back(raw_bytes(
        {0x82, 0xa1, 'x', 0xd4, 0x01, 0x00, 0xa7, 'v', 'e', 'r', 's', 'i', 'o', 'n', 0x08}));
    for(const auto& buffer : buffers)
    {
        EXPECT(not migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "version"));
        // The full path still decides: it throws as it did before the scan existed.
        EXPECT(test::throws([&] { migraphx::load_buffer(buffer); }));
    }
}

TEST_CASE(msgpack_peek_truncated_and_corrupted_buffers)
{
    int cur     = migraphx::program{}.get_program_file_version();
    auto buffer = migraphx::to_msgpack(program_value_version_last(migraphx::value(cur)));
    for(std::size_t n = 0; n <= buffer.size(); n++)
    {
        auto result = migraphx::msgpack_peek_field(buffer.data(), n, "version");
        // Only the complete buffer holds the last entry.
        EXPECT(bool{result} == (n == buffer.size()));
        if(result)
            EXPECT(result->to<int>() == cur);
    }
    for(std::size_t i = 0; i < buffer.size(); i++)
    {
        for(int byte : {0x00, 0xc1, 0xdf, 0xff})
        {
            auto corrupted = buffer;
            corrupted[i]   = static_cast<char>(byte);
            // No crash or overread; any answer is acceptable here.
            auto result =
                migraphx::msgpack_peek_field(corrupted.data(), corrupted.size(), "version");
            EXPECT(result.has_value() or not result.has_value());
        }
    }
}

TEST_CASE(msgpack_duplicate_version_keys)
{
    // The scan sees the first entry; the decoder keeps the last, which from_value checks.
    std::vector<char> buffer = raw_bytes({0x82});
    append(buffer, version_key());
    append(buffer, raw_bytes({0x08}));
    append(buffer, version_key());
    append(buffer, raw_bytes({0xcc, 0xff}));
    auto peeked = migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "version");
    EXPECT(bool{peeked});
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::load_buffer(buffer); },
                                             "Program version mismatch"));
}

TEST_CASE(msgpack_peek_key_must_match_exactly)
{
    // Keys that are a prefix of "version", extend it, or differ in case are other fields.
    std::vector<char> buffer =
        raw_bytes({0x86, 0xa6, 'v', 'e', 'r', 's',  'i',  'o',  0x01, 0xa8, 'v',
                   'e',  'r',  's', 'i', 'o', 'n',  's',  0x02, 0xa7, 'V',  'e',
                   'r',  's',  'i', 'o', 'n', 0x03, 0xa0, 0x04, 0xa1, 'v',  0x05});
    append(buffer, version_key());
    append(buffer, raw_bytes({0x08}));
    auto peeked = migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "version");
    EXPECT(bool{peeked});
    if(peeked)
        EXPECT(peeked->to<int>() == 8);
    // The same buffer has no field called "versionx", and the empty key is found by its own name.
    EXPECT(not migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "versionx"));
    auto empty = migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "");
    EXPECT(bool{empty});
    if(empty)
        EXPECT(empty->to<int>() == 4);
}

TEST_CASE(json_version_checked_before_conversion)
{
    // The conversion of {"bytes": "oops"} fails; the version error must come first.
    std::string text = R"({"version": 1000, "x": {"bytes": "oops"}})";
    migraphx::file_options options;
    options.format = "json";
    EXPECT(test::throws([&] { migraphx::from_json_string(text); }));
    EXPECT(test::throws<migraphx::exception>(
        [&] { migraphx::load_buffer(text.data(), text.size(), options); },
        "Program version mismatch"));
}

TEST_CASE(json_hook_is_called_with_the_top_level_field)
{
    std::string text = R"({"a": 1, "version": 8, "z": {"bytes": [1, 2]}})";
    int calls        = 0;
    migraphx::value seen;
    auto result = migraphx::from_json_string(
        text.data(), text.size(), "version", [&](const migraphx::value& field) {
            calls++;
            seen = field;
        });
    EXPECT(calls == 1);
    EXPECT(seen.without_key() == migraphx::value(std::uint64_t{8}));
    EXPECT(result.at("a").to<int>() == 1);
    EXPECT(result.at("z").is_binary());

    std::string without = R"({"a": 1})";
    calls               = 0;
    migraphx::from_json_string(
        without.data(), without.size(), "version", [&](const migraphx::value&) { calls++; });
    EXPECT(calls == 0);
    std::string array = "[1, 2]";
    migraphx::from_json_string(
        array.data(), array.size(), "version", [&](const migraphx::value&) { calls++; });
    EXPECT(calls == 0);
}

TEST_CASE(msgpack_peek_deeply_nested_value_before_version)
{
    // The decoder overflows the stack on nesting this deep, so the version has to be found, and
    // a too-new file rejected, without decoding and without recursing in the scan.
    std::vector<char> buffer = raw_bytes({0x82, 0xa1, 'x'});
    for(int i = 0; i < 100000; i++)
        buffer.push_back(static_cast<char>(0x91));
    buffer.push_back(static_cast<char>(0xc0));
    append(buffer, version_key());
    append(buffer, raw_bytes({0xcc, 0xff}));
    auto peeked = migraphx::msgpack_peek_field(buffer.data(), buffer.size(), "version");
    EXPECT(bool{peeked});
    if(peeked)
        EXPECT(peeked->to<int>() == 255);
    EXPECT(test::throws<migraphx::exception>([&] { migraphx::load_buffer(buffer); },
                                             "Program version mismatch"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
