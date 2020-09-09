#include <migraphx/program.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/op/add.hpp>
#include "test.hpp"
#include <cstdio>

migraphx::program create_program()
{
    migraphx::program p;

    auto x   = p.add_parameter("x", {migraphx::shape::int32_type});
    auto two = p.add_literal(2);
    auto add = p.add_instruction(migraphx::op::add{}, x, two);
    p.add_return({add});
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
    std::string filename = "migraphx_program.dat";
    migraphx::program p1 = create_program();
    migraphx::save(p1, filename);
    migraphx::program p2 = migraphx::load(filename);
    std::remove(filename.c_str());
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(compiled)
{
    migraphx::program p1 = create_program();
    p1.compile(migraphx::cpu::target{});
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
