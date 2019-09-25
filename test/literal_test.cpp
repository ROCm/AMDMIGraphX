
#include <migraphx/literal.hpp>
#include <sstream>
#include <string>
#include "test.hpp"

TEST_CASE(literal_test)
{
    EXPECT(migraphx::literal{1} == migraphx::literal{1});
    EXPECT(migraphx::literal{1} != migraphx::literal{2});
    EXPECT(migraphx::literal{} == migraphx::literal{});
    EXPECT(migraphx::literal{} != migraphx::literal{2});

    migraphx::literal l1{1};
    migraphx::literal l2 = l1; // NOLINT
    EXPECT(l1 == l2);
    EXPECT(l1.at<int>(0) == 1);
    EXPECT(!l1.empty());
    EXPECT(!l2.empty());

    migraphx::literal l3{};
    migraphx::literal l4{};
    EXPECT(l3 == l4);
    EXPECT(l3.empty());
    EXPECT(l4.empty());
}

TEST_CASE(literal_os1)
{
    migraphx::literal l{1};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str() == "1");
}

TEST_CASE(literal_os2)
{
    migraphx::literal l{};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str().empty());
}

TEST_CASE(literal_os3)
{
    migraphx::shape s{migraphx::shape::int64_type, {3}};
    migraphx::literal l{s, {1, 2, 3}};
    std::stringstream ss;
    ss << l;
    EXPECT(ss.str() == "1, 2, 3");
}

TEST_CASE(literal_visit_at)
{
    migraphx::literal x{1};
    bool visited = false;
    x.visit_at([&](int i) {
        visited = true;
        EXPECT(i == 1);
    });
    EXPECT(visited);
}

TEST_CASE(literal_visit)
{
    migraphx::literal x{1};
    migraphx::literal y{1};
    bool visited = false;
    x.visit([&](auto i) {
        y.visit([&](auto j) {
            visited = true;
            EXPECT(i == j);
        });
    });
    EXPECT(visited);
}

TEST_CASE(literal_visit_all)
{
    migraphx::literal x{1};
    migraphx::literal y{1};
    bool visited = false;
    migraphx::visit_all(x, y)([&](auto i, auto j) {
        visited = true;
        EXPECT(i == j);
    });
    EXPECT(visited);
}

TEST_CASE(literal_visit_mismatch_shape)
{
    migraphx::literal x{1};
    migraphx::shape s{migraphx::shape::int64_type, {3}};
    migraphx::literal y{s, {1, 2, 3}};
    bool visited = false;
    x.visit([&](auto i) {
        y.visit([&](auto j) {
            visited = true;
            EXPECT(i != j);
        });
    });
    EXPECT(visited);
}

TEST_CASE(literal_visit_all_mismatch_type)
{
    migraphx::shape s1{migraphx::shape::int32_type, {1}};
    migraphx::literal x{s1, {1}};
    migraphx::shape s2{migraphx::shape::int8_type, {1}};
    migraphx::literal y{s2, {1}};
    EXPECT(
        test::throws<migraphx::exception>([&] { migraphx::visit_all(x, y)([&](auto, auto) {}); }));
}

TEST_CASE(literal_visit_empty)
{
    migraphx::literal x{};
    EXPECT(test::throws([&] { x.visit([](auto) {}); }));
    EXPECT(test::throws([&] { x.visit_at([](auto) {}); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
