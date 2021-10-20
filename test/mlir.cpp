#include <migraphx/mlir.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>

TEST_CASE(conv)
{
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
    auto w = m.add_parameter("w", {migraphx::shape::float_type, {2, 8, 3, 3}});
    m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto s = migraphx::dump_mlir(m);
    std::cout << s << std::endl;
    EXPECT(migraphx::contains(s, "migraphx.convolution"));
    EXPECT(not migraphx::contains(s, "migraphx.@param"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
