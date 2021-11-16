#include <migraphx/gpu/mlir.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <test.hpp>

using migraphx::trim;

std::string encode(std::string s)
{
    std::stringstream ss;
    bool prespace = false;
    for(auto c : s)
    {
        if(std::isspace(c))
        {
            if(not prespace)
                ss << "  ";
            prespace = true;
        }
        else if(std::isprint(c))
        {
            ss << c;
            prespace = false;
        }
    }
    return migraphx::trim(ss.str());
}

TEST_CASE(conv)
{
    const std::string mlir_output = R"__migraphx__(
module  {
  func @main(%arg0: tensor<1x8x4x4xf32>, %arg1: tensor<2x8x3x3xf32>) -> tensor<1x2x2x2xf32> {
    %0 = "migraphx.convolution"(%arg0, %arg1) {dilation = [1 : si64, 1 : si64], group = 1 : si64, padding = [0 : si64, 0 : si64], padding_mode = 0 : si64, stride = [1 : si64, 1 :
si64]} : (tensor<1x8x4x4xf32>, tensor<2x8x3x3xf32>) -> tensor<1x2x2x2xf32>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
    auto w = m.add_parameter("w", {migraphx::shape::float_type, {2, 8, 3, 3}});
    m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    EXPECT(encode(s) == encode(mlir_output));
}

TEST_CASE(conv_add_relu)
{
    const std::string mlir_output = R"__migraphx__(
module  {
  func @main(%arg0: tensor<1x8x4x4xf32>, %arg1: tensor<2x8x3x3xf32>, %arg2: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
    %0 = "migraphx.convolution"(%arg0, %arg1) {dilation = [1 : si64, 1 : si64], group = 1 : si64, padding = [0 : si64, 0 : si64], padding_mode = 0 : si64, stride = [1 : si64, 1 : si64]} : (tensor<1x8x4x4xf32>, tensor<2x8x3x3xf32>) -> tensor<1x2x2x2xf32>
    %1 = "migraphx.add"(%0, %arg2) : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %2 = "migraphx.relu"(%1) : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  }
}
)__migraphx__";
    migraphx::module m;
    auto x    = m.add_parameter("x", {migraphx::shape::float_type, {1, 8, 4, 4}});
    auto w    = m.add_parameter("w", {migraphx::shape::float_type, {2, 8, 3, 3}});
    auto b    = m.add_parameter("b", {migraphx::shape::float_type, {1, 2, 2, 2}});
    auto conv = m.add_instruction(migraphx::make_op("convolution"), x, w);
    auto add  = m.add_instruction(migraphx::make_op("add"), conv, b);
    m.add_instruction(migraphx::make_op("relu"), add);
    auto s = migraphx::gpu::dump_mlir(m);
    // Skip test if MLIR is not enabled
    if(s.empty())
        return;
    EXPECT(encode(s) == encode(mlir_output));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
