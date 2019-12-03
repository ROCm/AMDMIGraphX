#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include "models.hpp"

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

migraphx::program alexnet(unsigned batch) // NOLINT(readability-function-size)
{
    migraphx::program p;
    auto m0 =
        p.add_parameter("0", migraphx::shape{migraphx::shape::float_type, {batch, 3, 224, 224}});
    auto mx0 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000}}, 0));
    auto mx1 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {1000, 4096}}, 1));
    auto mx2 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {4096}}, 2));
    auto mx3 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {4096, 4096}}, 3));
    auto mx4 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {4096}}, 4));
    auto mx5 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {4096, 9216}}, 5));
    auto mx6 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 6));
    auto mx7 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}}, 7));
    auto mx8 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {256}}, 8));
    auto mx9  = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {256, 384, 3, 3}}, 9));
    auto mx10 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {384}}, 10));
    auto mx11 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {384, 192, 3, 3}}, 11));
    auto mx12 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {192}}, 12));
    auto mx13 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {192, 64, 5, 5}}, 13));
    auto mx14 = p.add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {64}}, 14));
    auto mx15 = p.add_literal(migraphx::generate_literal(
        migraphx::shape{migraphx::shape::float_type, {64, 3, 11, 11}}, 15));
    migraphx::op::convolution convolution16;
    convolution16.padding  = {2, 2};
    convolution16.stride   = {4, 4};
    convolution16.dilation = {1, 1};
    convolution16.group    = 1;
    auto mx16              = p.add_instruction(convolution16, m0, mx15);
    migraphx::op::broadcast broadcast17;
    broadcast17.axis           = 1;
    broadcast17.broadcast_lens = {batch, 64, 55, 55};
    auto mx17                  = p.add_instruction(broadcast17, mx14);
    migraphx::op::add add18;
    auto mx18 = p.add_instruction(add18, mx16, mx17);
    migraphx::op::relu relu19;
    auto mx19 = p.add_instruction(relu19, mx18);
    migraphx::op::pooling pooling20;
    pooling20.mode    = "max";
    pooling20.padding = {0, 0};
    pooling20.stride  = {2, 2};
    pooling20.lengths = {3, 3};
    auto mx20         = p.add_instruction(pooling20, mx19);
    migraphx::op::convolution convolution21;
    convolution21.padding  = {2, 2};
    convolution21.stride   = {1, 1};
    convolution21.dilation = {1, 1};
    convolution21.group    = 1;
    auto mx21              = p.add_instruction(convolution21, mx20, mx13);
    migraphx::op::broadcast broadcast22;
    broadcast22.axis           = 1;
    broadcast22.broadcast_lens = {batch, 192, 27, 27};
    auto mx22                  = p.add_instruction(broadcast22, mx12);
    migraphx::op::add add23;
    auto mx23 = p.add_instruction(add23, mx21, mx22);
    migraphx::op::relu relu24;
    auto mx24 = p.add_instruction(relu24, mx23);
    migraphx::op::pooling pooling25;
    pooling25.mode    = "max";
    pooling25.padding = {0, 0};
    pooling25.stride  = {2, 2};
    pooling25.lengths = {3, 3};
    auto mx25         = p.add_instruction(pooling25, mx24);
    migraphx::op::convolution convolution26;
    convolution26.padding  = {1, 1};
    convolution26.stride   = {1, 1};
    convolution26.dilation = {1, 1};
    convolution26.group    = 1;
    auto mx26              = p.add_instruction(convolution26, mx25, mx11);
    migraphx::op::broadcast broadcast27;
    broadcast27.axis           = 1;
    broadcast27.broadcast_lens = {batch, 384, 13, 13};
    auto mx27                  = p.add_instruction(broadcast27, mx10);
    migraphx::op::add add28;
    auto mx28 = p.add_instruction(add28, mx26, mx27);
    migraphx::op::relu relu29;
    auto mx29 = p.add_instruction(relu29, mx28);
    migraphx::op::convolution convolution30;
    convolution30.padding  = {1, 1};
    convolution30.stride   = {1, 1};
    convolution30.dilation = {1, 1};
    convolution30.group    = 1;
    auto mx30              = p.add_instruction(convolution30, mx29, mx9);
    migraphx::op::broadcast broadcast31;
    broadcast31.axis           = 1;
    broadcast31.broadcast_lens = {batch, 256, 13, 13};
    auto mx31                  = p.add_instruction(broadcast31, mx8);
    migraphx::op::add add32;
    auto mx32 = p.add_instruction(add32, mx30, mx31);
    migraphx::op::relu relu33;
    auto mx33 = p.add_instruction(relu33, mx32);
    migraphx::op::convolution convolution34;
    convolution34.padding  = {1, 1};
    convolution34.stride   = {1, 1};
    convolution34.dilation = {1, 1};
    convolution34.group    = 1;
    auto mx34              = p.add_instruction(convolution34, mx33, mx7);
    migraphx::op::broadcast broadcast35;
    broadcast35.axis           = 1;
    broadcast35.broadcast_lens = {batch, 256, 13, 13};
    auto mx35                  = p.add_instruction(broadcast35, mx6);
    migraphx::op::add add36;
    auto mx36 = p.add_instruction(add36, mx34, mx35);
    migraphx::op::relu relu37;
    auto mx37 = p.add_instruction(relu37, mx36);
    migraphx::op::pooling pooling38;
    pooling38.mode    = "max";
    pooling38.padding = {0, 0};
    pooling38.stride  = {2, 2};
    pooling38.lengths = {3, 3};
    auto mx38         = p.add_instruction(pooling38, mx37);
    migraphx::op::flatten flatten39;
    flatten39.axis = 1;
    auto mx39      = p.add_instruction(flatten39, mx38);
    migraphx::op::identity identity40;
    auto mx40 = p.add_instruction(identity40, mx39);
    migraphx::op::transpose transpose41;
    transpose41.dims = {1, 0};
    auto mx41        = p.add_instruction(transpose41, mx5);
    migraphx::op::multibroadcast multibroadcast42;
    multibroadcast42.output_lens = {batch, 4096};
    auto mx42                    = p.add_instruction(multibroadcast42, mx4);
    migraphx::op::dot dot43;
    dot43.alpha = 1;
    dot43.beta  = 1;
    auto mx43   = p.add_instruction(dot43, mx40, mx41, mx42);
    migraphx::op::relu relu44;
    auto mx44 = p.add_instruction(relu44, mx43);
    migraphx::op::identity identity45;
    auto mx45 = p.add_instruction(identity45, mx44);
    migraphx::op::transpose transpose46;
    transpose46.dims = {1, 0};
    auto mx46        = p.add_instruction(transpose46, mx3);
    migraphx::op::multibroadcast multibroadcast47;
    multibroadcast47.output_lens = {batch, 4096};
    auto mx47                    = p.add_instruction(multibroadcast47, mx2);
    migraphx::op::dot dot48;
    dot48.alpha = 1;
    dot48.beta  = 1;
    auto mx48   = p.add_instruction(dot48, mx45, mx46, mx47);
    migraphx::op::relu relu49;
    auto mx49 = p.add_instruction(relu49, mx48);
    migraphx::op::transpose transpose50;
    transpose50.dims = {1, 0};
    auto mx50        = p.add_instruction(transpose50, mx1);
    migraphx::op::multibroadcast multibroadcast51;
    multibroadcast51.output_lens = {batch, 1000};
    auto mx51                    = p.add_instruction(multibroadcast51, mx0);
    migraphx::op::dot dot52;
    dot52.alpha = 1;
    dot52.beta  = 1;
    p.add_instruction(dot52, mx49, mx50, mx51);
    return p;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
