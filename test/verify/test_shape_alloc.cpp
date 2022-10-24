
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/reduce_mean.hpp>

struct test_shape_alloc : verify_program<test_shape_alloc>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 1, 1, 2048}};
        auto x = mm->add_parameter("x", s);
        /**
         * @brief main:@479 = allocate[shape=float_type, {1, 1, 1, 2048}, {2048, 2048, 2048, 1}] -> float_type, {1, 1, 1, 2048}, {2048, 2048, 2048, 1}
main:@480 = gpu::precompile_op[op=reduce_mean[axes={1, 2}]](main:@478,main:@479) -> float_type, {1, 1, 1, 2048}, {2048, 1, 1, 1}
main:@481 = transpose[permutation={0, 3, 1, 2}](main:@480) -> float_type, {1, 2048, 1, 1}, {2048, 1, 1, 1}
main:@482 = allocate[shape=int8_type, {0}, {1}] -> int8_type, {0}, {1}
main:@483 = allocate[shape=float_type, {1, 1001, 1, 1}, {1001, 1, 1, 1}] -> float_type, {1, 1001, 1, 1}, {1001, 1, 1, 1}
main:@484 = gpu::convolution[padding={0, 0, 0, 0},stride={1, 1},dilation={1, 1},group=1,padding_mode=1,solution_id=88](main:@481,main:@62,main:@482,main:@483) -> float_type, {1, 1001, 1, 1}, {1001, 1, 1, 1}

         * 
         */

        auto red_op = migraphx::make_op("reduce_mean", {{"axes", {1, 2}}}  );
        auto preco_op =  migraphx::make_op("gpu::precompile_op", {{"op", to_value(red_op)}});
        auto preco_instr = mm->add_instruction( preco_op, {x, x});


        auto tl1 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), preco_instr);

        migraphx::shape int8_shape{migraphx::shape::int8_type, {0}};
        migraphx::shape float_shape{migraphx::shape::float_type, {1, 1001, 1, 1}};
        auto alloc2 = mm->add_instruction(
            migraphx::make_op("allocate",  {{"shape", to_value(int8_shape)}}));
        auto alloc3 = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", to_value(float_shape)}}));

        auto convo_op = migraphx::make_op("gpu::convolution", 
                { {"padding", {0, 0, 0, 0}}, 
                {"stride", {1,1}}, 
                {"dilation", {1,1}}   ,
                {"group", 1},
                {"padding_mode", 1},
                {"solution_id", 88},                
                });
        // main:@62 = @literal{ ... } -> float_type, {1001, 2048, 1, 1}, {2048, 1, 1, 1}
        auto lit1 = mm->add_literal({migraphx::shape{migraphx::shape::float_type}, {1001, 2048, 1, 1}});

        mm->add_instruction(convo_op, tl1, lit1, alloc2, alloc3);
        return p;
    }
};
