
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/reduce_mean.hpp>

/**
 * @brief test_shape_alloc sets up a situation that could lead to an exception "convolution: Shapes 
 * are not in standard layout" if a "replace_allocate" compiler pass is not followed with
 *   "adjust_allocation".  The last transpose instruction generates a shape with a stride of 1 in 
 *   the 2nd index, a non-standard layout that should be reallocated by adjust_allocation.
 */
struct test_shape_alloc : verify_program<test_shape_alloc>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        /**
         * @brief 
        migraphx::shape s3{migraphx::shape::float_type, {1, 1, 2048, 1001}};
        auto x3 = mm->add_parameter("x3", s3);
main:@135 = @literal{ ... } -> float_type, {1001, 2048, 1, 1}, {2048, 1, 1, 1}  generate_literal

main:@413 = pointwise(main:@409,main:@394,main:@411,main:@412,main:@134), [main:pointwise146] -> float_type, {1, 2048, 7, 7}, {100352, 49, 7, 1}

main:@414 = transpose[permutation={0, 2, 3, 1}](main:@413) -> float_type, {1, 7, 7, 2048}, {100352, 7, 1, 49}

// see if a different instruction,m same shape, causes the same problem
main:@416 = reduce_mean[axes={1, 2}](main:@415) -> float_type, {1, 1, 1, 2048}, {2048, 2048, 2048, 1}
main:@417 = transpose[permutation={0, 3, 1, 2}](main:@416) -> float_type, {1, 2048, 1, 1}, {2048, 1, 2048, 2048}
// size of weights from 135
main:@419 = convolution[padding={0, 0, 0, 0},stride={1, 1},dilation={1, 1},group=1,padding_mode=0](main:@418,main:@135) -> float_type, {1, 1001, 1, 1}, {1001, 1, 1, 1}

         * 
         */
        migraphx::shape s3{migraphx::shape::float_type, {1001, 2048, 1, 1}, {2048, 1, 1, 1} };

        auto lit135 = migraphx::generate_literal (s3);
        auto main135 = mm->add_literal(lit135); 

        migraphx::shape s413{migraphx::shape::float_type, {1, 2048, 7, 7}};
        auto main413 = mm->add_parameter("x413", s413);

        auto main414 =  mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), main413);  //  -> float_type, {1, 7, 7, 2048}, {100352, 7, 1, 49}

        // this leads to the problem but it should happen also with any other ops besides reduce_mean
        auto main416 = mm->add_instruction(
            migraphx::make_op("reduce_mean", {{"axes", {1,2}}}), main414); //  -> float_type, {1, 1, 1, 2048}, {2048, 2048, 2048, 1}
        auto main417 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), main416);//  -> float_type, {1, 2048, 1, 1}, {2048, 1, 2048, 2048}

        auto convo_op419 = migraphx::make_op("convolution", 
                { {"padding", {0, 0, 0, 0}}, 
                {"group", 1},
                {"padding_mode", 0},           
                });
        mm->add_instruction(convo_op419, main417, main135);
        
        return p;
    }
};
