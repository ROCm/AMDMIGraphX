#include "verify_program.hpp"  
#include <migraphx/program.hpp>  
#include <migraphx/generate.hpp>  
#include <migraphx/make_op.hpp>  
  
template<int ChannelSize, int LrnSize>  
struct test_lrn : verify_program<test_lrn<ChannelSize, LrnSize>>  
{  
    migraphx::program create_program() const  
    {  
        migraphx::program p;  
        auto* mm = p.get_main_module();  
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, ChannelSize, 28, 28}});  
        mm->add_instruction(  
            migraphx::make_op("lrn",  
                              {{"alpha", 0.0001}, {"beta", 0.75}, {"bias", 1.0}, {"size", LrnSize}}),  
            x);  
        return p;  
    }  
};  

template struct test_lrn<32, 6>;  
template struct test_lrn<32, 5>;  
template struct test_lrn<31, 8>;
template struct test_lrn<31, 5>;
