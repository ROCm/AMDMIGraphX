
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_lstm_bidirct_3args_und : verify_program<test_lstm_bidirct_3args_und>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        auto seq = mm->add_parameter("seq", in_shape);
        auto w   = mm->add_parameter("w", w_shape);
        auto r   = mm->add_parameter("r", r_shape);
        auto und = mm->add_instruction(migraphx::op::undefined{});
        mm->add_instruction(
            migraphx::op::gru{hidden_size,
                              {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                              migraphx::op::rnn_direction::bidirectional,
                              clip},
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);

        return p;
    }
    std::string section() const { return "rnn"; }
};
