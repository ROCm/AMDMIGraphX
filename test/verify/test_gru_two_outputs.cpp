
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_gru_two_outputs : verify_program<test_gru_two_outputs>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 2;
        std::size_t seq_len     = 1;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 1;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        auto seq = p.add_parameter("seq", in_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto hs  = p.add_instruction(
            migraphx::op::gru{hidden_size, {}, migraphx::op::rnn_direction::forward, clip},
            seq,
            w,
            r);
        auto last_hs = p.add_instruction(migraphx::op::rnn_last_hs_output{}, hs);
        p.add_return({hs, last_hs});

        return p;
    }
    std::string section() const { return "rnn"; }
};


