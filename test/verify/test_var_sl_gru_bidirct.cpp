
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_var_sl_gru_bidirct : verify_program<test_var_sl_gru_bidirct>
{
    migraphx::program create_program() const
    {
        std::size_t batch_size  = 3;
        std::size_t seq_len     = 3;
        std::size_t hidden_size = 5;
        std::size_t input_size  = 8;
        std::size_t num_dirct   = 2;
        float clip              = 0.0f;

        migraphx::program p;
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 3 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
        migraphx::shape sl_shape{migraphx::shape::int32_type, {batch_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        std::vector<int> sl_data{2, 1, 3};
        auto sql = p.add_literal(migraphx::literal{sl_shape, sl_data});

        auto hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              sql,
                              ih);
        auto lho = p.add_instruction(migraphx::op::rnn_last_hs_output{}, hs);
        p.add_return({hs, lho});

        return p;
    }
    std::string section() const { return "rnn"; }
};


