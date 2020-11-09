
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/operators.hpp>

struct test_lstm_bidirct_hs : verify_program<test_lstm_bidirct_hs>
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
        migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
        migraphx::shape w_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, input_size}};
        migraphx::shape r_shape{migraphx::shape::float_type,
                                {num_dirct, 4 * hidden_size, hidden_size}};
        migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
        migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
        migraphx::shape sl_shape{migraphx::shape::int32_type, {batch_size}};

        auto seq  = p.add_parameter("seq", in_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", b_shape);
        auto ih   = p.add_parameter("ih", ih_shape);
        std::vector<int> sl_data{3, 2};
        auto sql = p.add_literal(migraphx::literal{migraphx::literal{sl_shape, sl_data}});

        p.add_instruction(migraphx::op::lstm{hidden_size,
                                             {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                             migraphx::op::rnn_direction::bidirectional,
                                             clip},
                          seq,
                          w,
                          r,
                          bias,
                          sql,
                          ih);

        return p;
    }
    std::string section() const { return "rnn"; }
};
