#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

TEST_CASE(rnn_forward)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{0.4691,
                              0.3185,
                              -0.2227,
                              0.4423,
                              -0.0609,
                              -0.2803,
                              0.1744,
                              0.3146,
                              0.4049,
                              -0.3973,
                              -0.0890,
                              -0.1636};

    std::vector<float> r_data{-0.0456,
                              0.1061,
                              0.1574,
                              -0.4928,
                              -0.4300,
                              -0.1909,
                              -0.0225,
                              -0.2668,
                              0.1840,
                              -0.4453,
                              -0.4896,
                              0.1302,
                              -0.0929,
                              0.3545,
                              -0.4981,
                              0.0616};

    std::vector<float> bias_data{
        -0.4938, 0.4355, -0.3186, 0.2094, 0.1037, -0.1071, 0.4504, -0.3990};
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    float clip = 0.0f;
    // concatenation of hidden states as program output
    {

        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027,
                                        0.03445704,
                                        0.19167931,
                                        -0.3946827,
                                        -0.30889652,
                                        -0.22276389,
                                        0.44193283,
                                        -0.16477929,
                                        -0.11893477};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // rnn last output as program output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn_direction::forward, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);

        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477};
        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }

    // multiple rnn_last_output operators
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn_direction::forward, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477};
        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }

    // 3 args
    {
        migraphx::program p;
        auto seq = p.add_literal(migraphx::literal{in_shape, input});
        auto w   = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r   = p.add_literal(migraphx::literal{r_shape, r_data});

        auto out_hs = p.add_instruction(
            migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn_direction::forward, clip},
            seq,
            w,
            r);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{
            0.2935145, -0.23719997, -0.31123261, -0.18357255, 0., 0., 0., 0.};
        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }

    // seq_len = 1
    {
        seq_len = 1;
        std::vector<float> input_1(seq_len * batch_size * input_size, 0);
        input_1[0] = input_1[1] = 1.0;
        migraphx::shape in_shape_1{migraphx::shape::float_type, {seq_len, batch_size, input_size}};

        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape_1, input_1});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(rnn_reverse)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{-0.0296,
                              -0.1341,
                              0.1761,
                              -0.2325,
                              -0.0717,
                              0.1852,
                              0.2720,
                              0.1471,
                              -0.1097,
                              0.3363,
                              -0.0587,
                              -0.2302};
    std::vector<float> r_data{0.2528,
                              -0.2333,
                              0.3973,
                              0.1593,
                              -0.0388,
                              0.1702,
                              0.3829,
                              -0.0712,
                              -0.1668,
                              0.3074,
                              -0.2854,
                              0.4049,
                              -0.3737,
                              -0.1051,
                              0.4482,
                              -0.2841};
    std::vector<float> bias_data{-0.3188, 0.1341, -0.4446, 0.1389, 0.3117, 0.3664, 0.2352, 0.2552};
    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);
    float clip = 0.0f;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    // concatenation of hidden states as program output
    {

        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(
            migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn_direction::reverse, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.29385301,
                                        0.16796815,
                                        0.51075965,
                                        0.40258689,
                                        -0.13818839,
                                        0.44124447,
                                        0.14365635,
                                        0.14803654,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // rnn last output as program output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn_direction::reverse, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);

        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{-0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};
        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }
}

TEST_CASE(rnn_bidirectional)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 2;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    std::vector<float> w_data{0.4691,  0.3185,  -0.2227, 0.4423,  -0.0609, -0.2803,
                              0.1744,  0.3146,  0.4049,  -0.3973, -0.0890, -0.1636,
                              -0.0296, -0.1341, 0.1761,  -0.2325, -0.0717, 0.1852,
                              0.2720,  0.1471,  -0.1097, 0.3363,  -0.0587, -0.2302};

    std::vector<float> r_data{-0.0456, 0.1061,  0.1574,  -0.4928, -0.4300, -0.1909, -0.0225,
                              -0.2668, 0.1840,  -0.4453, -0.4896, 0.1302,  -0.0929, 0.3545,
                              -0.4981, 0.0616,  0.2528,  -0.2333, 0.3973,  0.1593,  -0.0388,
                              0.1702,  0.3829,  -0.0712, -0.1668, 0.3074,  -0.2854, 0.4049,
                              -0.3737, -0.1051, 0.4482,  -0.2841};

    std::vector<float> bias_data{-0.4938,
                                 0.4355,
                                 -0.3186,
                                 0.2094,
                                 0.1037,
                                 -0.1071,
                                 0.4504,
                                 -0.3990,
                                 -0.3188,
                                 0.1341,
                                 -0.4446,
                                 0.1389,
                                 0.3117,
                                 0.3664,
                                 0.2352,
                                 0.2552};

    std::vector<float> input(seq_len * batch_size * input_size, 0);
    input[0] = input[1] = 1.0;
    std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    float clip = 0.0f;
    // concatenation of hidden state for program output
    {

        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        p.add_instruction(
            migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn_direction::bidirectional, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.37780784,  0.61055139,  0.55168478,  -0.5888475, -0.37144644, 0.31708236,
            0.13104209,  -0.18736027, -0.29385301, 0.16796815, 0.51075965,  0.40258689,
            -0.13818839, 0.44124447,  0.14365635,  0.14803654, 0.03445704,  0.19167931,
            -0.3946827,  -0.30889652, -0.22276389, 0.44193283, -0.16477929, -0.11893477,
            -0.0070999,  0.46251031,  -0.20639211, 0.37488942, -0.0070999,  0.46251031,
            -0.20639211, 0.37488942};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last rnn output for program output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hidden_size,
                                                {migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);

        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477,
                                                 -0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};

        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }

    // 4 args
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hidden_size,
                                                {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias);

        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{0.03445704,
                                                 0.19167931,
                                                 -0.3946827,
                                                 -0.30889652,
                                                 -0.22276389,
                                                 0.44193283,
                                                 -0.16477929,
                                                 -0.11893477,
                                                 -0.29385301,
                                                 0.16796815,
                                                 0.51075965,
                                                 0.40258689,
                                                 -0.13818839,
                                                 0.44124447,
                                                 0.14365635,
                                                 0.14803654};

        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }

    // 3 args
    {
        migraphx::program p;
        auto seq = p.add_literal(migraphx::literal{in_shape, input});
        auto w   = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r   = p.add_literal(migraphx::literal{r_shape, r_data});

        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip},
                          seq,
                          w,
                          r);
        p.compile(migraphx::cpu::target{});

        auto last_output = p.eval({});
        std::vector<float> last_output_data;
        last_output.visit([&](auto out) { last_output_data.assign(out.begin(), out.end()); });

        std::vector<float> last_output_data_gold{
            0.6570473,   0.36392266,  0.45342238,  -0.45127486, 0., 0., 0., 0.,
            -0.16225325, -0.29515147, 0.39617197,  0.27068236,  0., 0., 0., 0.,
            0.2935145,   -0.23719997, -0.31123261, -0.18357255, 0., 0., 0., 0.,
            0.,          0.,          0.,          0.,          0., 0., 0., 0.};

        EXPECT(migraphx::verify_range(last_output_data, last_output_data_gold));
    }

    // concatenation of hidden state for program output
    {
        seq_len = 1;
        std::vector<float> input_1(seq_len * batch_size * input_size, 0);
        input_1[0] = input_1[1] = 1.0;
        migraphx::shape in_shape_1{migraphx::shape::float_type, {seq_len, batch_size, input_size}};

        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape_1, input_1});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        p.add_instruction(
            migraphx::op::rnn{hidden_size, {}, migraphx::op::rnn_direction::bidirectional, clip},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.37780784,
                                        0.61055139,
                                        0.55168478,
                                        -0.5888475,
                                        -0.37144644,
                                        0.31708236,
                                        0.13104209,
                                        -0.18736027,
                                        -0.16915828,
                                        0.1938169,
                                        0.20667936,
                                        0.58609703,
                                        -0.0070999,
                                        0.46251031,
                                        -0.20639211,
                                        0.37488942};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_forward)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;
    // concatenation of hidden states for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.27298412, 0.42363745,  -0.09368783, 0.4823072,   -0.02183238, -0.6873896,
            0.16144305,  0.31932795,  0.6104771,   0.79759157,  -0.31791314, 0.5249062,
            0.08800987,  0.46404213,  -0.11872687, -0.26210734, 0.34448293,  -0.0176422,
            0.48523626,  0.60002893,  -0.3969709,  0.43360898,  0.35775262,  0.23280787,
            -0.52179873, -0.21944991, 0.4535257,   -0.13735442, 0.51757574,  0.50380427};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip,
                                                1},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // two rnn_last_output operators after gru
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip,
                                                1},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip,
                                                0},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.53291196,
                                        0.50160867,
                                        0.39010462,
                                        0.39292926,
                                        -0.5960838,
                                        -0.38451535,
                                        0.454239,
                                        -0.10620412,
                                        0.6014447,
                                        0.43445644};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_forward_args)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;

    // 3 args
    {
        migraphx::program p;
        auto seq = p.add_literal(migraphx::literal{in_shape, input});
        auto w   = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r   = p.add_literal(migraphx::literal{r_shape, r_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.114674, -0.129581,  -0.218156,  -0.140788,  -0.114242,
                                        -0.346569, 0.321367,   -0.0838253, 0.102097,   0.00232137,
                                        -0.149055, 0.0590743,  -0.0533094, -0.0446122, -0.112588,
                                        0.0153261, 0.168883,   -0.326836,  0.0843562,  0.160872,
                                        -0.232523, 0.00214573, 0.231693,   -0.160475,  -0.518952,
                                        0.0467166, 0.12327,    -0.374162,  0.137778,   0.251976};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 4 args (bias is used)
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.273619, 0.0931375, -0.104717,  0.0203752, -0.0797887,
                                        -0.493948, 0.472118,  -0.0336318, 0.332706,  0.0182268,
                                        -0.341684, 0.38063,   0.0589275,  0.2644,    -0.115737,
                                        -0.152324, 0.442277,  -0.201626,  0.408909,  0.12905,
                                        -0.416866, 0.377186,  0.32922,    0.162214,  -0.519973,
                                        -0.140072, 0.465076,  -0.229563,  0.500164,  0.195166};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 4 args (ih is used)
    {
        migraphx::program p;
        auto seq = p.add_literal(migraphx::literal{in_shape, input});
        auto w   = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r   = p.add_literal(migraphx::literal{r_shape, r_data});
        auto ih  = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto und = p.add_instruction(migraphx::op::undefined{});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          und,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.0801064, 0.27025,   -0.20704,   0.333579,   -0.0452438,
                                        -0.56265,   0.061061,  0.262172,   0.405193,   0.775226,
                                        -0.100683,  0.258729,  -0.0187297, 0.215815,   -0.108936,
                                        -0.0941018, 0.129665,  -0.159421,  0.190636,   0.597412,
                                        -0.197,     0.0885705, 0.269396,   -0.0414511, -0.515137,
                                        -0.03075,   0.158326,  -0.296488,  0.177983,   0.519498};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_forward_actv_funcs)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;

    // no activation function specified, so default is used.
    {
        migraphx::program p;
        auto seq       = p.add_literal(migraphx::literal{in_shape, input});
        auto w         = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r         = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = p.add_instruction(migraphx::op::undefined{});
        auto ih        = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = p.add_instruction(
            migraphx::op::gru{hidden_size, {}, migraphx::op::rnn_direction::forward, clip, 1},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.3969709,
                                        0.43360898,
                                        0.35775262,
                                        0.23280787,
                                        -0.52179873,
                                        -0.21944991,
                                        0.4535257,
                                        -0.13735442,
                                        0.51757574,
                                        0.50380427};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 1 activation function (sigmoid) specified
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.26905832, 0.5669211,  0.20464146, 0.67195725, 0.24752215,
                                        0.11411376, 0.12353572, 0.4245067,  0.73908687, 0.8644615,
                                        0.34754312, 0.61424744, 0.36769435, 0.6499579,  0.3168031,
                                        0.3296533,  0.3055136,  0.42514813, 0.6851256,  0.7967266,
                                        0.35652235, 0.6033026,  0.52634895, 0.5815402,  0.3001663,
                                        0.39814138, 0.4354002,  0.4310627,  0.6708563,  0.7509278};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 1 activation function (tanh) specified
    {
        migraphx::program p;
        auto seq       = p.add_literal(migraphx::literal{in_shape, input});
        auto w         = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r         = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = p.add_instruction(migraphx::op::undefined{});
        auto ih        = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = p.add_instruction(
            migraphx::op::gru{
                hidden_size, {migraphx::op::tanh{}}, migraphx::op::rnn_direction::forward, clip, 1},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.49333298,
                                        -0.06104589,
                                        0.5629142,
                                        -0.97955984,
                                        -0.9314696,
                                        -0.03033514,
                                        0.5280315,
                                        -0.27354342,
                                        0.65615714,
                                        0.53612584};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // seq length of 1
    {
        migraphx::program p;
        seq_len = 1;
        migraphx::shape in_shape_one{migraphx::shape::float_type,
                                     {seq_len, batch_size, input_size}};
        std::vector<float> input_one{-0.8432, -0.9887, 1.3041, -2.6430, -0.3306, -0.8504};
        auto seq  = p.add_literal(migraphx::literal{in_shape_one, input_one});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.27298412,
                                        0.42363745,
                                        -0.09368783,
                                        0.4823072,
                                        -0.02183238,
                                        -0.6873896,
                                        0.16144305,
                                        0.31932795,
                                        0.6104771,
                                        0.79759157};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_reverse)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3485,  -0.0378, -0.1782, 0.1416,  -0.3096, -0.2212, -0.3883, 0.1983,  -0.2418,
        0.1480,  -0.3255, 0.1359,  -0.3551, -0.3605, -0.3482, -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097, -0.1211, -0.0412, 0.1801,  0.1721,  -0.4327, -0.0498,
        0.2628,  -0.1573, -0.1577, 0.2759,  -0.2023, -0.1185, -0.2136, 0.1294,  -0.2331,
        0.0701,  0.4316,  0.0480,  0.0247,  -0.0166, -0.2729, 0.1712,  -0.3984, -0.3905};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        0.2848,  -0.2851, -0.3466, -0.1718, -0.1492, -0.0082, 0.2452,  -0.0401, 0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373, 0.3848,  -0.0130, -0.4339, 0.0406,  -0.1926, -0.1131,
        0.4285,  -0.0013, 0.2243,  0.2752,  0.1776,  -0.1720, 0.0822,  -0.0295, 0.1062,  -0.2721,
        -0.2736, -0.1826, 0.3541,  -0.4259, 0.2188,  0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744, 0.2122,  -0.4346, 0.2760,  0.4076,  0.1183,  -0.1500, -0.1704, 0.3090,  -0.0706,
        -0.2442, 0.3021,  0.1680,  0.0783,  -0.3754, -0.3469, -0.2972, -0.0170, 0.4143,  0.3801,
        0.3852,  -0.1170, -0.2937, 0.2979,  -0.1357, 0.4257,  0.3884,  -0.2916, 0.1071,  0.0934,
        0.3645,  -0.4310, -0.3480, 0.0702,  -0.1558};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        0.0560,  0.0310, -0.1669, -0.0781, 0.1793, -0.1758, 0.3173,  -0.1650, -0.3732, 0.2946,
        -0.0912, 0.3118, 0.1391,  0.2755,  0.2695, -0.1059, -0.2357, 0.3629,  -0.2534, -0.0494,
        0.0556,  0.0881, -0.2592, -0.2213, 0.2310, -0.4044, 0.1801,  0.1438,  0.3108,  -0.3607};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{
        -0.0468, 0.5691, -0.0882, 0.8340, 0.1483, -0.3902, -0.5348, 0.4178, 1.0175, 0.9212};
    float clip = 0.0f;

    // concatenation of hidden states for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::reverse,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.263403, 0.317655,  -0.00634162, 0.200443, -0.349125,
                                        -0.600874, 0.542386,  -0.0856531,  0.55703,  0.54711,
                                        -0.276245, 0.521348,  0.302874,    0.394353, -0.334369,
                                        -0.187861, 0.213553,  -0.0708377,  0.545435, 0.654301,
                                        -0.329512, 0.476095,  0.284044,    0.392077, -0.369226,
                                        -0.3275,   -0.027301, 0.143774,    0.655686, 0.782831};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip,
                                                1},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.263403,
                                        0.317655,
                                        -0.00634162,
                                        0.200443,
                                        -0.349125,
                                        -0.600874,
                                        0.542386,
                                        -0.0856531,
                                        0.55703,
                                        0.54711};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip,
                                                0},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.388654,
                                        0.384975,
                                        0.0179455,
                                        0.350101,
                                        -0.456872,
                                        -0.690085,
                                        0.534512,
                                        -0.0558191,
                                        0.646604,
                                        0.463943};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // no activation function specified, so default is used.
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(
            migraphx::op::gru{hidden_size, {}, migraphx::op::rnn_direction::reverse, clip, 1},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.263403, 0.317655,  -0.00634162, 0.200443, -0.349125,
                                        -0.600874, 0.542386,  -0.0856531,  0.55703,  0.54711,
                                        -0.276245, 0.521348,  0.302874,    0.394353, -0.334369,
                                        -0.187861, 0.213553,  -0.0708377,  0.545435, 0.654301,
                                        -0.329512, 0.476095,  0.284044,    0.392077, -0.369226,
                                        -0.3275,   -0.027301, 0.143774,    0.655686, 0.782831};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // seq length of 1
    {
        migraphx::program p;
        seq_len = 1;
        migraphx::shape in_shape_one{migraphx::shape::float_type,
                                     {seq_len, batch_size, input_size}};
        std::vector<float> input_one{-0.8432, -0.9887, 1.3041, -2.6430, -0.3306, -0.8504};
        auto seq  = p.add_literal(migraphx::literal{in_shape_one, input_one});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::reverse,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.272984,
                                        0.423637,
                                        -0.0936878,
                                        0.482307,
                                        -0.0218324,
                                        -0.68739,
                                        0.161443,
                                        0.319328,
                                        0.610477,
                                        0.797592};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_bidirectional)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3809,  0.4283,  0.2294,  -0.1018, -0.1226, -0.0037, 0.2449,  -0.2712, -0.1418,
        0.1363,  -0.3453, -0.0693, -0.2281, 0.2699,  -0.2024, -0.3085, -0.3338, 0.4109,
        0.2605,  -0.1019, -0.2813, 0.3323,  -0.1590, 0.0788,  -0.3535, 0.0397,  0.2732,
        0.2906,  0.0519,  0.3617,  -0.2664, 0.1441,  0.0464,  -0.1057, 0.2204,  -0.3294,
        0.3670,  0.1411,  0.3852,  0.3572,  0.3918,  0.0483,  -0.3906, -0.2841, -0.2778,

        -0.4272, 0.2335,  -0.1811, -0.3885, -0.1279, 0.1000,  0.0206,  -0.3284, -0.0353,
        0.1197,  0.1190,  0.3862,  0.0965,  -0.0492, 0.2657,  -0.1430, 0.0597,  0.1408,
        -0.0315, 0.1248,  0.0751,  0.3838,  0.3020,  0.0515,  0.2375,  -0.4255, 0.1714,
        -0.0432, 0.3447,  -0.2441, -0.3989, -0.3428, -0.4204, -0.4080, -0.2683, -0.0996,
        -0.1685, -0.0532, -0.1258, 0.1663,  -0.3526, -0.3915, -0.1721, 0.1292,  -0.2279};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        -0.2683, 0.0699,  -0.4021, -0.1379, 0.0042,  -0.2447, 0.4006,  0.0270,  -0.0446, 0.1063,
        0.1381,  0.1310,  -0.3596, 0.3869,  0.3929,  0.2750,  0.0890,  0.3069,  -0.1691, -0.2194,
        -0.1066, 0.3187,  -0.4369, -0.0603, -0.0834, -0.1182, -0.2047, 0.3253,  -0.2931, 0.2082,
        0.0424,  0.1111,  -0.2773, -0.0279, -0.0869, 0.1413,  -0.4227, -0.3672, 0.4137,  0.0609,
        0.4223,  -0.4032, 0.2945,  0.3600,  0.3345,  -0.3880, -0.0192, -0.0090, -0.2648, 0.4339,
        -0.0155, 0.4437,  -0.1766, 0.1957,  0.2475,  0.3773,  -0.2710, 0.3289,  -0.2077, -0.2534,
        -0.0832, -0.1632, 0.0728,  0.2520,  0.4153,  0.1659,  -0.4342, 0.0541,  0.1812,  -0.2305,
        0.4440,  0.0946,  0.0410,  -0.4381, -0.3161, 0.3906,  -0.3958, -0.4238, 0.1975,  0.3440,
        0.1437,  -0.0568, 0.1492,  -0.4248, -0.3304, 0.2786,  -0.1328, -0.3740, -0.3566, 0.3074,
        0.0924,  0.2684,  -0.1527, 0.1826,  0.2424,  0.2002,  0.3479,  -0.1089, 0.3472,  -0.3677,
        -0.4231, -0.0798, -0.3709, 0.3924,  0.2774,  -0.3690, -0.0233, 0.2845,  0.1969,  0.1618,
        -0.3742, -0.3619, 0.2925,  -0.1838, -0.1495, -0.3747, 0.0341,  -0.4243, -0.0732, -0.3997,
        0.2139,  0.2425,  0.4171,  -0.3358, 0.3534,  0.0938,  -0.0582, -0.2681, -0.4293, 0.1027,
        0.4101,  0.2641,  -0.4110, -0.1681, 0.3582,  -0.2089, 0.0852,  0.0963,  0.3866,  0.1955,
        -0.2174, 0.1996,  -0.2252, 0.1748,  0.1833,  -0.3155, 0.2567,  -0.4387, 0.3402,  0.0599};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        -0.1582, -0.0826, 0.4008,  0.0118,  0.2511,  0.1900,  -0.2838, 0.2549,  -0.2484, 0.2363,
        -0.4083, -0.0295, -0.1161, 0.1211,  0.2509,  -0.1414, -0.2628, -0.2992, 0.1517,  0.1817,
        -0.2783, 0.3183,  -0.1629, -0.3108, -0.3418, 0.0411,  0.2203,  0.2187,  -0.2990, -0.0416,
        0.0209,  -0.1024, 0.4443,  -0.4420, -0.0330, -0.3591, -0.2990, 0.2167,  0.1395,  0.2317,
        0.1318,  0.1909,  -0.3615, 0.1953,  -0.2582, -0.2217, 0.3723,  0.1458,  0.2630,  -0.0377,
        0.1754,  0.0800,  -0.3964, -0.3247, 0.4219,  -0.0900, 0.3553,  0.2614,  -0.1298, -0.1124};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{-0.0468, 0.5691,  -0.0882, 0.8340,  0.1483, -0.3902, -0.5348,
                               0.4178,  1.0175,  0.9212,  -0.0468, 0.5691, -0.0882, 0.8340,
                               0.1483,  -0.3902, -0.5348, 0.4178,  1.0175, 0.9212};

    float clip = 0.0f;

    // concatenation of hidden states for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352243, 0.0146756,  0.00570925, 0.152446,  0.208683,   0.214342,   -0.0454273,
            -0.135177, -0.0800739, 0.903659,   0.0248217, 0.435231,   -0.144448,  0.101531,
            -0.111305, 0.381317,   0.468983,   0.230557,  0.348021,   0.180229,   -0.0930435,
            0.174108,  -0.063834,  0.0909285,  0.22759,   -0.221983,  -0.139656,  -0.0938906,
            -0.247681, 0.69647,    -0.159396,  0.299061,  -0.116652,  0.238649,   0.109945,
            0.192866,  0.307073,   0.191113,   0.658287,  -0.0340374, -0.0959787, 0.0794681,
            0.241526,  0.321104,   0.00693533, -0.311839, -0.12802,   -0.16643,   -0.393849,
            0.648851,  -0.395918,  0.231694,   -0.160503, 0.383289,   0.0879262,  -0.0254665,
            0.079043,  0.322652,   0.752701,   0.243775};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip,
                                                1},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.0959787, 0.0794681, 0.241526,  0.321104,  0.00693533,
                                        -0.311839,  -0.12802,  -0.16643,  -0.393849, 0.648851,
                                        0.0248217,  0.435231,  -0.144448, 0.101531,  -0.111305,
                                        0.381317,   0.468983,  0.230557,  0.348021,  0.180229};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // last output for output, linear_before_reset = 0
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs =
            p.add_instruction(migraphx::op::gru{hidden_size,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip,
                                                0},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.09280921, 0.18506107, 0.32247013, 0.17034212, -0.00115255, -0.29865006, -0.04513004,
            -0.10688055, -0.4767866, 0.6317833,  0.00286336, 0.53692746,  -0.00617076, 0.04564289,
            -0.18030001, 0.39584228, 0.53879917, 0.384983,   0.2759448,   0.11611474};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_bidirectional_args)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3809,  0.4283,  0.2294,  -0.1018, -0.1226, -0.0037, 0.2449,  -0.2712, -0.1418,
        0.1363,  -0.3453, -0.0693, -0.2281, 0.2699,  -0.2024, -0.3085, -0.3338, 0.4109,
        0.2605,  -0.1019, -0.2813, 0.3323,  -0.1590, 0.0788,  -0.3535, 0.0397,  0.2732,
        0.2906,  0.0519,  0.3617,  -0.2664, 0.1441,  0.0464,  -0.1057, 0.2204,  -0.3294,
        0.3670,  0.1411,  0.3852,  0.3572,  0.3918,  0.0483,  -0.3906, -0.2841, -0.2778,

        -0.4272, 0.2335,  -0.1811, -0.3885, -0.1279, 0.1000,  0.0206,  -0.3284, -0.0353,
        0.1197,  0.1190,  0.3862,  0.0965,  -0.0492, 0.2657,  -0.1430, 0.0597,  0.1408,
        -0.0315, 0.1248,  0.0751,  0.3838,  0.3020,  0.0515,  0.2375,  -0.4255, 0.1714,
        -0.0432, 0.3447,  -0.2441, -0.3989, -0.3428, -0.4204, -0.4080, -0.2683, -0.0996,
        -0.1685, -0.0532, -0.1258, 0.1663,  -0.3526, -0.3915, -0.1721, 0.1292,  -0.2279};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        -0.2683, 0.0699,  -0.4021, -0.1379, 0.0042,  -0.2447, 0.4006,  0.0270,  -0.0446, 0.1063,
        0.1381,  0.1310,  -0.3596, 0.3869,  0.3929,  0.2750,  0.0890,  0.3069,  -0.1691, -0.2194,
        -0.1066, 0.3187,  -0.4369, -0.0603, -0.0834, -0.1182, -0.2047, 0.3253,  -0.2931, 0.2082,
        0.0424,  0.1111,  -0.2773, -0.0279, -0.0869, 0.1413,  -0.4227, -0.3672, 0.4137,  0.0609,
        0.4223,  -0.4032, 0.2945,  0.3600,  0.3345,  -0.3880, -0.0192, -0.0090, -0.2648, 0.4339,
        -0.0155, 0.4437,  -0.1766, 0.1957,  0.2475,  0.3773,  -0.2710, 0.3289,  -0.2077, -0.2534,
        -0.0832, -0.1632, 0.0728,  0.2520,  0.4153,  0.1659,  -0.4342, 0.0541,  0.1812,  -0.2305,
        0.4440,  0.0946,  0.0410,  -0.4381, -0.3161, 0.3906,  -0.3958, -0.4238, 0.1975,  0.3440,
        0.1437,  -0.0568, 0.1492,  -0.4248, -0.3304, 0.2786,  -0.1328, -0.3740, -0.3566, 0.3074,
        0.0924,  0.2684,  -0.1527, 0.1826,  0.2424,  0.2002,  0.3479,  -0.1089, 0.3472,  -0.3677,
        -0.4231, -0.0798, -0.3709, 0.3924,  0.2774,  -0.3690, -0.0233, 0.2845,  0.1969,  0.1618,
        -0.3742, -0.3619, 0.2925,  -0.1838, -0.1495, -0.3747, 0.0341,  -0.4243, -0.0732, -0.3997,
        0.2139,  0.2425,  0.4171,  -0.3358, 0.3534,  0.0938,  -0.0582, -0.2681, -0.4293, 0.1027,
        0.4101,  0.2641,  -0.4110, -0.1681, 0.3582,  -0.2089, 0.0852,  0.0963,  0.3866,  0.1955,
        -0.2174, 0.1996,  -0.2252, 0.1748,  0.1833,  -0.3155, 0.2567,  -0.4387, 0.3402,  0.0599};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        -0.1582, -0.0826, 0.4008,  0.0118,  0.2511,  0.1900,  -0.2838, 0.2549,  -0.2484, 0.2363,
        -0.4083, -0.0295, -0.1161, 0.1211,  0.2509,  -0.1414, -0.2628, -0.2992, 0.1517,  0.1817,
        -0.2783, 0.3183,  -0.1629, -0.3108, -0.3418, 0.0411,  0.2203,  0.2187,  -0.2990, -0.0416,
        0.0209,  -0.1024, 0.4443,  -0.4420, -0.0330, -0.3591, -0.2990, 0.2167,  0.1395,  0.2317,
        0.1318,  0.1909,  -0.3615, 0.1953,  -0.2582, -0.2217, 0.3723,  0.1458,  0.2630,  -0.0377,
        0.1754,  0.0800,  -0.3964, -0.3247, 0.4219,  -0.0900, 0.3553,  0.2614,  -0.1298, -0.1124};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{-0.0468, 0.5691,  -0.0882, 0.8340,  0.1483, -0.3902, -0.5348,
                               0.4178,  1.0175,  0.9212,  -0.0468, 0.5691, -0.0882, 0.8340,
                               0.1483,  -0.3902, -0.5348, 0.4178,  1.0175, 0.9212};

    float clip = 0.0f;

    // 3 args
    {
        migraphx::program p;
        auto seq = p.add_literal(migraphx::literal{in_shape, input});
        auto w   = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r   = p.add_literal(migraphx::literal{r_shape, r_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip,
                                            0},
                          seq,
                          w,
                          r);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0863793,  -0.227845,  0.0283059, -0.258645, 0.14187,    0.43541,     0.190748,
            -0.530196,  -0.440444,  0.293767,  0.0402142, 0.0788687,  -0.013,      -0.233298,
            -0.0739615, 0.467104,   0.446285,  0.306097,  0.125636,   0.272524,    0.0949838,
            0.0522264,  -0.0872712, -0.084203, 0.140013,  0.12739,    -0.0111171,  -0.431119,
            -0.468382,  0.388067,   -0.109174, -0.119064, -0.0242958, -0.180555,   0.118983,
            0.341578,   0.275472,   0.0853083, 0.332205,  -0.0498387, 0.140338,    0.0319435,
            0.247019,   0.275848,   -0.158223, 0.0495464, -0.0681034, -0.418158,   -0.523234,
            0.469122,   -0.306578,  -0.221095, -0.106449, -0.248934,  -0.00682121, 0.288407,
            0.198708,   0.0695644,  0.211621,  0.00246037};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 4 args (bias is used)
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            -0.156667, -0.248473,  0.0255282,  -0.24566,  0.211589,   0.192707,   0.253025,
            -0.515283, -0.414174,  0.227127,   0.124773,  0.284532,   -0.203929,  -0.120517,
            -0.2794,   0.547635,   0.518549,   0.0447674, 0.258461,   0.0502881,  -0.219516,
            0.0927382, -0.0760062, -0.0906231, 0.237615,  -0.215638,  0.0128074,  -0.425813,
            -0.433378, 0.375383,   -0.0381738, 0.117793,  -0.180851,  -0.0841245, -0.116649,
            0.419469,  0.393515,   -0.076395,  0.427436,  -0.264071,  -0.185829,  0.0483585,
            0.242955,  0.25233,    0.0148512,  -0.304127, -0.0616653, -0.411568,  -0.491748,
            0.476508,  -0.313413,  -0.0361821, -0.173037, -0.235731,  -0.163113,  0.349008,
            0.248674,  -0.0295413, 0.291437,   -0.165005};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 4 args (ih is used)
    {
        migraphx::program p;
        auto seq = p.add_literal(migraphx::literal{in_shape, input});
        auto w   = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r   = p.add_literal(migraphx::literal{r_shape, r_data});
        auto ih  = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto und = p.add_instruction(migraphx::op::undefined{});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          und,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.248571,   0.0982155,  0.00808877, 0.0986508,   0.0969705, 0.434692,  -0.141696,
            -0.164271,  -0.121157,  0.863222,   -0.0718357,  0.137711,  0.109221,  -0.00207995,
            0.0331223,  0.262705,   0.346587,   0.457158,    0.240744,  0.404261,  0.222779,
            0.179757,   -0.0845316, 0.0690347,  0.10204,     0.100155,  -0.190286, -0.122062,
            -0.274379,  0.547281,   -0.226753,  -0.0397069,  0.120404,  0.171299,  0.259989,
            0.0864604,  0.111322,   0.331784,   0.604653,    0.181017,  0.237426,  0.0911999,
            0.233106,   0.32996,    -0.17175,   0.0190231,   -0.154805, -0.205631, -0.405354,
            0.519054,   -0.380409,  -0.0350301, -0.00633752, 0.403791,  0.181883,  -0.0977917,
            -0.0339407, 0.413089,   0.721238,   0.431879};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

TEST_CASE(gru_bidirectional_actv_funcs)
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, input_size}};
    std::vector<float> w_data{
        0.3809,  0.4283,  0.2294,  -0.1018, -0.1226, -0.0037, 0.2449,  -0.2712, -0.1418,
        0.1363,  -0.3453, -0.0693, -0.2281, 0.2699,  -0.2024, -0.3085, -0.3338, 0.4109,
        0.2605,  -0.1019, -0.2813, 0.3323,  -0.1590, 0.0788,  -0.3535, 0.0397,  0.2732,
        0.2906,  0.0519,  0.3617,  -0.2664, 0.1441,  0.0464,  -0.1057, 0.2204,  -0.3294,
        0.3670,  0.1411,  0.3852,  0.3572,  0.3918,  0.0483,  -0.3906, -0.2841, -0.2778,

        -0.4272, 0.2335,  -0.1811, -0.3885, -0.1279, 0.1000,  0.0206,  -0.3284, -0.0353,
        0.1197,  0.1190,  0.3862,  0.0965,  -0.0492, 0.2657,  -0.1430, 0.0597,  0.1408,
        -0.0315, 0.1248,  0.0751,  0.3838,  0.3020,  0.0515,  0.2375,  -0.4255, 0.1714,
        -0.0432, 0.3447,  -0.2441, -0.3989, -0.3428, -0.4204, -0.4080, -0.2683, -0.0996,
        -0.1685, -0.0532, -0.1258, 0.1663,  -0.3526, -0.3915, -0.1721, 0.1292,  -0.2279};

    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size, hidden_size}};
    std::vector<float> r_data{
        -0.2683, 0.0699,  -0.4021, -0.1379, 0.0042,  -0.2447, 0.4006,  0.0270,  -0.0446, 0.1063,
        0.1381,  0.1310,  -0.3596, 0.3869,  0.3929,  0.2750,  0.0890,  0.3069,  -0.1691, -0.2194,
        -0.1066, 0.3187,  -0.4369, -0.0603, -0.0834, -0.1182, -0.2047, 0.3253,  -0.2931, 0.2082,
        0.0424,  0.1111,  -0.2773, -0.0279, -0.0869, 0.1413,  -0.4227, -0.3672, 0.4137,  0.0609,
        0.4223,  -0.4032, 0.2945,  0.3600,  0.3345,  -0.3880, -0.0192, -0.0090, -0.2648, 0.4339,
        -0.0155, 0.4437,  -0.1766, 0.1957,  0.2475,  0.3773,  -0.2710, 0.3289,  -0.2077, -0.2534,
        -0.0832, -0.1632, 0.0728,  0.2520,  0.4153,  0.1659,  -0.4342, 0.0541,  0.1812,  -0.2305,
        0.4440,  0.0946,  0.0410,  -0.4381, -0.3161, 0.3906,  -0.3958, -0.4238, 0.1975,  0.3440,
        0.1437,  -0.0568, 0.1492,  -0.4248, -0.3304, 0.2786,  -0.1328, -0.3740, -0.3566, 0.3074,
        0.0924,  0.2684,  -0.1527, 0.1826,  0.2424,  0.2002,  0.3479,  -0.1089, 0.3472,  -0.3677,
        -0.4231, -0.0798, -0.3709, 0.3924,  0.2774,  -0.3690, -0.0233, 0.2845,  0.1969,  0.1618,
        -0.3742, -0.3619, 0.2925,  -0.1838, -0.1495, -0.3747, 0.0341,  -0.4243, -0.0732, -0.3997,
        0.2139,  0.2425,  0.4171,  -0.3358, 0.3534,  0.0938,  -0.0582, -0.2681, -0.4293, 0.1027,
        0.4101,  0.2641,  -0.4110, -0.1681, 0.3582,  -0.2089, 0.0852,  0.0963,  0.3866,  0.1955,
        -0.2174, 0.1996,  -0.2252, 0.1748,  0.1833,  -0.3155, 0.2567,  -0.4387, 0.3402,  0.0599};

    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    std::vector<float> bias_data{
        -0.1582, -0.0826, 0.4008,  0.0118,  0.2511,  0.1900,  -0.2838, 0.2549,  -0.2484, 0.2363,
        -0.4083, -0.0295, -0.1161, 0.1211,  0.2509,  -0.1414, -0.2628, -0.2992, 0.1517,  0.1817,
        -0.2783, 0.3183,  -0.1629, -0.3108, -0.3418, 0.0411,  0.2203,  0.2187,  -0.2990, -0.0416,
        0.0209,  -0.1024, 0.4443,  -0.4420, -0.0330, -0.3591, -0.2990, 0.2167,  0.1395,  0.2317,
        0.1318,  0.1909,  -0.3615, 0.1953,  -0.2582, -0.2217, 0.3723,  0.1458,  0.2630,  -0.0377,
        0.1754,  0.0800,  -0.3964, -0.3247, 0.4219,  -0.0900, 0.3553,  0.2614,  -0.1298, -0.1124};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    std::vector<float> input{-0.8432,
                             -0.9887,
                             1.3041,
                             -2.6430,
                             -0.3306,
                             -0.8504,
                             -0.3933,
                             0.5151,
                             -0.2951,
                             0.0093,
                             -1.1948,
                             -0.1239,
                             0.0373,
                             1.3211,
                             0.7854,
                             -0.4838,
                             -1.0536,
                             -0.2529};

    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data{-0.0468, 0.5691,  -0.0882, 0.8340,  0.1483, -0.3902, -0.5348,
                               0.4178,  1.0175,  0.9212,  -0.0468, 0.5691, -0.0882, 0.8340,
                               0.1483,  -0.3902, -0.5348, 0.4178,  1.0175, 0.9212};

    float clip = 0.0f;

    // no activation function specified, so default is used.
    {
        migraphx::program p;
        auto seq       = p.add_literal(migraphx::literal{in_shape, input});
        auto w         = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r         = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = p.add_instruction(migraphx::op::undefined{});
        auto ih        = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = p.add_instruction(
            migraphx::op::gru{hidden_size, {}, migraphx::op::rnn_direction::bidirectional, clip, 1},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{-0.0959787, 0.0794681, 0.241526,  0.321104,  0.00693533,
                                        -0.311839,  -0.12802,  -0.16643,  -0.393849, 0.648851,
                                        0.0248217,  0.435231,  -0.144448, 0.101531,  -0.111305,
                                        0.381317,   0.468983,  0.230557,  0.348021,  0.180229};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 1 activation function (sigmoid) specified
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip,
                                            0},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.325495, 0.469214, 0.164517, 0.585327, 0.328398, 0.457928, 0.065011, 0.35986,
            0.545029, 0.859425, 0.427923, 0.667133, 0.41591,  0.540971, 0.365475, 0.482058,
            0.565495, 0.556993, 0.607649, 0.543627, 0.428915, 0.537405, 0.306046, 0.518399,
            0.403561, 0.410694, 0.301163, 0.407397, 0.471334, 0.726446, 0.309389, 0.612072,
            0.360619, 0.590861, 0.366545, 0.367001, 0.433829, 0.501275, 0.72481,  0.512745,
            0.463795, 0.539649, 0.487682, 0.554471, 0.395916, 0.430744, 0.415923, 0.424275,
            0.409655, 0.698256, 0.126883, 0.554374, 0.216137, 0.671491, 0.263833, 0.0678646,
            0.132732, 0.477083, 0.802206, 0.626802};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 1 activation function (tanh) specified
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0919632, -0.398302,   -0.0267752, -0.326771,  0.401983,  0.949841,   0.557779,
            -0.745259, -1.52726,    0.946066,   0.330446,   0.301982,  -0.443763,  -0.0655817,
            -0.326473, 0.861394,    0.560799,   -0.101768,  0.145142,  0.128956,   -0.329758,
            0.458253,  -0.339208,   0.289109,   0.36728,    -1.09574,  -0.181394,  -0.575781,
            -0.823083, 0.804262,    -0.0965933, 0.20405,    -0.430215, 0.00884668, 0.0716857,
            0.844222,  0.516472,    -0.191571,  0.596968,   -0.545405, -0.336693,  -0.0280516,
            0.339058,  1.00367,     0.12655,    -0.0984504, -0.174945, -0.5365,    0.183188,
            0.66716,   -0.704461,   -0.393346,  -0.627123,  0.210395,  0.0563026,  0.31419,
            0.759629,  0.000258222, 0.350835,   -0.682684};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 3 activation functions specified
    {
        migraphx::program p;
        auto seq       = p.add_literal(migraphx::literal{in_shape, input});
        auto w         = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r         = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias      = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und       = p.add_instruction(migraphx::op::undefined{});
        auto ih        = p.add_literal(migraphx::literal{ih_shape, ih_data});
        auto concat_hs = p.add_instruction(
            migraphx::op::gru{hidden_size,
                              {migraphx::op::tanh{}, migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                              migraphx::op::rnn_direction::bidirectional,
                              clip,
                              1},
            seq,
            w,
            r,
            bias,
            und,
            ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.351019, 0.474363, 0.570719,  0.717703,   0.468843,
                                        1.15142,  0.457633, 0.300962,  0.361245,   0.666199,
                                        0.330446, 0.301982, -0.443763, -0.0655817, -0.326473,
                                        0.861394, 0.560799, -0.101768, 0.145142,   0.128956};
        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // 4 activation functions all specified
    {
        migraphx::program p;
        auto seq  = p.add_literal(migraphx::literal{in_shape, input});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{},
                                             migraphx::op::tanh{},
                                             migraphx::op::sigmoid{},
                                             migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{
            0.0352243, 0.0146756,  0.00570925, 0.152446,  0.208683,   0.214342,   -0.0454273,
            -0.135177, -0.0800739, 0.903659,   0.0248217, 0.435231,   -0.144448,  0.101531,
            -0.111305, 0.381317,   0.468983,   0.230557,  0.348021,   0.180229,   -0.0930435,
            0.174108,  -0.063834,  0.0909285,  0.22759,   -0.221983,  -0.139656,  -0.0938906,
            -0.247681, 0.69647,    -0.159396,  0.299061,  -0.116652,  0.238649,   0.109945,
            0.192866,  0.307073,   0.191113,   0.658287,  -0.0340374, -0.0959787, 0.0794681,
            0.241526,  0.321104,   0.00693533, -0.311839, -0.12802,   -0.16643,   -0.393849,
            0.648851,  -0.395918,  0.231694,   -0.160503, 0.383289,   0.0879262,  -0.0254665,
            0.079043,  0.322652,   0.752701,   0.243775};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }

    // seq length of 1
    {
        migraphx::program p;
        seq_len = 1;
        migraphx::shape in_shape_one{migraphx::shape::float_type,
                                     {seq_len, batch_size, input_size}};
        std::vector<float> input_one{-0.8432, -0.9887, 1.3041, -2.6430, -0.3306, -0.8504};
        auto seq  = p.add_literal(migraphx::literal{in_shape_one, input_one});
        auto w    = p.add_literal(migraphx::literal{w_shape, w_data});
        auto r    = p.add_literal(migraphx::literal{r_shape, r_data});
        auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
        auto und  = p.add_instruction(migraphx::op::undefined{});
        auto ih   = p.add_literal(migraphx::literal{ih_shape, ih_data});
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip,
                                            1},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);

        p.compile(migraphx::cpu::target{});
        auto hs_concat = p.eval({});
        std::vector<float> hs_data;
        hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

        std::vector<float> hs_data_gold{0.0352243,  0.0146756,  0.00570925, 0.152446,   0.208683,
                                        0.214342,   -0.0454273, -0.135177,  -0.0800739, 0.903659,
                                        -0.0271321, 0.624762,   -0.117084,  0.509115,   -0.0175078,
                                        -0.144492,  -0.0115366, 0.409153,   0.487015,   0.550755};

        EXPECT(migraphx::verify_range(hs_data, hs_data_gold));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
