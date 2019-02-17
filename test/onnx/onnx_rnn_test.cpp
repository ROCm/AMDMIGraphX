#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

TEST_CASE(rnn_test)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;
    // bidirectional
    {
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w = p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, hs, is}});
        auto r = p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 2 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_rnn_bi.onnx");

        EXPECT(p == prog);
    }

    // forward
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w = p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, hs, is}});
        auto r = p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 2 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_rnn_forward.onnx");

        EXPECT(p == prog);
    }

    // reverse
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w = p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, hs, is}});
        auto r = p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 2 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_rnn_reverse.onnx");

        EXPECT(p == prog);
    }

    // 3 argumments
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w   = p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, hs, is}});
        auto r   = p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, hs, hs}});
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip},
                              seq,
                              w,
                              r,
                              und,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_rnn_3args.onnx");

        EXPECT(p == prog);
    }

    // 5 argumments
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w = p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, hs, is}});
        auto r = p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 2 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::rnn{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_rnn_5args.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(gru_test)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;
    // forward
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip,
                                                1},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_forward.onnx");

        EXPECT(p == prog);
    }

    // reverse
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_reverse.onnx");

        EXPECT(p == prog);
    }

    // bidirectional
    {
        nd = 2;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::tanh{},
                                                 migraphx::op::sigmoid{},
                                                 migraphx::op::relu{},
                                                 migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_bi.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(gru_test_args)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;

    // 3 arguments
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto und = p.add_instruction(migraphx::op::undefined{});
        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip},
                              seq,
                              w,
                              r,
                              und,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_3arg.onnx");

        EXPECT(p == prog);
    }

    // 4 arguments
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_4arg.onnx");

        EXPECT(p == prog);
    }

    // 5 arguments
    {
        nd = 2;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_5arg.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(gru_test_actv_funcs)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;
    // bidirection, 0 actv function
    {
        nd = 2;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = p.add_instruction(
            migraphx::op::gru{hs, {}, migraphx::op::rnn_direction::bidirectional, clip},
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_bi_0.onnx");

        EXPECT(p == prog);
    }

    // bidirection, 1 actv function
    {
        nd = 2;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = p.add_instruction(
            migraphx::op::gru{
                hs, {migraphx::op::tanh{}}, migraphx::op::rnn_direction::bidirectional, clip},
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_bi_1.onnx");

        EXPECT(p == prog);
    }

    // bidirection, 2 actv functions
    {
        nd = 2;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::tanh{}, migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_bi_2.onnx");

        EXPECT(p == prog);
    }

    // bidirection, 3 actv functions
    {
        nd = 2;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = p.add_instruction(
            migraphx::op::gru{hs,
                              {migraphx::op::tanh{}, migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                              migraphx::op::rnn_direction::bidirectional,
                              clip},
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_bi_3.onnx");

        EXPECT(p == prog);
    }

    // forward, 0 actv function
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs, {}, migraphx::op::rnn_direction::forward, clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_forward_0.onnx");

        EXPECT(p == prog);
    }

    // reverse, 1 actv function
    {
        nd = 1;
        migraphx::program p;

        auto seq =
            p.add_parameter("seq", migraphx::shape{migraphx::shape::float_type, {sl, bs, is}});
        auto w =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, is}});
        auto r =
            p.add_parameter("r", migraphx::shape{migraphx::shape::float_type, {nd, 3 * hs, hs}});
        auto bias =
            p.add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {nd, 6 * hs}});
        auto seq_len =
            p.add_parameter("seq_len", migraphx::shape{migraphx::shape::int32_type, {bs}});
        auto ih = p.add_parameter("h0", migraphx::shape{migraphx::shape::float_type, {nd, bs, hs}});

        auto out_hs = p.add_instruction(
            migraphx::op::gru{
                hs, {migraphx::op::relu{}}, migraphx::op::rnn_direction::reverse, clip},
            seq,
            w,
            r,
            bias,
            seq_len,
            ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = migraphx::parse_onnx("onnx_gru_reverse_1.onnx");

        EXPECT(p == prog);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
