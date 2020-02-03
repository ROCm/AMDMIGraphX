#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/onnx.hpp>
#include "test.hpp"

migraphx::program optimize_onnx(const std::string& name, bool eliminate_deadcode = true)
{
    auto prog = migraphx::parse_onnx(name);
    if(eliminate_deadcode)
        migraphx::run_passes(prog, {migraphx::dead_code_elimination{}});

    // remove the last identity instruction
    auto last_ins = std::prev(prog.end());
    if(last_ins->name() == "ret")
    {
        prog.remove_instruction(last_ins);
    }

    return prog;
}

TEST_CASE(rnn_test_bidirectional)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 2;  // num directions
    float clip     = 0.0f;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 2 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};

    migraphx::program p;

    auto seq     = p.add_parameter("seq", seq_shape);
    auto w       = p.add_parameter("w", w_shape);
    auto r       = p.add_parameter("r", r_shape);
    auto bias    = p.add_parameter("bias", bias_shape);
    auto seq_len = p.add_parameter("seq_len", sl_shape);
    auto ih      = p.add_parameter("h0", ih_shape);

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
    auto prog = optimize_onnx("onnx_rnn_bi.onnx");

    EXPECT(p == prog);
}

TEST_CASE(rnn_test_one_direction)
{
    std::size_t sl = 5;  // sequence len
    std::size_t bs = 3;  // batch size
    std::size_t hs = 20; // hidden size
    std::size_t is = 10; // input size
    std::size_t nd = 1;  // num directions
    float clip     = 0.0f;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 2 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};

    // forward
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);

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
        auto prog = optimize_onnx("onnx_rnn_forward.onnx");

        EXPECT(p == prog);
    }

    // reverse
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
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
        auto prog = optimize_onnx("onnx_rnn_reverse.onnx");

        EXPECT(p == prog);
    }

    // 3 argumments
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
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
        auto prog = optimize_onnx("onnx_rnn_3args.onnx");

        EXPECT(p == prog);
    }

    // 5 argumments
    {
        migraphx::program p;

        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

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
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_rnn_5args.onnx");

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
        auto prog = optimize_onnx("onnx_gru_forward.onnx");

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
        auto prog = optimize_onnx("onnx_gru_reverse.onnx");

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
        auto prog = optimize_onnx("onnx_gru_bi.onnx");

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
        auto prog = optimize_onnx("onnx_gru_3arg.onnx");

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
                                                {migraphx::op::relu{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_gru_4arg.onnx");

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
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_gru_5arg.onnx");

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

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::sigmoid{},
                                                 migraphx::op::tanh{},
                                                 migraphx::op::sigmoid{},
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
        auto prog = optimize_onnx("onnx_gru_bi_0.onnx");

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

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::sigmoid{},
                                                 migraphx::op::sigmoid{},
                                                 migraphx::op::sigmoid{},
                                                 migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_gru_bi_1.onnx");

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
                                                {migraphx::op::tanh{},
                                                 migraphx::op::sigmoid{},
                                                 migraphx::op::tanh{},
                                                 migraphx::op::sigmoid{}},
                                                migraphx::op::rnn_direction::bidirectional,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_gru_bi_2.onnx");

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

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::tanh{},
                                                 migraphx::op::sigmoid{},
                                                 migraphx::op::tanh{},
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
        auto prog = optimize_onnx("onnx_gru_bi_3.onnx");

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
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                                migraphx::op::rnn_direction::forward,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_gru_forward_0.onnx");

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

        auto out_hs =
            p.add_instruction(migraphx::op::gru{hs,
                                                {migraphx::op::relu{}, migraphx::op::relu{}},
                                                migraphx::op::rnn_direction::reverse,
                                                clip},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_gru_reverse_1.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_forward)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 1;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
        auto ic      = p.add_parameter("c0", ih_shape);
        auto pph     = p.add_parameter("pph", pph_shape);

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_forward.onnx");

        EXPECT(p == prog);
    }

    // 3 args
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_f3args.onnx");

        EXPECT(p == prog);
    }

    // 3 args, hs output
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});

        p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        auto prog = optimize_onnx("onnx_lstm_hs.onnx");

        EXPECT(p == prog);
    }

    // 3 args, last output
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_last.onnx");

        EXPECT(p == prog);
    }

    // 3 args, cell output
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::lstm_last_cell_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_cell.onnx");

        EXPECT(p == prog);
    }

    // 4 args
    {
        migraphx::program p;
        auto seq  = p.add_parameter("seq", seq_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", bias_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            bias,
            und,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_f4args.onnx");

        EXPECT(p == prog);
    }

    // 5 args
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            bias,
            seq_len,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.add_instruction(migraphx::op::lstm_last_cell_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_f5args.onnx");

        EXPECT(p == prog);
    }

    // 6 args
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.add_instruction(migraphx::op::lstm_last_cell_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_f6args.onnx");

        EXPECT(p == prog);
    }

    // 7 args
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
        auto ic      = p.add_parameter("c0", ih_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.add_instruction(migraphx::op::lstm_last_cell_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_f7args.onnx");

        EXPECT(p == prog);
    }
}

// activation functions
TEST_CASE(lstm_forward_actv_func)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 1;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    // no activation function specified
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_f0af.onnx");

        EXPECT(p == prog);
    }

    // 1 activation function specified
    {
        migraphx::program p;
        auto seq  = p.add_parameter("seq", seq_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", bias_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::sigmoid{}, migraphx::op::sigmoid{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            bias,
            und,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_f1af.onnx");

        EXPECT(p == prog);
    }

    // 2 activation function specified
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::tanh{}, migraphx::op::sigmoid{}, migraphx::op::sigmoid{}},
                migraphx::op::rnn_direction::forward,
                clip,
                input_forget},
            seq,
            w,
            r,
            bias,
            seq_len,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.add_instruction(migraphx::op::lstm_last_cell_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_f2af.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_reverse)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 1;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
        auto ic      = p.add_parameter("c0", ih_shape);
        auto pph     = p.add_parameter("pph", pph_shape);

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::reverse,
                clip,
                input_forget},
            seq,
            w,
            r,
            bias,
            seq_len,
            ih,
            ic,
            pph);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_reverse.onnx");

        EXPECT(p == prog);
    }

    // 5 args
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::reverse,
                clip,
                input_forget},
            seq,
            w,
            r,
            bias,
            seq_len,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        p.add_instruction(migraphx::op::lstm_last_cell_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_r5args.onnx");

        EXPECT(p == prog);
    }

    // no activation function specified
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs = p.add_instruction(
            migraphx::op::lstm{
                hs,
                {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                migraphx::op::rnn_direction::reverse,
                clip,
                input_forget},
            seq,
            w,
            r,
            und,
            und,
            und,
            und,
            und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_r0af.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_bidirectional)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 2;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
        auto ic      = p.add_parameter("c0", ih_shape);
        auto pph     = p.add_parameter("pph", pph_shape);

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih,
                              ic,
                              pph);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi.onnx");

        EXPECT(p == prog);
    }

    // 3 args
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              und,
                              und,
                              und,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi3args.onnx");

        EXPECT(p == prog);
    }

    // 4 args
    {
        migraphx::program p;
        auto seq  = p.add_parameter("seq", seq_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", bias_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              und,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi4args.onnx");

        EXPECT(p == prog);
    }

    // 5 args
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              und,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi5args.onnx");

        EXPECT(p == prog);
    }

    // 6 args
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi6args.onnx");

        EXPECT(p == prog);
    }

    // 7 args
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
        auto ic      = p.add_parameter("c0", ih_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih,
                              ic,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi7args.onnx");

        EXPECT(p == prog);
    }
}

TEST_CASE(lstm_bi_actv_funcs)
{
    std::size_t sl   = 5;  // sequence len
    std::size_t bs   = 3;  // batch size
    std::size_t hs   = 20; // hidden size
    std::size_t is   = 10; // input size
    std::size_t nd   = 2;  // num directions
    float clip       = 0.0f;
    int input_forget = 1;
    migraphx::shape seq_shape{migraphx::shape::float_type, {sl, bs, is}};
    migraphx::shape w_shape{migraphx::shape::float_type, {nd, 4 * hs, is}};
    migraphx::shape r_shape{migraphx::shape::float_type, {nd, 4 * hs, hs}};
    migraphx::shape bias_shape{migraphx::shape::float_type, {nd, 8 * hs}};
    migraphx::shape sl_shape{migraphx::shape::int32_type, {bs}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {nd, bs, hs}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {nd, 3 * hs}};
    // 0 activation function
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              und,
                              und,
                              und,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi0af.onnx");

        EXPECT(p == prog);
    }

    // 1 activation function
    {
        migraphx::program p;
        auto seq  = p.add_parameter("seq", seq_shape);
        auto w    = p.add_parameter("w", w_shape);
        auto r    = p.add_parameter("r", r_shape);
        auto bias = p.add_parameter("bias", bias_shape);
        auto und  = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::sigmoid{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              bias,
                              und,
                              und,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi1af.onnx");

        EXPECT(p == prog);
    }

    // 2 activation functions
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              und,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi2af.onnx");

        EXPECT(p == prog);
    }

    // 4 activation functions
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi4af.onnx");

        EXPECT(p == prog);
    }

    // 5 activation functions
    {
        migraphx::program p;
        auto seq     = p.add_parameter("seq", seq_shape);
        auto w       = p.add_parameter("w", w_shape);
        auto r       = p.add_parameter("r", r_shape);
        auto bias    = p.add_parameter("bias", bias_shape);
        auto seq_len = p.add_parameter("seq_len", sl_shape);
        auto ih      = p.add_parameter("h0", ih_shape);
        auto ic      = p.add_parameter("c0", ih_shape);
        auto und     = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::sigmoid{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              bias,
                              seq_len,
                              ih,
                              ic,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi5af.onnx");

        EXPECT(p == prog);
    }

    // 6 activation functions
    {
        migraphx::program p;
        auto seq = p.add_parameter("seq", seq_shape);
        auto w   = p.add_parameter("w", w_shape);
        auto r   = p.add_parameter("r", r_shape);
        auto und = p.add_instruction(migraphx::op::undefined{});

        auto out_hs =
            p.add_instruction(migraphx::op::lstm{hs,
                                                 {migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::tanh{},
                                                  migraphx::op::sigmoid{},
                                                  migraphx::op::tanh{}},
                                                 migraphx::op::rnn_direction::bidirectional,
                                                 clip,
                                                 input_forget},
                              seq,
                              w,
                              r,
                              und,
                              und,
                              und,
                              und,
                              und);
        p.add_instruction(migraphx::op::rnn_last_output{}, out_hs);
        auto prog = optimize_onnx("onnx_lstm_bi6af.onnx");

        EXPECT(p == prog);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
