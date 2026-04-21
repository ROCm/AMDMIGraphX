/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#include "migraphx/check_shapes.hpp"
#include <migraphx/fuse_flash_decoding_reduce.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

struct ck_tile_splitkv_combine 
{
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack();
    }

    std::string name() const { return "gpu::ck_tile_splitkv_combine"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2).same_ndims().min_ndims(4).max_ndims(5);
        auto o_shape   = inputs[0];
        auto o_lens    = o_shape.lens();
        auto split_axis = o_lens.size() - 3;
        o_lens.erase(o_lens.begin() + split_axis);
        return {o_shape.type(), o_lens};
    }

};
MIGRAPHX_REGISTER_OP(ck_tile_splitkv_combine);

struct find_flash_decoding_reduce
{
    auto matcher() const
    {
        // // Match the attention group instruction that produces a tuple {O', LSE}
        // auto group =
        //     match::name("group")(match::has_op_value("tag", "attention")).bind("group");
        //
        // // O' = get_tuple_elem[index=0], LSE = get_tuple_elem[index=1]
        // auto o_prime =
        //     match::name("get_tuple_elem")(match::has_op_value("index", 0),
        //                                   match::arg(0)(group))
        //         .bind("o_prime");
        // auto lse =
        //     match::name("get_tuple_elem")(match::has_op_value("index", 1),
        //                                   match::arg(0)(group))
        //         .bind("lse");

        // Scale computation: softmax across a reduction axis
        // The input to reduce_max is treated generically (e.g. LSE from flash decoding)
        auto input     = match::any().bind("lse");
        auto rmax      = match::name("reduce_max")(match::arg(0)(input)).bind("reduce_max");
        auto rmax_bcast = match::name("multibroadcast")(match::arg(0)(rmax));
        auto lse_sub   = match::name("sub")(match::arg(0)(input), match::arg(1)(rmax_bcast));
        auto lse_exp   = match::name("exp")(match::arg(0)(lse_sub));

        // exp -> reduce_sum -> multibroadcast -> div(exp, bcast)
        auto rsum       = match::name("reduce_sum")(match::arg(0)(lse_exp));
        auto rsum_bcast = match::name("multibroadcast")(match::arg(0)(rsum));
        auto scale_div  = match::name("div")(match::arg(0)(lse_exp), match::arg(1)(rsum_bcast));

        // div -> multibroadcast -> convert  (fuse_attention output)
        // OR div -> convert -> multibroadcast  (after simplify_reshapes)
        auto bcast_then_convert =
            match::name("convert")(match::arg(0)(
                match::name("multibroadcast")(match::arg(0)(scale_div))));
        auto convert_then_bcast =
            match::name("multibroadcast")(match::arg(0)(
                match::name("convert")(match::arg(0)(scale_div))));
        auto scale = match::any_of(bcast_then_convert, convert_then_bcast);

        // mul(operand, scale) -> reduce_sum -> squeeze
        auto operand  = match::any().bind("o");
        auto scaled   = match::name("mul")(match::arg(0)(operand),
                                         match::arg(1)(scale));
        auto reduced  = match::name("reduce_sum")(match::arg(0)(scaled));
        auto squeezed = match::name("squeeze")(match::arg(0)(reduced)).bind("squeeze");

        // Optional trailing slice when padding was applied
        return match::any_of(match::name("slice")(match::arg(0)(squeezed)), squeezed);
    }

    void apply(module_pass_manager&, const match::matcher_result&) const {
        std::cout << "apply flash decoding reduce" << std::endl;
    }
};

} // namespace

void fuse_flash_decoding_reduce::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_flash_decoding_reduce{});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
