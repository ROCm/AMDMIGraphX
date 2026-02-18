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
#include <migraphx/gpu/prepare_reduce.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct parallel_reduce
{
    operation op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::parallel_reduce"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        std::vector<shape> result;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(result), [&](auto input) {
            return op.compute_shape({input});
        });
        return shape{result};
    }
};
MIGRAPHX_REGISTER_OP(parallel_reduce);

struct arg_reduce
{
    operation op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::arg_reduce"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        // inputs: [values, indices (lazy)]
        // output: tuple of (reduced_value_shape, reduced_index_shape)
        auto index_shape = op.compute_shape({inputs.front()});
        auto value_shape = index_shape.with_type(inputs.front().type());
        return shape{{value_shape, index_shape}};
    }
};
MIGRAPHX_REGISTER_OP(arg_reduce);

struct make_indices
{
    std::size_t size = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"));
    }

    std::string name() const { return "gpu::make_indices"; }

    // This op produces a lazy index tensor,shape matches the reduction dimension
    shape compute_shape(const std::vector<shape>&) const
    {
        return shape{shape::int64_type, {size}};
    }
};
MIGRAPHX_REGISTER_OP(make_indices);

namespace {

// find argmin/argmax operations
std::vector<instruction_ref> find_arg_reduce(module& m)
{
    std::vector<instruction_ref> result;
    auto im = iterator_for(m);
    std::copy_if(im.begin(), im.end(), std::back_inserter(result), [](auto ins) {
        return ins->name() == "argmin" or ins->name() == "argmax";
    });
    return result;
}

// rewrite argmin/argmax to return lazy indices and values tuple
void rewrite_arg_reduce(module& m)
{
    for(auto ins : find_arg_reduce(m))
    {
        auto input      = ins->inputs().front();
        auto v          = ins->get_operator().to_value();
        auto axis_val   = v["axis"].to<int64_t>();
        auto ndim       = input->get_shape().ndim();
        auto axis       = axis_val < 0 ? axis_val + ndim : axis_val;
        auto axis_size  = input->get_shape().lens()[axis];

        // make_indices to generate lazy indices
        auto indices = m.insert_instruction(ins, make_indices{axis_size});
        // arg_reduce op to get values and indices tuple
        auto arg_reduce_ins =
            m.insert_instruction(ins, arg_reduce{ins->get_operator()}, input, indices);
        auto result =
            m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", 1}}), arg_reduce_ins);
        m.replace_instruction(ins, result);
    }
}

std::vector<instruction_ref> find_reduce(module& m)
{
    std::vector<instruction_ref> result;
    auto im = iterator_for(m);
    std::copy_if(im.begin(), im.end(), std::back_inserter(result), [](auto ins) {
        if(contains({"gpu::parallel_reduce", "reduce_mean", "gpu::arg_reduce"}, ins->name()))
            return false;
        return contains(ins->name(), "reduce");
    });
    return result;
}

std::vector<instruction_ref> find_parallel_reduce(const std::vector<instruction_ref>& r)
{
    std::vector<instruction_ref> result;
    auto ir = iterator_for(r);
    transform_if(
        ir.begin(),
        ir.end(),
        std::back_inserter(result),
        [&](auto x) {
            return std::none_of(
                std::next(x), r.end(), [&](auto reduce) { return reaches(*x, reduce); });
        },
        [](auto x) { return *x; });
    return result;
}

void fuse_reductions(module& m)
{
    auto rs = find_parallel_reduce(find_reduce(m));
    if(rs.size() < 2)
        return;
    // Only handle the same reduction operator (and its data-type) for now
    if(std::any_of(std::next(rs.cbegin()), rs.cend(), [&](auto r) {
           return (*rs.cbegin())->name() != r->name() or
                  (*rs.cbegin())->get_shape().type() != r->get_shape().type();
       }))
        return;

    auto last = rs.front();
    auto op   = last->get_operator();
    std::vector<instruction_ref> inputs;
    std::transform(rs.begin(), rs.end(), std::back_inserter(inputs), [&](auto r) {
        return r->inputs().front();
    });
    auto pr = m.insert_instruction(last, parallel_reduce{op}, inputs);
    int i   = 0;
    for(auto r : rs)
    {
        m.replace_instruction(r, make_op("get_tuple_elem", {{"index", i}}), pr);
        i++;
    }
    m.sort();
}

} // namespace

void prepare_reduce::apply(module& m) const
{
    // rewrite argmin/argmax to handle tuples
    rewrite_arg_reduce(m);
    fuse_reductions(m);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
