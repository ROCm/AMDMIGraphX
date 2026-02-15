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
 */
#include <migraphx/shape.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/value.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

// ============================================================
// Wrapper operation: gpu::gen::op
// Analogous to gpu::mlir_op - wraps a submodule for gen IR
// ============================================================

struct gen_op
{
    operation op = make_op("identity");

    std::string name() const { return "gpu::gen::op"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);
        if(mods.size() != 1)
            MIGRAPHX_THROW("gpu::gen::op should have one submodule.");
        auto result =
            mods[0]->compute_shapes(inputs, {.name = name(), .strict_type = true, .strict_lens = true});
        if(result.size() == 1)
            return result.front();
        return shape{result};
    }
};
MIGRAPHX_REGISTER_OP(gen_op);

// ============================================================
// ID Operations
// ============================================================

struct global_id
{
    std::string name() const { return "gpu::gen::global_id"; }

    shape compute_shape(std::vector<shape>) const { return shape{shape::uint32_type}; }

    value attributes() const { return {{"gpu_gen", "idx.global"}}; }
};
MIGRAPHX_REGISTER_OP(global_id);

struct local_id
{
    std::string name() const { return "gpu::gen::local_id"; }

    shape compute_shape(std::vector<shape>) const { return shape{shape::uint32_type}; }

    value attributes() const { return {{"gpu_gen", "idx.local"}}; }
};
MIGRAPHX_REGISTER_OP(local_id);

struct workgroup_id
{
    std::string name() const { return "gpu::gen::workgroup_id"; }

    shape compute_shape(std::vector<shape>) const { return shape{shape::uint32_type}; }

    value attributes() const { return {{"gpu_gen", "idx.group"}}; }
};
MIGRAPHX_REGISTER_OP(workgroup_id);

struct workgroup_size
{
    std::string name() const { return "gpu::gen::workgroup_size"; }

    shape compute_shape(std::vector<shape>) const { return shape{shape::uint32_type}; }

    value attributes() const { return {{"gpu_gen", "idx.nlocal()"}}; }
};
MIGRAPHX_REGISTER_OP(workgroup_size);

struct lane_id
{
    std::string name() const { return "gpu::gen::lane_id"; }

    shape compute_shape(std::vector<shape>) const { return shape{shape::uint32_type}; }

    value attributes() const { return {{"gpu_gen", "idx.local_wave()"}}; }
};
MIGRAPHX_REGISTER_OP(lane_id);

// ============================================================
// Memory Operations
// ============================================================

struct load
{
    std::string name() const { return "gpu::gen::load"; }

    // inputs: tensor, index
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return shape{inputs[0].type()};
    }

    value attributes() const { return {{"gpu_gen", "${0}[${1}]"}}; }
};
MIGRAPHX_REGISTER_OP(load);

struct store
{
    std::string name() const { return "gpu::gen::store"; }

    // inputs: tensor, index, value
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return shape{shape::uint32_type};
    }

    value attributes() const { return {{"gpu_gen", "${0}[${1}] = ${2}"}}; }
};
MIGRAPHX_REGISTER_OP(store);

struct vector_load
{
    std::size_t size = 4;

    std::string name() const { return "gpu::gen::vector_load"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"));
    }

    // inputs: tensor, index
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return shape{inputs[0].type(), {size}};
    }

    value attributes() const
    {
        return {{"gpu_gen",
                 "gen::vec_load<" + std::to_string(size) + ">(${0}.data(), ${1})"}};
    }
};
MIGRAPHX_REGISTER_OP(vector_load);

struct vector_store
{
    std::size_t size = 4;

    std::string name() const { return "gpu::gen::vector_store"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"));
    }

    // inputs: tensor, index, value
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return shape{shape::uint32_type};
    }

    value attributes() const
    {
        return {{"gpu_gen",
                 "(void)gen::vec_store<" + std::to_string(size) +
                     ">(${0}.data(), ${1}, ${2})"}};
    }
};
MIGRAPHX_REGISTER_OP(vector_store);

struct strided_load
{
    std::size_t size   = 1;
    std::size_t stride = 1;

    std::string name() const { return "gpu::gen::strided_load"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"), f(self.stride, "stride"));
    }

    // inputs: tensor, base_index
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return shape{inputs[0].type(), {size}};
    }

    value attributes() const
    {
        return {{"gpu_gen",
                 "gen::strided_load<" + std::to_string(size) + ", " +
                     std::to_string(stride) + ">(${0}.data(), ${1})"}};
    }
};
MIGRAPHX_REGISTER_OP(strided_load);

struct strided_store
{
    std::size_t size   = 1;
    std::size_t stride = 1;

    std::string name() const { return "gpu::gen::strided_store"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"), f(self.stride, "stride"));
    }

    // inputs: tensor, index, value
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return shape{shape::uint32_type};
    }

    value attributes() const
    {
        return {{"gpu_gen",
                 "(void)gen::strided_store<" + std::to_string(size) + ", " +
                     std::to_string(stride) + ">(${0}.data(), ${1}, ${2})"}};
    }
};
MIGRAPHX_REGISTER_OP(strided_store);

struct copy
{
    std::string schedule = "per_lane";

    std::string name() const { return "gpu::gen::copy"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.schedule, "schedule"));
    }

    // inputs: source tensor, destination tensor
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return inputs[1];
    }
};
MIGRAPHX_REGISTER_OP(copy);

struct lds_allocate
{
    shape alloc_shape;

    std::string name() const { return "gpu::gen::lds_allocate"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alloc_shape, "shape"));
    }

    shape compute_shape(std::vector<shape>) const { return alloc_shape; }
};
MIGRAPHX_REGISTER_OP(lds_allocate);

// ============================================================
// Tiling Operations
// ============================================================

struct tile_region
{
    std::vector<std::size_t> tile_dims;
    std::size_t axis = 0;

    std::string name() const { return "gpu::gen::tile_region"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.tile_dims, "tile_dims"), f(self.axis, "axis"));
    }

    // inputs: tensor, workgroup_id
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        auto s    = inputs[0];
        auto lens = s.lens();
        // Replace dimensions starting at axis with tile dimensions
        for(std::size_t i = 0; i < tile_dims.size() and (axis + i) < lens.size(); ++i)
        {
            lens[axis + i] = tile_dims[i];
        }
        return shape{s.type(), lens};
    }
};
MIGRAPHX_REGISTER_OP(tile_region);

// ============================================================
// Synchronization and Control
// ============================================================

struct barrier
{
    std::string name() const { return "gpu::gen::barrier"; }

    // Acts as identity for ordering: input passes through
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.empty())
            return shape{shape::uint32_type};
        return inputs[0];
    }

    value attributes() const { return {{"gpu_gen", "(void)__syncthreads()"}}; }
};
MIGRAPHX_REGISTER_OP(barrier);

struct check
{
    std::string name() const { return "gpu::gen::check"; }

    // inputs: condition, value to pass through
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return inputs[1];
    }

    value attributes() const { return {{"gpu_gen", "(MIGRAPHX_CHECK(${0}), ${1})"}}; }
};
MIGRAPHX_REGISTER_OP(check);

// ============================================================
// Reduction Operations
// ============================================================

struct lane_reduce
{
    std::string op = "sum";

    std::string name() const { return "gpu::gen::lane_reduce"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    // inputs: array value
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return shape{inputs[0].type()};
    }

    value attributes() const
    {
        return {{"gpu_gen", "gen::lane_reduce_" + op + "(${0})"}};
    }
};
MIGRAPHX_REGISTER_OP(lane_reduce);

struct dpp_reduce
{
    std::string op = "sum";

    std::string name() const { return "gpu::gen::dpp_reduce"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    // inputs: scalar value
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return inputs[0];
    }

    value attributes() const
    {
        return {{"gpu_gen", "gen::dpp_reduce_" + op + "(${0})"}};
    }
};
MIGRAPHX_REGISTER_OP(dpp_reduce);

struct reduce_waves
{
    std::string op = "sum";

    std::string name() const { return "gpu::gen::reduce_waves"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    // inputs: value, lds_buffer
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return inputs[0];
    }

    value attributes() const
    {
        return {{"gpu_gen",
                 "gen::block_reduce_" + op +
                     "(${0}, ${1}.data(), idx.nwaves(), idx.wave_id(), idx.local_wave())"}};
    }
};
MIGRAPHX_REGISTER_OP(reduce_waves);

// ============================================================
// Index Transformation Operations
// ============================================================

struct pad_index
{
    shape input_shape;
    std::vector<std::int64_t> pads;

    std::string name() const { return "gpu::gen::pad_index"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.input_shape, "input_shape"), f(self.pads, "pads"));
    }

    // inputs: index
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return shape{shape::uint32_type};
    }
};
MIGRAPHX_REGISTER_OP(pad_index);

struct gather_index
{
    shape input_shape;
    std::int64_t axis = 0;

    std::string name() const { return "gpu::gen::gather_index"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.input_shape, "input_shape"), f(self.axis, "axis"));
    }

    // inputs: indices_tensor, index
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return shape{shape::uint32_type};
    }
};
MIGRAPHX_REGISTER_OP(gather_index);

struct reverse_index
{
    shape input_shape;
    std::vector<std::int64_t> axes;

    std::string name() const { return "gpu::gen::reverse_index"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.input_shape, "input_shape"), f(self.axes, "axes"));
    }

    // inputs: index
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return shape{shape::uint32_type};
    }
};
MIGRAPHX_REGISTER_OP(reverse_index);

struct shape_index
{
    shape input_shape;
    shape output_shape;

    std::string name() const { return "gpu::gen::shape_index"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.input_shape, "input_shape"), f(self.output_shape, "output_shape"));
    }

    // inputs: index
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return shape{shape::uint32_type};
    }
};
MIGRAPHX_REGISTER_OP(shape_index);

struct conditional_load
{
    std::size_t size = 1;

    std::string name() const { return "gpu::gen::conditional_load"; }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"));
    }

    // inputs: tensor, index, fill_value
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        if(size > 1)
            return shape{inputs[0].type(), {size}};
        return shape{inputs[0].type()};
    }

    value attributes() const
    {
        return {{"gpu_gen", "gen::conditional_load(${0}, ${1}, ${2})"}};
    }
};
MIGRAPHX_REGISTER_OP(conditional_load);

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
