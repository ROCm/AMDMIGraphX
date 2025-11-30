/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/module.hpp>
#include <migraphx/register_op.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

/// Gen pointwise operation - wraps a pointwise module for gen IR compilation
struct pointwise
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::pointwise"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>&) const
    {
        // Output shape is always the allocation (last input)
        return inputs.back();
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};
MIGRAPHX_REGISTER_OP(pointwise);

/// Tile region operation - represents a tiled computation region
struct tile_region
{
    std::vector<std::size_t> tile_dims = {};
    std::size_t axis                   = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.tile_dims, "tile_dims"), f(self.axis, "axis"));
    }

    std::string name() const { return "gpu::gen::tile_region"; }

    value attributes() const { return {{"point_op", "tile<${tile_dims}>(${0})"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        return inputs.front();
    }
};
MIGRAPHX_REGISTER_OP(tile_region);

/// Lane ID - returns the lane index within a wavefront
struct lane_id
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::lane_id"; }

    value attributes() const { return {{"point_op", "idx.local_wave()"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(0);
        return shape{shape::uint64_type};
    }
};
MIGRAPHX_REGISTER_OP(lane_id);

/// Local ID - returns the thread index within a workgroup
struct local_id
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::local_id"; }

    value attributes() const { return {{"point_op", "idx.local"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(0);
        return shape{shape::uint64_type};
    }
};
MIGRAPHX_REGISTER_OP(local_id);

/// Global ID - returns the global thread index
struct global_id
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::global_id"; }

    value attributes() const { return {{"point_op", "idx.global"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(0);
        return shape{shape::uint64_type};
    }
};
MIGRAPHX_REGISTER_OP(global_id);

/// Workgroup ID - returns the workgroup index
struct workgroup_id
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::workgroup_id"; }

    value attributes() const { return {{"point_op", "idx.group"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(0);
        return shape{shape::uint64_type};
    }
};
MIGRAPHX_REGISTER_OP(workgroup_id);

/// Workgroup size - returns the workgroup size
struct workgroup_size
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::workgroup_size"; }

    value attributes() const { return {{"point_op", "idx.nlocal()"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(0);
        return shape{shape::uint64_type};
    }
};
MIGRAPHX_REGISTER_OP(workgroup_size);

/// Barrier - workgroup synchronization primitive
struct barrier
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::barrier"; }

    value attributes() const { return {{"point_op", "__syncthreads()"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(0);
        return shape{};
    }
};
MIGRAPHX_REGISTER_OP(barrier);

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
