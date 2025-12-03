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

/// Gen operation - wraps a submodule containing high-level tensor operations
/// (pointwise, reduce, pad, gather, etc.) for gen IR compilation.
/// Similar to mlir_op, this uses the submodule to compute shapes and
/// the tiling/lower passes generate the tiles and loads/stores needed.
struct op
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::op"; }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>&) const
    {
        // The last input is always the output allocation
        // Return its shape as the output
        return inputs.back();
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};
MIGRAPHX_REGISTER_OP(op);

/// Tile region operation - represents a tiled view of a tensor for a specific workgroup
/// Input 0: tensor (the full tensor to tile)
/// Input 1: workgroup_id (which tile this workgroup processes)
/// Output: a view into the tensor for the current tile
struct tile_region
{
    std::vector<std::size_t> tile_dims = {}; // Size of each tile
    std::size_t axis                   = 0;  // Starting axis for tiling

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.tile_dims, "tile_dims"), f(self.axis, "axis"));
    }

    std::string name() const { return "gpu::gen::tile_region"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);
        auto tensor_shape = inputs.front();

        // Compute the tiled output shape
        // Dimensions before axis stay as 1 (batch dims processed by outer tiles)
        // Dimensions at axis onwards are replaced by tile_dims
        std::vector<std::size_t> out_lens;
        std::vector<std::size_t> out_strides;

        auto orig_lens    = tensor_shape.lens();
        auto orig_strides = tensor_shape.strides();

        // Keep dimensions before axis as 1
        for(std::size_t i = 0; i < axis && i < orig_lens.size(); i++)
        {
            out_lens.push_back(1);
            out_strides.push_back(orig_strides[i]);
        }

        // Add tile dimensions
        for(std::size_t i = 0; i < tile_dims.size() && (axis + i) < orig_lens.size(); i++)
        {
            out_lens.push_back(tile_dims[i]);
            out_strides.push_back(orig_strides[axis + i]);
        }

        return shape{tensor_shape.type(), out_lens, out_strides};
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

    // (void) prefix indicates this returns void and should not create a variable
    value attributes() const { return {{"point_op", "(void)__syncthreads()"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(0);
        return shape{};
    }
};
MIGRAPHX_REGISTER_OP(barrier);

/// Check - asserts a condition is true, aborts if false
struct check
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::check"; }

    // (void) prefix indicates this returns void and should not create a variable
    value attributes() const { return {{"point_op", "(void)MIGRAPHX_CHECK(${0})"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        // Check takes a boolean condition and returns nothing
        return shape{};
    }
};
MIGRAPHX_REGISTER_OP(check);

/// Offset - computes the linear memory offset from a logical index using tensor shape
/// Input 0: index (the logical linear index)
/// Attribute shape: the tensor shape used to compute strides
/// Output: the memory offset
struct offset
{
    shape s; // Shape used to compute the offset (defines strides)

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"));
    }

    std::string name() const { return "gpu::gen::offset"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        // Output is a single offset value
        return shape{shape::uint64_type};
    }
};
MIGRAPHX_REGISTER_OP(offset);

/// Vector load - loads a vector of elements from a tensor at an offset
/// Input 0: tensor (the tensor_view parameter)
/// Input 1: offset (the memory offset to load from, computed by offset op)
/// Output: vector of `size` elements
struct vector_load
{
    std::size_t size = 1; // Vector size (1, 2, 4, 8, etc.)

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"));
    }

    std::string name() const { return "gpu::gen::vector_load"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);
        // First input is the tensor, second is the offset
        auto tensor_type = inputs.front().type();
        // Output is a vector of `size` elements
        return shape{tensor_type, {size}};
    }
};
MIGRAPHX_REGISTER_OP(vector_load);

/// Vector store - stores a vector of elements to a tensor at an offset
/// Input 0: tensor (the tensor_view parameter - destination)
/// Input 1: offset (the memory offset to store at, computed by offset op)
/// Input 2: data (the vector of elements to store)
/// No output (side effect only)
struct vector_store
{
    std::size_t size = 1; // Vector size (1, 2, 4, 8, etc.)

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"));
    }

    std::string name() const { return "gpu::gen::vector_store"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(3);
        // No output, just side effect
        return shape{};
    }
};
MIGRAPHX_REGISTER_OP(vector_store);

/// Strided load - loads a single element at a strided position
/// Used for reductions where each thread loads elements at: base + i * stride
/// Input 0: tensor (the tensor_view to load from)
/// Input 1: base index (starting position for this thread)
/// Input 2: iteration index (which iteration of the reduction loop)
/// Input 3: stride (distance between consecutive loads, typically nthreads)
/// Output: loaded value
struct strided_load
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::strided_load"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(4);
        // Output is a scalar of the tensor's element type
        auto tensor_type = inputs.front().type();
        return shape{tensor_type};
    }
};
MIGRAPHX_REGISTER_OP(strided_load);

/// Strided store - stores a value to tensor at an index
/// Input 0: tensor (the tensor_view to store to)
/// Input 1: index (where to store)
/// Input 2: value (the value to store)
/// No output (side effect only)
struct strided_store
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    std::string name() const { return "gpu::gen::strided_store"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(3);
        // No output, just side effect
        return shape{};
    }
};
MIGRAPHX_REGISTER_OP(strided_store);

/// Accumulate - accumulates a value into an accumulator using an operation
/// Used for per-lane accumulation in reductions
/// Input 0: accumulator (current accumulated value)
/// Input 1: value (new value to accumulate)
/// Attribute: op (accumulation operation: "sum", "product", "max", "min")
/// Output: new accumulated value
struct accumulate
{
    std::string op = "sum"; // Accumulation operation

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::gen::accumulate"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);
        // Output has same type as inputs
        return inputs.front();
    }
};
MIGRAPHX_REGISTER_OP(accumulate);

/// Copy - copies data from source to destination tensor
/// Input 0: source tensor
/// Input 1: destination tensor
/// Output: destination tensor (aliased)
struct copy
{
    std::string schedule = "per_thread"; // "per_thread" or "per_block"

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.schedule, "schedule"));
    }

    std::string name() const { return "gpu::gen::copy"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);
        // Output shape matches destination
        return inputs.back();
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};
MIGRAPHX_REGISTER_OP(copy);

/// LDS allocate - allocates shared memory (LDS) for a tile
/// No inputs - shape is specified in the operation
/// Output: a tensor_view pointing to LDS memory
struct lds_allocate
{
    shape s; // Shape of the LDS allocation (with padding to avoid bank conflicts)

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"));
    }

    std::string name() const { return "gpu::gen::lds_allocate"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(0);
        return s;
    }
};
MIGRAPHX_REGISTER_OP(lds_allocate);

// ============================================================================
// Index Transformation Operators
// These operators transform logical indices, enabling fusion of operations
// like pad, gather, and reverse without materializing intermediate tensors.
// ============================================================================

/// Pad index - transforms an index for a padded tensor
/// Returns the adjusted index into the source tensor, or a sentinel value if out of bounds
/// Input 0: index (the logical index in the padded output)
/// Attributes: pads (before/after padding for each dimension), input_shape
/// Output: the transformed index into the source tensor (or -1 if in padding region)
struct pad_index
{
    std::vector<std::size_t> pads =
        {};            // Padding values [before_0, after_0, before_1, after_1, ...]
    shape input_shape; // Shape of the input (unpadded) tensor

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.pads, "pads"), f(self.input_shape, "input_shape"));
    }

    std::string name() const { return "gpu::gen::pad_index"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        // Output is a transformed index (int64 to handle -1 sentinel)
        return shape{shape::int64_type};
    }
};
MIGRAPHX_REGISTER_OP(pad_index);

/// Gather index - transforms an index using a gather indices tensor
/// Input 0: index (the logical index in the output)
/// Input 1: indices tensor (the gather indices)
/// Attributes: axis (the axis to gather along), input_shape
/// Output: the transformed index into the source tensor
struct gather_index
{
    std::size_t axis = 0; // Axis to gather along
    shape input_shape;    // Shape of the input tensor being gathered from

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"), f(self.input_shape, "input_shape"));
    }

    std::string name() const { return "gpu::gen::gather_index"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);
        // Output is a transformed index
        return shape{shape::int64_type};
    }
};
MIGRAPHX_REGISTER_OP(gather_index);

/// Reverse index - transforms an index for a reversed tensor
/// Input 0: index (the logical index in the output)
/// Attributes: axes (dimensions to reverse), input_shape
/// Output: the transformed index into the source tensor
struct reverse_index
{
    std::vector<std::size_t> axes = {}; // Axes to reverse
    shape input_shape;                  // Shape of the input tensor

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.input_shape, "input_shape"));
    }

    std::string name() const { return "gpu::gen::reverse_index"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        // Output is a transformed index
        return shape{shape::int64_type};
    }
};
MIGRAPHX_REGISTER_OP(reverse_index);

/// Shape index - transforms a linear index using tensor shape strides
/// This unifies slice_index, broadcast_index, and transpose_index
/// since they can all be represented as shape-based index transformations.
/// Input 0: linear index in the output space
/// Attribute: input_shape (shape with strides defining the transformation)
/// Output: offset into the source tensor
struct shape_index
{
    shape input_shape; // Shape with strides that define the index transformation

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.input_shape, "input_shape"));
    }

    std::string name() const { return "gpu::gen::shape_index"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        // Output is a transformed index
        return shape{shape::uint64_type};
    }
};
MIGRAPHX_REGISTER_OP(shape_index);

/// Conditional load - loads from tensor if index is valid, otherwise returns fill value
/// Input 0: tensor
/// Input 1: offset (from offset op, may be -1 for invalid indices)
/// Input 2: fill value (used when index is invalid, e.g., for padding)
/// Output: loaded value or fill value
struct conditional_load
{
    std::size_t size = 1; // Vector size

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.size, "size"));
    }

    std::string name() const { return "gpu::gen::conditional_load"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(3);
        auto tensor_type = inputs.front().type();
        return shape{tensor_type, {size}};
    }
};
MIGRAPHX_REGISTER_OP(conditional_load);

// ============================================================================
// Reduction Operators
// These operators implement efficient GPU reductions using DPP and LDS.
// ============================================================================

/// DPP reduce - reduces values within a wavefront using DPP instructions
/// Input 0: value to reduce
/// Attribute: op (the reduction operation: "sum", "product", "max", "min")
/// Output: reduced value (broadcast to all lanes in the wave)
struct dpp_reduce
{
    std::string op = "sum"; // Reduction operation: "sum", "product", "max", "min"

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::gen::dpp_reduce"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        // Output has same type as input
        return inputs.front();
    }
};
MIGRAPHX_REGISTER_OP(dpp_reduce);

/// Reduce waves - reduces values across wavefronts within a workgroup using LDS
/// Input 0: value (the per-wave reduced value, one per wave)
/// Input 1: LDS buffer (allocated via lds_allocate, size = number of waves)
/// Attribute: op (the reduction operation: "sum", "product", "max", "min")
/// Output: final reduced value (available to all threads after barrier)
///
/// The operation flow is:
/// 1. Each wave writes its partial result to LDS[wave_id]
/// 2. Barrier synchronization
/// 3. First wave reads all partial results and reduces them
/// 4. Result is broadcast to all threads
struct reduce_waves
{
    std::string op = "sum"; // Reduction operation: "sum", "product", "max", "min"

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    std::string name() const { return "gpu::gen::reduce_waves"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(2);
        // Output has same type as the input value
        return inputs.front();
    }
};
MIGRAPHX_REGISTER_OP(reduce_waves);

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
