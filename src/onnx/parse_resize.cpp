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

#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/op/resize.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <vector>
#include <map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

/*
 * Algorithm of calc_neighbor_points():
 * Input: vvv_ind, a collection of neighbors per resized dimension as:
 *               layer-1: (# resized dimensions, vector)
 *               layer-2: (A vector of 2 of: hi/low)
 *               layer-3: Neighor index of every pixel in that output dimension (vector)
 *        in_s,  the original input tensor shape (vector)
 *        out_s, the output tensor shape (vector)
 *    resized_m, lens indices that have to resized (map)
 *
 * Output: per resized pixel, its neighboring hi/lo indexes (vector): all permutations.
 * This api stitches all the neighbors (for every dimension) for a resized pixel,
 * to yield its neighbor index w.r.t to the input shape, in_s.
 */

static std::vector<int>
calc_neighbor_points(const std::vector<std::vector<std::vector<std::size_t>>>& vvv_ind,
                     const shape& in_s,
                     const shape& out_s,
                     const std::map<size_t, size_t>& resized_m)
{
    std::size_t ndims       = out_s.ndim();
    const auto& strides     = out_s.strides();
    std::size_t elements_ct = vvv_ind[0][0].size();

    // This function computes for each element, all permutations of its neighbor indices into an
    // Perm block in one go. (Instead of computing each permutation in isolation per element)
    size_t permutations = 1u << resized_m.size();
    std::vector<std::vector<std::size_t>> perm_blk(permutations, std::vector<size_t>(strides));

    // final outputted vector: permutations of neighbors.
    std::vector<int> out_idx_vec(permutations * elements_ct);

    for(size_t e_idx = 0; e_idx < elements_ct; ++e_idx)
    {
        size_t t_idx = e_idx;
        for(size_t l_idx = 0; l_idx != ndims; ++l_idx)
        {
            auto entry = resized_m.find(l_idx);
            if(entry != resized_m.end())
            {
                size_t hi_cmp_bit = 1u << entry->second;
                auto lo           = vvv_ind[entry->second][0][e_idx];
                auto hi           = vvv_ind[entry->second][1][e_idx];
                for(size_t i = 0; i < permutations; i++)
                    perm_blk[i][l_idx] = ((i & hi_cmp_bit) != 0) ? hi : lo;
            }
            else
            {
                size_t idx = t_idx / strides[l_idx];
                // no permutations in an unmodified lens index, so idx is copied over:
                for(size_t i = 0; i < permutations; i++)
                    perm_blk[i][l_idx] = idx;
            }
            t_idx %= strides[l_idx];
        }
        // write out the permuted indices, calculated off the perm_blk:
        for(size_t i = 0; i < permutations; i++)
            out_idx_vec[e_idx + elements_ct * i] = in_s.index(perm_blk[i]);
    }
    return out_idx_vec;
}

struct parse_resize : op_parser<parse_resize>
{
    std::vector<op_desc> operators() const
    {
        return {{"Resize", "resize"}, {"Upsample", "upsample"}};
    }

    struct resize_attr
    {
        std::vector<int64_t> axes;                // resize - 18
        int antialias             = 0;            // resize - 18
        int exclude_outside       = 0;            // resize - 11
        float cubic_coeff_a       = -0.75f;       // resize - 11
        float extrapolation_value = 0.0f;         // resize - 11
        std::string coord_t_mode  = "half_pixel"; // resize - 11
        std::string nearest_mode  = "round_prefer_floor";
        std::string keep_aspect   = "stretch"; // resize - 18

        // Overlaps with upsample operator
        std::string mode = "nearest";

        // Upsample related
        std::vector<float> scales = {}; // Upsample 7
    };

    struct resize_args
    {
        // Since inception opset(10)
        instruction_ref x;

        // For Upscale this may be an attr
        std::optional<instruction_ref> scales; // resize/upsample-10

        // Added in resize - 11
        // resize 13 makes roi optional
        std::optional<instruction_ref> roi;
        std::optional<instruction_ref> sizes;

        resize_attr r_attr;

        shape in_s;
        std::vector<size_t> in_lens;
        std::vector<size_t> out_lens;
        std::vector<float> vec_scale;
        instruction_ref scales_sizes_arg;

        int opset_version = -1;

        // if scale an attr must be greater or equal to 1
        bool is_scale_attr() const { return not r_attr.scales.empty(); }

        bool is_axes_used() const { return not r_attr.axes.empty(); }

        bool is_constant_scale_input() const { return not vec_scale.empty(); }

        std::string get_nearest_mode() const { return r_attr.nearest_mode; }
        std::string get_coord_trans_mode() const { return r_attr.coord_t_mode; }
        std::string get_mode() const { return r_attr.mode; }
        std::float get_cubic_coeff() const {return r_attr.cubic_coeff_a; }

        void set_scales_sizes_arg(instruction_ref ref) { scales_sizes_arg = ref; }

        instruction_ref get_scales_sizes_arg() const { return scales_sizes_arg; }

        void check_scales_and_inputs() const
        {
            if(in_lens.size() != vec_scale.size())
            {
                MIGRAPHX_THROW("PARSE_RESIZE: ranks of input and scale are different!");
            }
        }

        bool is_output_not_set() const
        {
            return all_of(out_lens.cbegin(), out_lens.cend(), [](auto o) { return o == 0; });
        }

        void compute_output_sizes()
        {
            std::transform(
                in_lens.begin(),
                in_lens.end(),
                vec_scale.begin(),
                out_lens.begin(),
                [&](auto idx, auto scale) { return static_cast<std::size_t>(idx * scale); });
        }

        void compute_scales()
        {
            vec_scale.resize(in_lens.size());
            std::transform(in_lens.begin(),
                           in_lens.end(),
                           out_lens.begin(),
                           vec_scale.begin(),
                           [](auto iss, auto oss) { return 1.0 * oss / iss; });
        }

        void assign_scale_or_size(const std::vector<instruction_ref>& args)
        {
            set_scales_sizes_arg(args[0]);
            if(not is_constant_scale_input())
            {
                // Depending on the args, it *must* populate the `vec_scale`, and might populate
                // `out_lens`. Skip first input and `roi` input (if present)
                size_t args_offset = args.size() > 2 ? 2 : 1;
                std::vector<instruction_ref> inputs{args.begin() + args_offset, args.end()};
                for(const auto& arg : inputs)
                {
                    if(is_arg_invalid(arg))
                        continue;

                    scales_sizes_arg = arg;
                    auto arg_out     = arg->eval();

                    auto type = arg->get_shape().type();
                    if(is_arg_skipped(arg_out))
                        break;

                    if(type == shape::int64_type)
                    { // When input is using sizes
                        assign_output_sizes(arg_out);
                        check_output_size();
                        compute_scales();
                        break;
                    }
                    else if(type == shape::float_type)
                    { // When input is using scales
                        if(is_scale_rank_valid(arg))
                        {
                            assign_scales(arg_out);
                        }
                        break;
                    }
                    else
                    {
                        MIGRAPHX_THROW("PARSE_RESIZE: invalid shape type ");
                    }
                }

                if(vec_scale.empty() and out_lens.empty())
                    MIGRAPHX_THROW("PARSE_RESIZE: no shapes for scales/size input provided");
            }

            if(is_constant_scale_input())
            {
                check_scales_and_inputs();

                if(is_output_not_set())
                {
                    compute_output_sizes();
                }
            }
        }

        void set_coord_trans_mode(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "coordinate_transformation_mode"))
            {
                auto coord_trans_mode = attr.at("coordinate_transformation_mode").s();
                // does not support transformation mode "tf_crop_and_resize"
                if(coord_trans_mode == "tf_crop_and_resize")
                {
                    MIGRAPHX_THROW("PARSE_RESIZE: \"tf_crop_and_resize\" mode is not supported!");
                }
                r_attr.coord_t_mode = coord_trans_mode;
            }
        }

        void set_cubic_coeff(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "cubic_coeff_a"))
            {
                auto coeff           = attr.at("cubic_coeff_a").f();
                r_attr.cubic_coeff_a = coeff;
            }
        }

        void set_mode(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "mode"))
            { 
                auto mode = attr.at("mode").s();
                if(mode != "nearest" and mode != "linear" and mode != "cubic")
                {
                    MIGRAPHX_THROW("PARSE_RESIZE: only cubic, linear, and nearest modes are supported!");
                }
                r_attr.mode = mode;
            }
        }

        void set_nearest_mode(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "nearest_mode"))
            {
                r_attr.nearest_mode = attr.at("nearest_mode").s();
            }
        }

        void set_exclude_outside(const onnx_parser::attribute_map& attr)
        {
            // TODO: Add support for exclude outside = 1
            if(contains(attr, "exclude_outside") and attr.at("exclude_outside").i() == 1)
            {
                MIGRAPHX_THROW("PARSE_RESIZE exclude_outside 1 is not supported!");
            }
        }

        void set_extrapolation_val(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "extrapolation_value"))
            {
                r_attr.extrapolation_value = attr.at("extrapolation_value").f();
            }
        }

        void set_axes(const onnx_parser::attribute_map& attr)
        {
            // TODO: support implementation of 'axes' attribute.
            // For now, it's used to check the length of 'sizes' input (if present)
            if(contains(attr, "axes"))
            {
                auto&& axes_vals = attr.at("axes").ints();
                r_attr.axes      = std::vector<int64_t>(axes_vals.begin(), axes_vals.end());
            }
        }

        void set_aspect_ratio_policy(const onnx_parser::attribute_map& attr,
                                     const std::vector<instruction_ref>& args) const
        {
            // TODO: Add support for this instead of keeping it as a check
            if(contains(attr, "keep_aspect_ratio_policy"))
            {
                shape last_arg_shape     = args.back()->get_shape();
                size_t last_arg_elements = last_arg_shape.elements();
                // Check if the last arg is 'sizes' input.
                // This attribute is only relevant if 'sizes' input is used.
                // The shape constraints for 'sizes' are below:
                if(last_arg_shape.type() == shape::int64_type and
                   (last_arg_elements == args.front()->get_shape().ndim() or
                    (is_axes_used() and last_arg_elements == r_attr.axes.size())))
                {
                    MIGRAPHX_THROW("PARSE_RESIZE: keep_aspect_ratio_policy is not supported!");
                }
            }
        }

        // "scales" is an attribute of the deprecated Upsample op. ver7 only
        void set_scales(const onnx_parser::attribute_map& attr)
        {
            if(contains(attr, "scales"))
            {
                copy(attr.at("scales").floats(), std::back_inserter(r_attr.scales));
                vec_scale = r_attr.scales;
                compute_output_sizes();
            }
        }

        bool is_arg_skipped(const argument& arg) const { return arg.empty(); }

        bool is_arg_invalid(const instruction_ref arg) const
        {
            if(arg->name() == "undefined")
                return true;

            // skip any empty input (some of the Onnx args. are optional)
            auto lens = arg->get_shape().lens();
            return lens.empty();
        }

        void check_output_size() const
        {
            if(out_lens.size() != in_lens.size())
            {
                MIGRAPHX_THROW(
                    "PARSE_RESIZE: specified output size's rank does not match input size");
            }
        }

        void assign_output_sizes(const argument& arg_out)
        {
            arg_out.visit([&](const auto& ol) { out_lens.assign(ol.begin(), ol.end()); });
        }

        void assign_scales(const argument& arg_out)
        {
            arg_out.visit([&](const auto& v) { vec_scale.assign(v.begin(), v.end()); });
        }

        bool is_scale_rank_valid(const instruction_ref arg) const
        {
            return arg->get_shape().lens().at(0) == in_lens.size();
        }

        size_t get_input_rank() const
        {
            return in_lens.size();
        }

        // Get Dimension of the data - relevant to how we'll scale and dimensions
        // if dims are 2 or 3 they're treated as input with channel and batch set to 1

        // 2D - (batch, channel, height, width) or (height, width)
        bool is_2d_image() const
        {
            return (get_input_rank() == 2 or get_input_rank() == 4);
        }

        // 3D - (batch, channel, height, width, depth) or (height, width, depth)
        bool is_3d_image() const
        {
            return (get_input_rank() == 3 or get_input_rank() == 5);
        }
    };


    // Helper to add a "reshape" and "gather" instruction.  These can implement
    // Nearest mode resizing if all sizes are known at compile time.
    static instruction_ref make_gather_instruction(const onnx_parser::node_info& info,
                                                   resize_args& resize,
                                                   instruction_ref args_0)
    {
        auto in_s      = resize.in_s;
        auto in_lens   = resize.in_lens;
        auto out_lens  = resize.out_lens;
        auto vec_scale = resize.vec_scale;

        shape out_s{in_s.type(), out_lens};
        std::size_t out_elements = out_s.elements();
        std::string nearest_mode = resize.get_nearest_mode();
        std::vector<int> ind(out_elements);

        // map out_idx to in_idx
        auto nearest_op              = op::resize::get_nearest_op(nearest_mode);
        std::string coord_trans_mode = resize.get_coord_trans_mode();
        auto idx_op                  = op::resize::get_original_idx_op(coord_trans_mode);

        shape_for_each(out_s, [&](const auto& out_idx_v, size_t out_idx) {
            std::vector<size_t> in_idx(out_idx_v.size());
            for(auto ii = 0; ii < in_lens.size(); ++ii)
            {
                auto idx_val = idx_op(in_lens[ii], out_lens[ii], out_idx_v[ii], vec_scale[ii]);
                in_idx[ii]   = nearest_op(in_lens[ii], idx_val);
            }

            ind[out_idx] = static_cast<int64_t>(in_s.index(in_idx));
        });
        // reshape input to one-dimension
        std::vector<int64_t> rsp_lens = {static_cast<int64_t>(in_s.elements())};
        auto rsp = info.add_instruction(make_op("reshape", {{"dims", rsp_lens}}), args_0);

        // ins_ind should be a multi dimensional index that will restore original rank
        shape ind_s{shape::int32_type, out_lens};
        auto ins_ind = info.add_literal(literal(ind_s, ind));
        return info.add_instruction(make_op("gather", {{"axis", 0}}), rsp, ins_ind);
    }

    static instruction_ref handle_nearest_neighbor(const onnx_parser::node_info& info,
                                                   resize_args& resize,
                                                   instruction_ref args_0)
    {
        if(args_0->get_shape().dynamic() or not resize.is_constant_scale_input())
        {
            // Resize's compute_shape() will read scales_sizes_arg as "scales" or "sizes"
            // depending on its data type
            return info.add_instruction(
                make_op("resize",
                        {{"nearest_mode", resize.get_nearest_mode()},
                         {"coordinate_transformation_mode", resize.get_coord_trans_mode()}}),
                args_0,
                resize.get_scales_sizes_arg());
        }
        else
        {
            // If there are no dynamic shapes and size/scale attributes are literals, then
            // all the indexes can be calculated now at compile time and
            // the Resize can be accomplished with Gather operation.  Preferred for
            // better performance.

            return make_gather_instruction(info, resize, args_0);
        }
    }

    static instruction_ref handle_linear_mode(const op_desc& opd,
                                              const onnx_parser::node_info& info,
                                              resize_args& resize,
                                              instruction_ref& args_0)

    {
        auto in_s      = resize.in_s;
        auto in_lens   = resize.in_lens;
        auto out_lens  = resize.out_lens;
        auto vec_scale = resize.vec_scale;

        // out_lens and other variables can't be populated if non-constant (runtime) size
        // inputs.
        if(not resize.is_constant_scale_input())
            MIGRAPHX_THROW("PARSE_" + opd.onnx_name +
                           ": linear mode not supported for non-constant inputs");

        if(in_lens == out_lens)
            return args_0; // if input and output shapes are the same, return the input

        shape out_s{in_s.type(), out_lens};

        // reshape input to one-dimension
        std::vector<int64_t> rsp_lens = {static_cast<int64_t>(in_s.elements())};
        auto rsp = info.add_instruction(make_op("reshape", {{"dims", rsp_lens}}), args_0);

        auto nearest_floor = op::resize::get_nearest_op("floor");
        auto nearest_ceil  = op::resize::get_nearest_op("ceil");

        std::vector<size_t> resized_axes; // vector of dimensions to be resized
        std::size_t out_elements = 1;     // total number of elements to be resized
        size_t resized_ct        = 0;
        std::map<size_t, size_t> resized_m; // modified indices --> vvv_ind index below
        for(std::size_t axis = 0; axis != out_lens.size(); ++axis)
        {
            out_elements *= out_lens[axis];
            if(in_lens[axis] == out_lens[axis])
                continue;
            resized_axes.push_back(axis);
            resized_m[axis] = resized_ct++;
        }

        // Neighbor indices. For an axis. Two sets of max/min per element:
        std::vector<std::vector<std::size_t>> vv_ind(2, std::vector<std::size_t>(out_elements));
        // Neighbor indices. For all resized axes:
        std::vector<std::vector<std::vector<std::size_t>>> vvv_ind(resized_ct, vv_ind);
        // Delta list. For each resized axes - per element.
        std::vector<std::vector<float>> delta(resized_ct, std::vector<float>(out_elements));

        auto idx_op = op::resize::get_original_idx_op(resize.get_coord_trans_mode());
        shape_for_each(out_s, [&](const auto& out_idx_v, std::size_t out_idx) {
            for(size_t ii = 0; ii != resized_ct; ++ii)
            {
                auto idx     = resized_axes[ii];
                auto idx_val = idx_op(in_lens[idx], out_lens[idx], out_idx_v[idx], vec_scale[idx]);
                vvv_ind[ii][0][out_idx] = nearest_floor(in_lens[idx], idx_val);
                vvv_ind[ii][1][out_idx] = nearest_ceil(in_lens[idx], idx_val);
                delta[ii][out_idx]      = idx_val - vvv_ind[ii][0][out_idx];
            }
        });

        auto ind = calc_neighbor_points(vvv_ind, in_s, out_s, resized_m);

        auto dim_lens = out_lens;
        // indices matrix size grows 2x per resized-axis:
        dim_lens[0] *= (1u << resized_ct);
        shape ind_s{shape::int32_type, dim_lens};
        auto ins_ind = info.add_literal(literal(ind_s, ind));
        auto data    = info.add_instruction(make_op("gather", {{"axis", 0}}), rsp, ins_ind);

        for(auto idx = resized_ct; idx != 0u; --idx)
        {
            dim_lens[0] /= 2; // halved for 2 slices of data (hi & low below)
            shape dim_s{in_s.type(), dim_lens};
            const auto& dim_delta = delta[idx - 1];
            std::vector<float> delta_data;
            for(std::size_t j = 0; j < dim_lens[0] / out_lens[0]; ++j)
                delta_data.insert(delta_data.begin(), dim_delta.begin(), dim_delta.end());
            auto ins_delta = info.add_literal(dim_s, delta_data);

            // slice the data
            int64_t slc_stride = dim_lens[0];
            auto low           = info.add_instruction(
                make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {slc_stride}}}), data);
            auto hi = info.add_instruction(
                make_op("slice",
                        {{"axes", {0}}, {"starts", {slc_stride}}, {"ends", {2 * slc_stride}}}),
                data);
            auto diff = info.add_instruction(make_op("sub"), hi, low);
            auto ddf  = info.add_instruction(make_op("mul"), diff, ins_delta);
            data      = info.add_instruction(make_op("add"), ddf, low);
        }
        return data;
    }

    static instruction_ref handle_cubic_mode(const op_desc& opd,
                                             const onnx_parser::node_info& info,
                                             resize_args& resize,
                                             instruction_ref& args_0)
    {

    }

    static void set_resize_attributes(const onnx_parser::node_info& info,
                                      const std::vector<instruction_ref>& args,
                                      resize_args& resize)
    {
        resize.set_coord_trans_mode(info.attributes);
        resize.set_cubic_coeff(info.attributes);
        resize.set_axes(info.attributes);
        resize.set_exclude_outside(info.attributes);
        resize.set_extrapolation_val(info.attributes);
        resize.set_aspect_ratio_policy(info.attributes, args);
        resize.set_nearest_mode(info.attributes);
        resize.set_mode(info.attributes);
    }

    static void set_resize_args(const std::vector<instruction_ref>& args, resize_args& resize)
    {
        resize.x = args.at(0);
        resize.assign_scale_or_size(args);
    }

    static void set_upsample_attributes(const onnx_parser::node_info& info, resize_args& resize)
    {
        resize.set_mode(info.attributes);
        resize.set_scales(info.attributes);
    }

    static void set_upsample_args(const std::vector<instruction_ref>& args, resize_args& resize)
    {
        resize.x = args.at(0);

        // scale is input it must be a required input
        if(not resize.is_scale_attr())
            resize.assign_scale_or_size(args);
    }

    // Split of what we handle since this parser is used for both resize/upscale operators
    static resize_args handle_inputs(const op_desc& opd,
                                     const onnx_parser::node_info& info,
                                     const std::vector<instruction_ref>& args)
    {
        resize_args resize;

        // input data shape info
        resize.in_s    = args[0]->get_shape().to_static(1);
        resize.in_lens = resize.in_s.lens();

        // output shape is explicitly specified
        resize.out_lens = std::vector<size_t>(resize.in_lens.size());

        if(opd.op_name == "upsample")
        {
            set_upsample_attributes(info, resize);
            set_upsample_args(args, resize);
        }
        else
        {
            set_resize_attributes(info, args, resize);
            set_resize_args(args, resize);
        }

        return resize;
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser&,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto resize = handle_inputs(opd, info, args);

        if(resize.get_mode() == "nearest")
        {
            return handle_nearest_neighbor(info, resize, args[0]);
        }
        else if(resize.get_mode() == "linear")
        {
            return handle_linear_mode(opd, info, resize, args[0]);
        }
        else
        {
            return handle_cubic_mode(opd, info, resize, args[0]);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
