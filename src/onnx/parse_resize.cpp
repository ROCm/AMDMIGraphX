/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/shape_for_each.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

const auto& get_nearest_op(const std::string& mode)
{
    using nearest_op = std::function<std::size_t(std::size_t, double)>;
    static std::unordered_map<std::string, nearest_op> const nearest_ops = {
        {"round_prefer_floor",
         [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::ceil((val - 0.5)));
         }},
        {"round_prefer_ceil",
         [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::round((val)));
         }},
        {"floor",
         [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::floor((val)));
         }},
        {"ceil", [=](std::size_t d_in, double val) {
             val = std::max(0.0, std::min(d_in - 1.0, val));
             return static_cast<std::size_t>(std::ceil((val)));
         }}};

    if(not contains(nearest_ops, mode))
    {
        MIGRAPHX_THROW("PARSE_RESIZE: nearest_mode " + mode + " not supported!");
    }

    return nearest_ops.at(mode);
}

const auto& get_original_idx_op(const std::string& mode)
{
    using original_idx_op = std::function<double(std::size_t, std::size_t, std::size_t, double)>;
    static std::unordered_map<std::string, original_idx_op> const idx_ops = {
        {"half_pixel",
         [=](std::size_t, std::size_t, std::size_t idx, double scale) {
             return (idx + 0.5) / scale - 0.5;
         }},
        {"pytorch_half_pixel",
         [=](std::size_t, std::size_t l_out, std::size_t idx, double scale) {
             return l_out > 1 ? (idx + 0.5) / scale - 0.5 : 0.0;
         }},
        {"align_corners",
         [=](std::size_t l_in, std::size_t l_out, std::size_t idx, double) {
             return (l_out == 1) ? 0.0 : (1.0 * idx * (l_in - 1.0) / (l_out - 1.0));
         }},
        {"asymmetric",
         [=](std::size_t, std::size_t, std::size_t idx, double scale) { return idx / scale; }},
        {"tf_half_pixel_for_nn", [=](std::size_t, std::size_t, std::size_t idx, double scale) {
             return (idx + 0.5) / scale;
         }}};

    if(not contains(idx_ops, mode))
    {
        MIGRAPHX_THROW("PARSE_RESIZE: coordinate_transformation_mode " + mode + " not supported!");
    }

    return idx_ops.at(mode);
}

static std::vector<int>
calc_neighbor_points(const std::vector<std::vector<std::vector<std::size_t>>>& vvv_ind,
                     int i_dim,
                     std::vector<std::vector<std::size_t>> vec_dims,
                     const shape& in_s)
{
    if(i_dim == vvv_ind.size())
    {
        std::vector<int> vec_ind(vec_dims.size());
        std::transform(vec_dims.begin(), vec_dims.end(), vec_ind.begin(), [&](auto idx) {
            return static_cast<int>(in_s.index(idx));
        });
        return vec_ind;
    }

    const auto& vv_lo = vvv_ind[i_dim][0];
    std::vector<std::vector<std::size_t>> vec_dims1;
    for(std::size_t start = 0; start < vec_dims.size(); start += vv_lo.size())
    {
        std::transform(vv_lo.begin(),
                       vv_lo.end(),
                       vec_dims.begin() + start,
                       std::back_inserter(vec_dims1),
                       [](auto i, auto dim) {
                           dim.push_back(i);
                           return dim;
                       });
    }

    const auto& vv_hi = vvv_ind[i_dim][1];
    for(std::size_t start = 0; start < vec_dims.size(); start += vv_hi.size())
    {
        std::transform(vv_hi.begin(),
                       vv_hi.end(),
                       vec_dims.begin() + start,
                       std::back_inserter(vec_dims1),
                       [](auto i, auto dim) {
                           dim.push_back(i);
                           return dim;
                       });
    }
    vec_dims.clear();
    return calc_neighbor_points(vvv_ind, i_dim + 1, std::move(vec_dims1), in_s);
}

static std::string get_coord_trans_mode(const onnx_parser::attribute_map& attr)
{
    std::string coord_trans_mode = "half_pixel";
    if(contains(attr, "coordinate_transformation_mode"))
    {
        coord_trans_mode = attr.at("coordinate_transformation_mode").s();
        // does not support transformation mode "tf_crop_and_resize"
        if(coord_trans_mode == "tf_crop_and_resize")
        {
            MIGRAPHX_THROW("PARSE_RESIZE: \"tf_crop_and_resize\" mode is not supported!");
        }
    }

    return coord_trans_mode;
}

static std::string get_mode(const onnx_parser::attribute_map& attr)
{
    std::string mode = "nearest";
    if(contains(attr, "mode"))
    {
        mode = attr.at("mode").s();
        if(mode != "nearest" and mode != "linear")
        {
            MIGRAPHX_THROW("PARSE_RESIZE: only nearest and linear modes are supported!");
        }
    }

    return mode;
}

static std::string get_nearest_mode(const onnx_parser::attribute_map& attr)
{
    std::string nearest_mode = "round_prefer_floor";
    if(contains(attr, "nearest_mode"))
    {
        nearest_mode = attr.at("nearest_mode").s();
    }

    return nearest_mode;
}

struct parse_resize : op_parser<parse_resize>
{
    std::vector<op_desc> operators() const { return {{"Resize"}, {"Upsample"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        // coord transform mode
        std::string coord_trans_mode = get_coord_trans_mode(info.attributes);

        // mode: only nearest and linear modes are supported for now
        std::string mode = get_mode(info.attributes);

        // nearest mode
        std::string nearest_mode = get_nearest_mode(info.attributes);

        // check exclude_outside, only support 0
        if(contains(info.attributes, "exclude_outside") and
           info.attributes.at("exclude_outside").i() == 1)
        {
            MIGRAPHX_THROW("PARSE_" + opd.op_name + ": exclude_outside 1 is not supported!");
        }

        // input data shape info.  Convert static lens to dynamic to simplify referencing them later
        auto in_s                                               = args[0]->get_shape().to_dynamic();
        std::vector<migraphx::shape::dynamic_dimension> in_dims = in_s.dyn_dims();

        // output shape is explicitly specified
        std::vector<size_t> out_lens(in_dims.size());

        // scale
        std::vector<double> vec_scale;

        // Infer either output size or scale, depending on input type
        for(const auto& arg : args)
        {
            if(arg->name() == "undefined" or arg == args.front())
            {
                continue;
            }

            // this is just developer code, figure out real requirement
            if(arg != args[0] and arg->get_shape().dynamic())
            {
                MIGRAPHX_THROW("parse_resize:  no other dynamic shapes allowed");
            }

            // skipped empty input
            auto lens = arg->get_shape().lens();
            if(lens.empty())
            {
                continue;
            }

            auto type = arg->get_shape().type();
            // output size
            if(type == shape::int64_type)
            {
                auto arg_out_s = arg->eval();
                check_arg_empty(arg_out_s,
                                "PARSE_" + opd.op_name + ": dynamic output size is not supported!");

                // reallocate a vector and copy the values to it.  All dimensions except batch, even
                // if originally dynamic, are required to be fixed so we can refer to their max
                // value WLOG.
                arg_out_s.visit([&](auto ol) {
                    // todo:  assign doesn't work with dynamic shapes
                    auto ols = ol.get_shape().to_dynamic();
                    for(auto it = ols.dyn_dims().begin(); it != ols.dyn_dims().end(); it++)
                    {
                        out_lens.push_back(it->max);
                    }

                    // out_lens.assign(ol.begin(), ol.end());
                });

                if(out_lens.size() != in_dims.size())
                {
                    MIGRAPHX_THROW("PARSE_" + opd.op_name +
                                   ": specified output size does not match input size");
                }

                // compute the scale in each dimension
                vec_scale.resize(in_dims.size());

                std::transform(in_dims.begin(),
                               in_dims.end(),
                               out_lens.begin(),
                               vec_scale.begin(),
                               [](auto iss, auto oss) { return double(1.0 * oss / iss.max); });
            }
            else
            {
                // scale input
                if(lens[0] == in_dims.size())
                {
                    auto arg_scale = arg->eval();
                    check_arg_empty(arg_scale,
                                    "PARSE_" + opd.op_name +
                                        ": dynamic input scale is not supported!");

                    arg_scale.visit([&](auto v) { vec_scale.assign(v.begin(), v.end()); });
                    if(in_dims.size() != vec_scale.size())
                    {
                        MIGRAPHX_THROW("PARSE_" + opd.op_name +
                                       ": ranks of input and scale are different!");
                    }

                    std::transform(in_dims.begin(),
                                   in_dims.end(),
                                   vec_scale.begin(),
                                   out_lens.begin(),
                                   [&](auto idx, auto scale) {
                                       // inferred output size is floor(idx.max * scale)
                                       return idx.max * scale;
                                   });
                }
            }
        }

        // Dynamic batch:  Only args[0] can have a dynamic shape, only the 0'th
        // dimension--batch size--can be non-fixed, and the only resize mode allowed is "nearest"
        if(args[0]->get_shape().dynamic())
        {
            if(mode == "nearest")
            {
                auto some_dims = args[0]->get_shape().dyn_dims();

                bool mostly_fixed =
                    std::all_of(some_dims.begin() + 1,
                                some_dims.end(),
                                [](shape::dynamic_dimension dd) { return dd.is_fixed(); });

                if(not mostly_fixed)
                    MIGRAPHX_THROW(
                        "PARSE_" + opd.op_name +
                        ": dynamic shape inputs other than batch size are not supported");

                // TODO:  Add support for channel dimension

                // take max_lens() to get static dimension set
                // Drop the 0'th dimension,
                auto fixed_dims = args[0]->get_shape().max_lens();
                fixed_dims.erase(fixed_dims.begin());
                // dimensions of the (scaled) output, also with the 0'th dimension dropped
                auto fixed_out_lens = out_lens;
                fixed_out_lens.erase(fixed_out_lens.begin());

                // create a shape with the scaled lens and no batch dimension
                migraphx::shape static_out_shape(args[0]->get_shape().type(), fixed_out_lens);

                size_t out_elements = std::accumulate(fixed_out_lens.begin(),
                                                      fixed_out_lens.end(),
                                                      std::size_t{1},
                                                      std::multiplies<>());
                std::vector<int> ind(out_elements);

                //               map out_idx to in_idx
                auto idx_op     = get_original_idx_op(coord_trans_mode);
                auto nearest_op = get_nearest_op(nearest_mode);

                // For each element of static_out_shape, find the matching location of input shape.
                // The indexes we find will be an argument to the gather op.
                shape_for_each(static_out_shape, [&](const auto& out_idx_v, size_t out_idx) {
                    std::vector<size_t> in_idx(out_idx_v.size());
                    for(auto ii = 0; ii < fixed_dims.size(); ++ii)
                    {
                        // Convert this index by scaling.  Inefficient since indexes are repeated
                        auto idx_val = idx_op(
                            fixed_dims[ii], fixed_out_lens[ii], out_idx_v[ii], vec_scale[ii]);
                        // round the scaled value to an index
                        in_idx[ii] = nearest_op(fixed_dims[ii], idx_val);
                    }

                    ind[out_idx] = static_cast<int64_t>(static_out_shape.index(in_idx));
                });

                // Create a static shape that's just like the scaled out_lens except we set to 1 the
                // 0'th dimension of output, later to be broadcasted to dynamic batch size
                out_lens[0] = 1;
                shape ind_s{shape::int32_type, out_lens};
                auto ins_ind = info.add_literal(literal(ind_s, ind));

                // define a dynamic shape including the batch dimension
                std::vector<shape::dynamic_dimension> out_dyn_dims(in_dims.size());
                out_dyn_dims[0] = in_dims[0];
                std::transform(fixed_out_lens.begin(),
                               fixed_out_lens.end(),
                               out_dyn_dims.begin() + 1,
                               [&](auto len) {
                                   return shape::dynamic_dimension{len, len};
                               });
                shape dyn_out_shape{in_s.type(), out_dyn_dims};

                // allocate op to create the output argument we want
                auto ins_dyn_out =
                    info.add_instruction(make_op("allocate", {{"shape", to_value(dyn_out_shape)}}));

                // multibroadcast op to convert static ins_ind to a dynamic shape
                auto ins_dyn =
                    info.add_instruction(make_op("multibroadcast"), ins_ind, ins_dyn_out);

                return info.add_instruction(make_op("gather", {{"axis", 0}}), args[0], ins_dyn);
            }
            else
            {
                MIGRAPHX_THROW("PARSE_RESIZE: only nearest_mode supports dynamic batch size input");
            }
        }
        else
        {
            //
            //        Static input shape.
            //
            in_s         = args[0]->get_shape();
            auto in_lens = args[0]->get_shape().lens();

            shape out_s{in_s.type(), out_lens};
            std::size_t out_elements = out_s.elements();
            auto idx_op              = get_original_idx_op(coord_trans_mode);

            // reshape input to one-dimension
            // TODO:  We did this in multi dimensions in the dynamic case.  Can we do
            //          the same here?
            std::vector<int64_t> rsp_lens = {static_cast<int64_t>(in_s.elements())};
            args[0]                       = info.make_contiguous(args[0]);
            auto rsp = info.add_instruction(make_op("reshape", {{"dims", rsp_lens}}), args[0]);

            if(mode == "nearest")
            {
                std::vector<int> ind(out_elements);

                // map out_idx to in_idx
                auto nearest_op = get_nearest_op(nearest_mode);
                shape_for_each(out_s, [&](const auto& out_idx_v, size_t out_idx) {
                    std::vector<size_t> in_idx(out_idx_v.size());
                    for(auto ii = 0; ii < in_lens.size(); ++ii)
                    {
                        auto idx_val =
                            idx_op(in_lens[ii], out_lens[ii], out_idx_v[ii], vec_scale[ii]);
                        in_idx[ii] = nearest_op(in_lens[ii], idx_val);
                    }

                    ind[out_idx] = static_cast<int64_t>(in_s.index(in_idx));
                });

                shape ind_s{shape::int32_type, out_lens};
                auto ins_ind = info.add_literal(literal(ind_s, ind));
                return info.add_instruction(make_op("gather", {{"axis", 0}}), rsp, ins_ind);
            }
            // linear mode
            else
            {
                auto nearest_floor = get_nearest_op("floor");
                auto nearest_ceil  = get_nearest_op("ceil");

                // get the number of dimensions
                std::size_t n_dim = out_lens.size();
                auto vvv_ind =
                    std::vector(n_dim, std::vector(2, std::vector<size_t>(out_elements)));
                std::vector<std::vector<float>> delta(n_dim, std::vector<float>(out_elements));

                shape_for_each(out_s, [&](const auto& out_idx_v, size_t out_idx) {
                    for(auto ii = 0; ii < in_lens.size(); ++ii)
                    {
                        auto idx_val =
                            idx_op(in_lens[ii], out_lens[ii], out_idx_v[ii], vec_scale[ii]);
                        vvv_ind[ii][0][out_idx] = nearest_floor(in_lens[ii], idx_val);
                        vvv_ind[ii][1][out_idx] = nearest_ceil(in_lens[ii], idx_val);
                        delta[ii][out_idx]      = idx_val - vvv_ind[ii][0][out_idx];
                    }
                });

                auto ind = calc_neighbor_points(
                    vvv_ind, 0, std::vector<std::vector<std::size_t>>(out_elements), in_s);
                auto ind_lens = out_lens;
                ind_lens[0] *= (std::size_t{1} << n_dim);
                shape ind_s{shape::int32_type, ind_lens};
                auto ins_ind = info.add_literal(literal(ind_s, ind));
                auto data    = info.add_instruction(make_op("gather", {{"axis", 0}}), rsp, ins_ind);

                auto dim_lens = out_lens;
                dim_lens[0] *= (std::size_t{1} << (n_dim - 1));
                for(std::size_t i = 0; i < n_dim; ++i)
                {
                    shape dim_s{shape::float_type, dim_lens};
                    const auto& dim_delta = delta[n_dim - i - 1];
                    std::vector<float> delta_data;
                    for(std::size_t j = 0; j < dim_lens[0] / out_lens[0]; ++j)
                    {
                        delta_data.insert(delta_data.begin(), dim_delta.begin(), dim_delta.end());
                    }
                    auto ins_delta = info.add_literal(dim_s, delta_data);

                    // slice the data
                    int64_t slc_stride = dim_lens[0];
                    auto low           = info.add_instruction(
                        make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {slc_stride}}}),
                        data);
                    auto hi = info.add_instruction(
                        make_op(
                            "slice",
                            {{"axes", {0}}, {"starts", {slc_stride}}, {"ends", {2 * slc_stride}}}),
                        data);
                    auto diff = info.add_instruction(make_op("sub"), hi, low);
                    auto ddf  = info.add_instruction(make_op("mul"), diff, ins_delta);
                    data      = info.add_instruction(make_op("add"), ddf, low);
                    dim_lens[0] /= 2;
                }

                return data;
            }
        }
    }
};

} // namespace onnx

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
