#ifndef MIGRAPHX_GUARD_OPERATORS_ROIALIGN_HPP
#define MIGRAPHX_GUARD_OPERATORS_ROIALIGN_HPP

#include <limits>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/shape_for_each.hpp>
#include <cmath>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct roialign
{
    std::string coord_trans_mode = "half_pixel";
    std::string mode       = "avg";
    int64_t output_height  = 1;
    int64_t output_width   = 1;
    int64_t sampling_ratio = 0;
    float spatial_scale    = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.coord_trans_mode, "coordinate_transformation_mode"),
                    f(self.mode, "mode"),
                    f(self.output_height, "output_height"),
                    f(self.output_width, "output_width"),
                    f(self.sampling_ratio, "sampling_ratio"),
                    f(self.spatial_scale, "spatial_scale"));
    }

    std::string name() const { return "roialign"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3).standard();
        auto x_lens = inputs.at(0).lens();
        auto roi_lens = inputs.at(1).lens();
        auto bi_lens = inputs.at(2).lens();
        auto type  = inputs.at(0).type();

        // check input correct
        if(bi_lens.size() != 1)
        {
            MIGRAPHX_THROW("ROIALIGN: batch indices should be 1 dimension!");
        }

        if(roi_lens.size() != 2 or roi_lens.at(1) != 4)
        {
            MIGRAPHX_THROW("ROIALIGN: rois should be 2 dimensions, and the second dim should be 4!");
        }

        if(roi_lens.front() != bi_lens.front())
        {
            MIGRAPHX_THROW("ROIALIGN: rois and batch indices inputs should have the same number!");
        }

        std::vector<std::size_t> out_lens = x_lens;
        out_lens[0]                       = roi_lens[0];
        out_lens[2]                       = output_height;
        out_lens[3]                       = output_width;

        return {type, out_lens};
    }

    template <class T>
    struct pos_weight
    {
        int64_t pos1;
        int64_t pos2;
        int64_t pos3;
        int64_t pos4;
        T w1;
        T w2;
        T w3;
        T w4;
    };

    template <typename T>
    void calc_pos_weight(const int64_t height,
                         const int64_t width,
                         const shape& comp_s,
                         T roi_start_h,
                         T roi_start_w,
                         T bin_size_h,
                         T bin_size_w,
                         int64_t roi_bin_grid_h,
                         int64_t roi_bin_grid_w,
                         std::vector<pos_weight<T>>& pos_weights) const
    {
        shape_for_each(comp_s, [&](auto idx) {
            auto ph    = idx[0];
            auto pw    = idx[1];
            auto iy    = idx[2];
            auto ix    = idx[3];
            auto index = comp_s.index(idx);
            const T yy = static_cast<T>(roi_start_h + ph * bin_size_h +
                                        static_cast<T>(iy + .5) * bin_size_h /
                                            static_cast<T>(roi_bin_grid_h)); // e.g., 0.5, 1.5
            const T xx = static_cast<T>(roi_start_w + pw * bin_size_w +
                                        static_cast<T>(ix + .5) * bin_size_w /
                                            static_cast<T>(roi_bin_grid_w));

            T x = xx;
            T y = yy;
            // deal with: inverse elements are out of feature map boundary
            if(y < -1.0 || y > height || x < -1.0 || x > width)
            {
                auto& pc = pos_weights[index];
                pc.pos1  = 0;
                pc.pos2  = 0;
                pc.pos3  = 0;
                pc.pos4  = 0;
                pc.w1    = 0;
                pc.w2    = 0;
                pc.w3    = 0;
                pc.w4    = 0;
                return;
            }

            y          = (y <= 0) ? 0 : y;
            x          = (x <= 0) ? 0 : x;
            auto y_low = static_cast<int64_t>(y);
            auto x_low = static_cast<int64_t>(x);
            int64_t y_high;
            int64_t x_high;

            y_high = y_low + 1;
            if(y_low >= height - 1)
            {
                y = y_high = y_low = height - 1;
            }

            x_high = x_low + 1;
            if(x_low >= width - 1)
            {
                x = x_high = x_low = width - 1;
            }

            T ly = static_cast<T>(y - y_low);
            T lx = static_cast<T>(x - x_low);
            T hy = static_cast<T>(1.) - ly;
            T hx = static_cast<T>(1.) - lx;
            T w1 = hy * hx;
            T w2 = hy * lx;
            T w3 = ly * hx;
            T w4 = ly * lx;

            // save weights and indeces
            pos_weight<T> pc;
            pc.pos1            = y_low * width + x_low;
            pc.pos2            = y_low * width + x_high;
            pc.pos3            = y_high * width + x_low;
            pc.pos4            = y_high * width + x_high;
            pc.w1              = w1;
            pc.w2              = w2;
            pc.w3              = w3;
            pc.w4              = w4;
            pos_weights[index] = pc;
        });
    }

    struct max_pool
    {
        double init() { return std::numeric_limits<double>::lowest(); }

        double op(double x, double y)
        {
            double m = std::max(x, y);
            return (m);
        }

        double final(double x, std::size_t) { return (x); }
    };

    struct avg_pool
    {
        double init() { return 0.0; }

        double op(double x, double y) { return x + y; }

        double final(double x, std::size_t y) { return (y == 0) ? 0.0 : (x / y); }
    };

    template <class T, class Op>
    double calc_pooling(const T* data,
                        int64_t roi_bin_grid_h,
                        int64_t roi_bin_grid_w,
                        const std::vector<pos_weight<T>>& pos_weights,
                        int64_t& index,
                        Op op) const
    {
        double output_val   = op.init();
        const int64_t count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4
        for(int64_t iy = 0; iy < roi_bin_grid_h; iy++)
        {
            for(int64_t ix = 0; ix < roi_bin_grid_w; ix++)
            {
                const auto& pc = pos_weights[index];
                output_val     = op.op(output_val, pc.w1 * data[pc.pos1]);
                output_val     = op.op(output_val, pc.w2 * data[pc.pos2]);
                output_val     = op.op(output_val, pc.w3 * data[pc.pos3]);
                output_val     = op.op(output_val, pc.w4 * data[pc.pos4]);
                index += 1;
            }
        }
        output_val = op.final(output_val, count);

        return output_val;
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        const auto& out_lens  = output_shape.lens();
        int64_t n_rois        = out_lens[0];
        int64_t channels      = out_lens[1];
        int64_t pooled_height = out_lens[2];
        int64_t pooled_width  = out_lens[3];
        auto x_lens           = args.at(0).get_shape().lens();
        auto height           = x_lens[2];
        auto width            = x_lens[3];
        auto roi_s            = args.at(1).get_shape();

        visit_all(result, args.at(0), args.at(1))([&](auto output, auto x, auto roi) {
            using T            = typename decltype(output)::value_type;
            auto& arg_ind      = args.at(2);
            auto batch_indices = make_view(arg_ind.get_shape(), arg_ind.data());
            par_for(n_rois, [&](auto n) {
                const T* bottom_data     = x.data();
                const auto roi_batch_ind = batch_indices[n];
                // Do not using rounding; this implementation detail is critical
                T roi_start_w = roi[roi_s.index({n, 0})] * static_cast<T>(spatial_scale);
                T roi_start_h = roi[roi_s.index({n, 1})] * static_cast<T>(spatial_scale);
                T roi_end_w   = roi[roi_s.index({n, 2})] * static_cast<T>(spatial_scale);
                T roi_end_h   = roi[roi_s.index({n, 3})] * static_cast<T>(spatial_scale);

                // Force malformed ROIs to be 1x1
                T roi_width  = (roi_end_w - roi_start_w) > T(1) ? (roi_end_w - roi_start_w) : T(1);
                T roi_height = (roi_end_h - roi_start_h) > T(1) ? (roi_end_h - roi_start_h) : T(1);
                T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
                T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

                // We use roi_bin_grid to sample the grid and mimic integral
                int64_t roi_bin_grid_h =
                    (sampling_ratio > 0)
                        ? sampling_ratio
                        : static_cast<int64_t>(std::ceil(roi_height / pooled_height)); // e.g., = 2
                int64_t roi_bin_grid_w =
                    (sampling_ratio > 0)
                        ? sampling_ratio
                        : static_cast<int64_t>(std::ceil(roi_width / pooled_width));

                // we want to precalculate indices and weights shared by all channels,
                // this is the key point of optimization
                std::vector<pos_weight<T>> pre_calc(roi_bin_grid_h * roi_bin_grid_w * pooled_width *
                                                    pooled_height);
                std::vector<int64_t> lens = {
                    pooled_height, pooled_width, roi_bin_grid_h, roi_bin_grid_w};
                std::vector<std::size_t> comp_lens(lens.begin(), lens.end());
                shape comp_s{shape::float_type, comp_lens};
                this->calc_pos_weight(height,
                                      width,
                                      comp_s,
                                      roi_start_h,
                                      roi_start_w,
                                      bin_size_h,
                                      bin_size_w,
                                      roi_bin_grid_h,
                                      roi_bin_grid_w,
                                      pre_calc);

                std::vector<int64_t> lens1 = {channels, pooled_height, pooled_width};
                std::vector<std::size_t> comp_lens1(lens1.begin(), lens1.end());
                shape comp_s1{migraphx::shape::float_type, comp_lens1};
                std::vector<int64_t> vec_index(channels, 0);
                std::vector<double> vec_outputs(channels);
                shape_for_each(comp_s1, [&](auto idx) {
                    auto c  = idx[0];
                    auto ph = idx[1];
                    auto pw = idx[2];

                    const T* offset_bottom_data =
                        bottom_data +
                        static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);
                    vec_outputs[c] = (mode == "avg") ? this->calc_pooling(offset_bottom_data,
                                                                          roi_bin_grid_h,
                                                                          roi_bin_grid_w,
                                                                          pre_calc,
                                                                          vec_index[c],
                                                                          avg_pool{})
                                                     : this->calc_pooling(offset_bottom_data,
                                                                          roi_bin_grid_h,
                                                                          roi_bin_grid_w,
                                                                          pre_calc,
                                                                          vec_index[c],
                                                                          max_pool{});
                    auto out_idx                        = output_shape.lens();
                    out_idx[0]                          = n;
                    out_idx[1]                          = c;
                    out_idx[2]                          = ph;
                    out_idx[3]                          = pw;
                    output[output_shape.index(out_idx)] = vec_outputs[c];
                });

            });
        });

        return result;
    }

    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
