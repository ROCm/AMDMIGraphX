#ifndef MIGRAPHX_GUARD_OPERATORS_NONMAXSUPPRESSION_HPP
#define MIGRAPHX_GUARD_OPERATORS_NONMAXSUPPRESSION_HPP

#include <cmath>
#include <queue>
#include <cstdint>
#include <iterator>
#include <migraphx/config.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/output_iterator.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct nonmaxsuppression
{
    bool center_point_box = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.center_point_box, "center_point_box"));
    }

    std::string name() const { return "nonmaxsuppression"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // requires at least 2 inputs
        check_shapes{inputs, *this}.standard();
        check_shapes{{inputs.at(0), inputs.at(1)}, *this}.only_dims(3);
        auto lens = inputs.front().lens();

        // check input shape
        if(lens[1] != inputs.at(1).lens()[2])
        {
            MIGRAPHX_THROW("NonMaxSuppression: dimension mismatch between first and second input!");
        }

        std::vector<int64_t> out_lens(2);
        out_lens.at(0) = lens.at(1);
        out_lens.at(1) = 3;
        return {shape::int64_type, out_lens};
    }

    struct box
    {
        std::array<float, 2> x;
        std::array<float, 2> y;

        void sort()
        {
            std::sort(x.begin(), x.end());
            std::sort(y.begin(), y.end());
        }

        std::array<float, 2>& operator[](std::size_t i) { return i == 0 ? x : y; }

        float area() const
        {
            assert(std::is_sorted(x.begin(), x.end()));
            assert(std::is_sorted(y.begin(), y.end()));
            return (x[1] - x[0]) * (y[1] - y[0]);
        }
    };

    template <class T>
    box batch_box(const T* boxes, std::size_t bidx) const
    {
        box result{};
        const T* start = boxes + 4 * bidx;
        if(center_point_box)
        {
            float half_width  = start[2] / 2.0f;
            float half_height = start[3] / 2.0f;
            float x_center    = start[0];
            float y_center    = start[1];
            result.x          = {x_center - half_width, x_center + half_width};
            result.y          = {y_center - half_height, y_center + half_height};
        }
        else
        {
            result.x = {start[1], start[3]};
            result.y = {start[0], start[2]};
        }

        return result;
    }

    inline bool suppress_by_iou(box b1, box b2, float iou_threshold) const
    {
        b1.sort();
        b2.sort();

        box intersection{};
        for(auto i : range(2))
        {
            intersection[i][0] = std::max(b1[i][0], b2[i][0]);
            intersection[i][1] = std::min(b1[i][1], b2[i][1]);
        }

        std::vector<std::array<float, 2>> bbox = {intersection.x, intersection.y};
        if(std::any_of(bbox.begin(), bbox.end(), [](auto bx) {
               return not std::is_sorted(bx.begin(), bx.end());
           }))
        {
            return false;
        }

        const float area1             = b1.area();
        const float area2             = b2.area();
        const float intersection_area = intersection.area();
        const float union_area        = area1 + area2 - intersection_area;

        if(area1 <= .0f or area2 <= .0f or union_area <= .0f)
        {
            return false;
        }

        const float intersection_over_union = intersection_area / union_area;

        return intersection_over_union > iou_threshold;
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};

        result.visit([&](auto out) { std::fill(out.begin(), out.end(), 0); });

        std::size_t max_output_boxes_per_class = 0;
        float iou_threshold                    = 0.0f;
        float score_threshold                  = 0.0f;

        if(args.size() > 2)
        {
            max_output_boxes_per_class = args.at(2).at<std::size_t>();
        }
        // max_output_boxes_per_class is 0, no output
        if(max_output_boxes_per_class == 0)
        {
            return result;
        }

        if(args.size() > 3)
        {
            iou_threshold = args.at(3).at<float>();
        }

        if(args.size() > 4)
        {
            score_threshold = args.at(4).at<float>();
        }

        const auto& lens = args.at(1).get_shape().lens();
        auto batch_num   = lens[0];
        auto class_num   = lens[1];
        auto box_num     = args.at(0).get_shape().lens()[1];

        std::vector<std::pair<float, int64_t>> selected_boxes_inside_class;
        std::vector<int64_t> selected_indices;
        selected_boxes_inside_class.reserve(output_shape.elements());

        auto scores        = make_view<float>(args.at(1).get_shape(), args.at(1).cast<float>());
        const float* boxes = args.at(0).cast<float>();
        shape comp_s{shape::float_type, {batch_num, class_num}};
        shape_for_each(comp_s, [&](auto idx) {
            auto bidx = idx[0];
            auto cidx = idx[1];

            std::size_t score_offset = (bidx * class_num + cidx) * box_num;
            const float* batch_boxes = boxes + bidx * box_num * 4;
            std::priority_queue<std::pair<float, int64_t>> sorted_boxes;
            auto insert_to_sorted_boxes =
                make_function_output_iterator([&](const auto& x) { sorted_boxes.push(x); });

            int64_t box_idx = 0;
            transform_if(
                scores.begin() + score_offset,
                scores.begin() + score_offset + box_num,
                insert_to_sorted_boxes,
                [&](auto sc) {
                    box_idx++;
                    return sc >= score_threshold;
                },
                [&](auto sc) { return std::make_pair(sc, box_idx - 1); });

            selected_boxes_inside_class.clear();
            // Get the next box with top score, filter by iou_threshold
            while(!sorted_boxes.empty() &&
                  selected_boxes_inside_class.size() < max_output_boxes_per_class)
            {
                const std::pair<float, int64_t>& next_top_score = sorted_boxes.top();

                // Check with existing selected boxes for this class, suppress if exceed the IOU
                // (Intersection Over Union) threshold
                bool not_selected = std::any_of(
                    selected_boxes_inside_class.begin(),
                    selected_boxes_inside_class.end(),
                    [&](auto selected_index) {
                        return this->suppress_by_iou(batch_box(batch_boxes, next_top_score.second),
                                                     batch_box(batch_boxes, selected_index.second),
                                                     iou_threshold);
                    });

                if(not not_selected)
                {
                    selected_boxes_inside_class.push_back(next_top_score);
                    selected_indices.push_back(bidx);
                    selected_indices.push_back(cidx);
                    selected_indices.push_back(next_top_score.second);
                }
                sorted_boxes.pop();
            }
        });

        result.visit([&](auto out) {
            std::copy(selected_indices.begin(), selected_indices.end(), out.begin());
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
