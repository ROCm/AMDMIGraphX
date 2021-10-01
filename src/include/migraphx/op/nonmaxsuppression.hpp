#ifndef MIGRAPHX_GUARD_OPERATORS_NONMAXSUPPRESSION_HPP
#define MIGRAPHX_GUARD_OPERATORS_NONMAXSUPPRESSION_HPP

#include <cmath>
#include <queue>
#include <cstdint>
#include <iterator>
#include <migraphx/config.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/shape_for_each.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct nonmaxsuppression
{
    int center_point_box = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.center_point_box, "center_point_box"));
    }

    std::string name() const { return "nonmaxsuppression"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto lens = inputs.front().lens();
        std::vector<int64_t> out_lens(2);
        out_lens.at(0) = lens.at(1);
        out_lens.at(1) = 3;
        return {shape::int64_type, out_lens};
    }

    struct box_info_ptr
    {
        float score{};
        int64_t index{};

        inline bool operator<(const box_info_ptr& rhs) const
        {
            return score < rhs.score || (float_equal(score, rhs.score) && index > rhs.index);
        }
    };

    void max_min(float lhs, float rhs, float& min, float& max) const
    {
        if(lhs >= rhs)
        {
            min = rhs;
            max = lhs;
        }
        else
        {
            min = lhs;
            max = rhs;
        }
    }

    inline bool suppress_by_iou(const float* boxes_data,
                                int64_t box_index1,
                                int64_t box_index2,
                                float iou_threshold) const
    {
        float x1_min{};
        float y1_min{};
        float x1_max{};
        float y1_max{};
        float x2_min{};
        float y2_min{};
        float x2_max{};
        float y2_max{};
        float intersection_x_min{};
        float intersection_x_max{};
        float intersection_y_min{};
        float intersection_y_max{};

        const float* box1 = boxes_data + 4 * box_index1;
        const float* box2 = boxes_data + 4 * box_index2;
        // center_point_box_ only support 0 or 1
        if(0 == center_point_box)
        {
            // boxes data format [y1, x1, y2, x2],
            max_min(box1[1], box1[3], x1_min, x1_max);
            max_min(box2[1], box2[3], x2_min, x2_max);

            intersection_x_min = std::max(x1_min, x2_min);
            intersection_x_max = std::min(x1_max, x2_max);
            if(intersection_x_max <= intersection_x_min)
                return false;

            max_min(box1[0], box1[2], y1_min, y1_max);
            max_min(box2[0], box2[2], y2_min, y2_max);
            intersection_y_min = std::max(y1_min, y2_min);
            intersection_y_max = std::min(y1_max, y2_max);
            if(intersection_y_max <= intersection_y_min)
                return false;
        }
        else
        {
            // 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
            float box1_width_half  = box1[2] / 2;
            float box1_height_half = box1[3] / 2;
            float box2_width_half  = box2[2] / 2;
            float box2_height_half = box2[3] / 2;

            x1_min = box1[0] - box1_width_half;
            x1_max = box1[0] + box1_width_half;
            x2_min = box2[0] - box2_width_half;
            x2_max = box2[0] + box2_width_half;

            intersection_x_min = std::max(x1_min, x2_min);
            intersection_x_max = std::min(x1_max, x2_max);
            if(intersection_x_max <= intersection_x_min)
                return false;

            y1_min = box1[1] - box1_height_half;
            y1_max = box1[1] + box1_height_half;
            y2_min = box2[1] - box2_height_half;
            y2_max = box2[1] + box2_height_half;

            intersection_y_min = std::max(y1_min, y2_min);
            intersection_y_max = std::min(y1_max, y2_max);
            if(intersection_y_max <= intersection_y_min)
                return false;
        }

        const float intersection_area =
            (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min);

        if(intersection_area <= .0f)
        {
            return false;
        }

        const float area1      = (x1_max - x1_min) * (y1_max - y1_min);
        const float area2      = (x2_max - x2_min) * (y2_max - y2_min);
        const float union_area = area1 + area2 - intersection_area;

        if(area1 <= .0f || area2 <= .0f || union_area <= .0f)
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

        int64_t max_output_boxes_per_class = 0;
        float iou_threshold                = 0.0f;
        float score_threshold              = 0.0f;

        if(args.size() > 2)
        {
            max_output_boxes_per_class = args.at(2).at<int64_t>();
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

        std::vector<box_info_ptr> selected_boxes_inside_class;
        std::vector<int64_t> selected_indices;
        selected_boxes_inside_class.reserve(output_shape.elements());

        auto scores = make_view<float>(args.at(1).get_shape(), args.at(1).cast<float>());
        shape comp_s{shape::float_type, {batch_num, class_num}};
        shape_for_each(comp_s, [&](auto idx) {
            auto bidx = idx[0];
            auto cidx = idx[1];

            std::size_t score_offset = (bidx * class_num + cidx) * box_num;
            const float* batch_boxes = args.at(0).cast<float>() + bidx * box_num * 4;
            std::vector<box_info_ptr> cand_boxes;
            cand_boxes.reserve(box_num);

            int64_t box_idx = 0;
            transform_if(scores.begin() + score_offset,
                         scores.begin() + score_offset + box_num,
                         std::back_inserter(cand_boxes),
                         [&](auto sc) {
                             box_idx++;
                             return sc >= score_threshold;
                         },
                         [&](auto sc) {
                             return box_info_ptr{sc, box_idx - 1};
                         });
            std::priority_queue<box_info_ptr, std::vector<box_info_ptr>> sorted_boxes(
                std::less<box_info_ptr>(), std::move(cand_boxes));

            selected_boxes_inside_class.clear();
            // Get the next box with top score, filter by iou_threshold
            while(!sorted_boxes.empty() &&
                  static_cast<int64_t>(selected_boxes_inside_class.size()) <
                      max_output_boxes_per_class)
            {
                const box_info_ptr& next_top_score = sorted_boxes.top();

                // Check with existing selected boxes for this class, suppress if exceed the IOU
                // (Intersection Over Union) threshold
                bool not_selected = std::any_of(
                    selected_boxes_inside_class.begin(),
                    selected_boxes_inside_class.end(),
                    [&](auto selected_index) {
                        return this->suppress_by_iou(
                            batch_boxes, next_top_score.index, selected_index.index, iou_threshold);
                    });

                if(not not_selected)
                {
                    selected_boxes_inside_class.push_back(next_top_score);
                    selected_indices.push_back(bidx);
                    selected_indices.push_back(cidx);
                    selected_indices.push_back(next_top_score.index);
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
