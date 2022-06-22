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
    box batch_box(const T& boxes, std::size_t box_ind, std::size_t box_idx) const
    {
        box result{};
        auto start = box_ind + 4 * box_idx;
        if(center_point_box)
        {
            float half_width  = boxes[start + 2] / 2.0;
            float half_height = boxes[start + 3] / 2.0;
            float x_center    = boxes[start + 0];
            float y_center    = boxes[start + 1];
            result.x          = {x_center - half_width, x_center + half_width};
            result.y          = {y_center - half_height, y_center + half_height};
        }
        else
        {
            result.x = {static_cast<float>(boxes[start + 1]), static_cast<float>(boxes[start + 3])};
            result.y = {static_cast<float>(boxes[start + 0]), static_cast<float>(boxes[start + 2])};
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

    // filter boxes below score_threshold
    template <class T>
    void filter_boxes_by_score(
        T scores,
        std::size_t score_offset_ind,
        std::size_t num_boxes,
        const float score_threshold,
        std::priority_queue<std::pair<float, int64_t>>& boxes_heap) const
    {
        auto insert_to_boxes_heap =
            make_function_output_iterator([&boxes_heap](const auto& x) { boxes_heap.push(x); });
        int64_t box_idx = 0;
        transform_if(
                scores.begin() + score_offset_ind,
                scores.begin() + score_offset_ind + num_boxes,
                insert_to_boxes_heap,
                [&](auto sc) {
                box_idx++;
                return sc >= score_threshold;
                },
                [&](auto sc) { return std::make_pair(sc, box_idx - 1); });
    }
    
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};

        std::size_t max_output_boxes_per_class =
            (args.size() > 2) ? (args.at(2).at<std::size_t>()) : 0;
        if(max_output_boxes_per_class == 0)
        {
            return result;
        }
        float iou_threshold   = (args.size() > 3) ? (args.at(3).at<float>()) : 0.0f;
        float score_threshold = (args.size() > 4) ? (args.at(4).at<float>()) : 0.0f;

        result.visit([&](auto output) {
            visit_all(args[0], args[1])([&](auto boxes, auto scores) {
                std::fill(output.begin(), output.end(), 0);
                const auto& lens       = scores.get_shape().lens();
                const auto num_batches = lens[0];
                const auto num_classes = lens[1];
                const auto num_boxes   = boxes.get_shape().lens()[1];
                // boxes of a class with NMS applied [score, index]
                std::vector<std::pair<float, int64_t>> selected_boxes_inside_class;
                std::vector<int64_t> selected_indices;
                selected_boxes_inside_class.reserve(output_shape.elements());
                // iterate over batches and classes
                shape comp_s{shape::float_type, {num_batches, num_classes}};
                shape_for_each(comp_s, [&](auto idx) {
                    auto batch_idx = idx[0];
                    auto class_idx = idx[1];
                    // index offset for this class
                    std::size_t score_offset_ind =
                        (batch_idx * num_classes + class_idx) * num_boxes;
                    // index to first value of this batch
                    std::size_t batch_boxes_ind = batch_idx * num_boxes * 4;
                    std::priority_queue<std::pair<float, int64_t>> boxes_heap;
                    filter_boxes_by_score(scores, score_offset_ind, num_boxes, score_threshold, boxes_heap);
                    selected_boxes_inside_class.clear();
                    // Get the next box with top score, filter by iou_threshold
                    while(!boxes_heap.empty() &&
                          selected_boxes_inside_class.size() < max_output_boxes_per_class)
                    {
                        // Check with existing selected boxes for this class, remove box if it
                        // exceeds the IOU (Intersection Over Union) threshold
                        const auto next_top_score = boxes_heap.top();
                        bool not_selected = std::any_of(
                                selected_boxes_inside_class.begin(),
                                selected_boxes_inside_class.end(),
                                [&](auto selected_index) {
                                return this->suppress_by_iou(
                                        batch_box(boxes, batch_boxes_ind, next_top_score.second),
                                        batch_box(boxes, batch_boxes_ind, selected_index.second),
                                        iou_threshold);
                                });

                        if(not not_selected)
                        {
                            selected_boxes_inside_class.push_back(next_top_score);
                            selected_indices.push_back(batch_idx);
                            selected_indices.push_back(class_idx);
                            selected_indices.push_back(next_top_score.second);
                        }
                        boxes_heap.pop();
                    }
                });
                std::copy(selected_indices.begin(), selected_indices.end(), output.begin());
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
