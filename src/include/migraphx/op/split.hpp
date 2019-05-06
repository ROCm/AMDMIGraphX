#ifndef MIGRAPHX_GUARD_OPERATORS_SPLIT_HPP
#define MIGRAPHX_GUARD_OPERATORS_SPLIT_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct split
{
    int axis = 0;
    std::vector<int> slice_dims;
    std::pair<int, int> slice_selector = {-1, -1};
    std::string name() const { return "split"; }

    std::vector<unsigned> compute_slice_elements(shape input_shape) const
    {
        unsigned accum = 1;
        int axis_id    = 0;

        //  compute accumulated elements on un-splitted axises.
        for(auto&& len : input_shape.lens())
        {
            if(axis_id != axis)
                accum *= len;
            axis_id++;
        }

        // compute number of elements for each slice.
        std::vector<unsigned> slice_elements;
        std::transform(slice_dims.begin(),
                       slice_dims.end(),
                       std::back_inserter(slice_elements),
                       [&](auto&& d) -> unsigned { return accum * d; });
        return slice_elements;
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // check_shapes{inputs, *this}.has(1);
        auto input_shape = inputs[0];
        std::vector<std::size_t> out_dims;

        if(slice_selector.first >= 0)
        {
            int first  = slice_selector.first;
            int second = slice_selector.second;
            if(second < first)
                MIGRAPHX_THROW("Illegal split selector");

            if(first == second)
            {
                int axis_id   = 0;
                int slice_dim = slice_dims[first];
                for(auto&& len : input_shape.lens())
                {
                    if(axis_id == axis)
                        out_dims.push_back(slice_dim);
                    else
                        out_dims.push_back(len);
                    axis_id++;
                }
            }
            else
            {
                std::vector<unsigned> slice_elements = compute_slice_elements(input_shape);
                int total_elements                   = 0;
                if(second >= slice_dims.size())
                    MIGRAPHX_THROW("Illegal split selector");

                for(int i = first; i <= second; i++)
                    total_elements += slice_elements[i];
                if(total_elements <= 0)
                    MIGRAPHX_THROW("Invalid number of elements");

                out_dims.push_back(total_elements);
            }
        }
        else
        {
            out_dims.push_back(input_shape.elements());
        }
        return {input_shape.type(), out_dims};
    }

    std::vector<int> compute_index_map(const shape& s) const
    {
        std::vector<int> index_map;
        int unit_slice = 1;
        int axis_id    = 0;
        if((axis < 0) || (axis >= s.lens().size()))
            MIGRAPHX_THROW("batch_contiguous: invalid split axis");

        for(auto&& len : s.lens())
        {
            if(axis_id++ > axis)
                unit_slice *= len;
        }
        int total_slice_dim = 0;
        for(auto&& dim : slice_dims)
        {
            if(dim == 0)
                MIGRAPHX_THROW("batch_contiguous: invalid split dimension");
            total_slice_dim += dim;
        }
        if(total_slice_dim != s.lens()[axis])
            MIGRAPHX_THROW("batch_contiguous: invalid split dimension");

        int stride            = unit_slice * total_slice_dim;
        std::size_t nelements = s.elements();
        std::vector<int> segment_size;
        std::vector<int> begin_index;
        int num_of_segments = nelements / stride;
        int index           = 0;

        // For each slice, compute segment size and begin index in the output.
        for(auto&& dim : slice_dims)
        {
            int size = unit_slice * dim;
            segment_size.push_back(size);
            begin_index.push_back(index);
            index += (num_of_segments * size);
        }

        // Map input index to output index.
        index_map.resize(nelements);
        for(std::size_t i = 0; i < nelements; i++)
        {
            std::size_t t_id          = i % stride;
            std::size_t slice_id      = 0;
            std::size_t element_index = 0;
            std::size_t segment_id    = i / stride;
            auto a_segment_size       = 0;
            std::size_t id            = 0;
            for(auto&& seg : segment_size)
            {
                if(t_id < a_segment_size + seg)
                {
                    slice_id      = id;
                    element_index = t_id - a_segment_size;
                    break;
                }
                a_segment_size += seg;
                id++;
            }
            int map_index =
                begin_index[slice_id] + segment_id * segment_size[slice_id] + element_index;
            index_map[map_index] = i;
        }
        return index_map;
    }

    unsigned compute_offset(shape input_shape) const
    {
        int first = slice_selector.first;
        if(first <= 0)
            return 0;
        unsigned offset                      = 0;
        std::vector<unsigned> slice_elements = compute_slice_elements(input_shape);
        int slice_ndx                        = 0;
        for(auto&& ele : slice_elements)
        {
            if(slice_ndx == first)
                break;
            offset += ele;
            slice_ndx++;
        }
        return offset;
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        auto arg0 = args[0];
        int first = slice_selector.first;
        if((axis == 0) && (first == -1))
            return {std::move(output_shape), std::move(arg0.data)};

        shape input_shape          = arg0.get_shape();
        std::vector<int> index_map = compute_index_map(input_shape);
        argument result{output_shape};
        std::size_t nelements = output_shape.elements();
        unsigned offset       = compute_offset(input_shape);
        visit_all(result, arg0)([&](auto output, auto input) {
            auto slice = make_view(output_shape, output.data());
            for(std::size_t i = 0; i < nelements; i++)
            {
                slice[i] = input[index_map[i + offset]];
            }
        });
        return result;
    }

    int output_alias(const std::vector<shape>& shapes) const { return shapes.size() - 1; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
