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
#ifndef MIGRAPHX_GUARD_KERNELS_GATHER_VIEW_HPP
#define MIGRAPHX_GUARD_KERNELS_GATHER_VIEW_HPP

#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/functional.hpp>

namespace migraphx {

/// The shape of the data shape gathered by the indices shape along Axis: the
/// axis counts the indices and the strides of the data are kept
template <index_int Axis, class DataShape, class IndicesShape>
constexpr auto gather_view_shape(DataShape, IndicesShape)
{
    constexpr auto lens = return_array_c([] {
        auto result  = DataShape{}.lens.base();
        result[Axis] = IndicesShape{}.elements();
        return result;
    });
    return make_shape(lens, DataShape{}.strides);
}

/// The shape of the data behind a gathered view: the gather axis has the
/// data length again and the strides are kept
template <index_int Axis, class Shape, class Data>
constexpr auto ungather_shape(Shape, Data)
{
    constexpr auto lens = return_array_c([] {
        auto result  = Shape{}.lens.base();
        result[Axis] = get_shape_c<Data>{}.lens[Axis];
        return result;
    });
    return make_shape(lens, Shape{}.strides);
}

template <index_int Axis, class Data, class Indices>
struct gather_view;

template <index_int Axis, class Data, class Indices>
constexpr gather_view<Axis, Data, Indices> make_gather_view(Data data, Indices indices);

/// A view of the data gathered by the indices along Axis: an element reads
/// the data at indices[i[Axis]] along the axis. The shape is the gathered
/// shape with the strides of the data, so slicing and vectorizing it works
/// like a plain view and taking the address of an element gives a plain view
/// into the data, which resolves the index once per slice.
template <index_int Axis, class Data, class Indices>
struct gather_view
{
    using type = typename Data::type;
    using shape_type =
        decltype(gather_view_shape<Axis>(get_shape_c<Data>{}, get_shape_c<Indices>{}));
    using memory_tag  = typename Data::memory_tag;
    using index_array = typename shape_type::index_array;

    Data base;
    Indices indices;

    constexpr shape_type get_shape() const { return {}; }
    constexpr auto size() const { return get_shape().elements(); }

    /// The offset into the data of a multi-index of the gathered shape. A
    /// negative index counts from the end, and the index is clamped so the
    /// kernel never reads outside the data, which also keeps the benchmark
    /// runs with arbitrary indices in bounds.
    constexpr index_int offset(index_array i) const
    {
        constexpr index_int len = get_shape_c<Data>{}.lens[Axis];
        auto g                  = indices[i[Axis]];
        if(g < 0)
            g += len;
        i[Axis] = g < 0 ? 0 : (g < len ? index_int(g) : len - 1);
        return base.get_shape().index(i);
    }

    constexpr index_int offset(index_int i) const { return offset(get_shape().multi(i)); }

    template <class I>
    constexpr type& operator[](I i) const
    {
        return base.data()[offset(i)];
    }

    constexpr type* data() const { return base.data(); }

    template <class U>
    constexpr auto with(U* y) const
    {
        return make_gather_view<Axis>(base.with(y), indices);
    }

    template <class U, class Shape2>
    constexpr auto with(U* y, Shape2 s) const
    {
        return make_gather_view<Axis>(base.with(y, ungather_shape<Axis>(s, base)), indices);
    }
};

template <index_int Axis, class Data, class Indices>
constexpr gather_view<Axis, Data, Indices> make_gather_view(Data data, Indices indices)
{
    return {data, indices};
}

template <index_int Axis, class Data, class Indices>
constexpr auto as_const(gather_view<Axis, Data, Indices> x)
{
    return make_gather_view<Axis>(as_const(x.base), x.indices);
}

/// The slice of a gathered view at the multi-index i: when the gather axis is
/// not sliced the index is resolved once and the slice is a plain view,
/// otherwise the axis is reduced and the slice stays a gathered view
/// resolving the index per element
template <index_int Axis, class Data, class Indices, class T, class Shape>
constexpr auto make_slice_view(gather_view<Axis, Data, Indices> input, T i, Shape s)
{
    if constexpr(Shape{}.lens[Axis] == 1)
    {
        return make_tensor_view(&input[i], s);
    }
    else
    {
        i[Axis] = 0;
        auto* p = input.base.data() + input.base.get_shape().index(i);
        return make_gather_view<Axis>(make_tensor_view(p, ungather_shape<Axis>(s, input.base)),
                                      input.indices);
    }
}

/// Replace the data argument with the view of it gathered by the indices
/// argument along the axis. The indices argument stays in place unused so
/// the arguments keep their positions.
template <index_int Axis, index_int DataIndex, index_int IndicesIndex>
constexpr auto gather_arg()
{
    return make_transform([](auto f, auto... xs) {
        return sequence_c<sizeof...(xs)>([&](auto... is) {
            auto select = [&](auto i) {
                if constexpr(decltype(i){} == _c<DataIndex>)
                    return make_gather_view<Axis>(arg_c<DataIndex>()(xs...),
                                                  arg_c<IndicesIndex>()(xs...));
                else
                    return arg(i)(xs...);
            };
            return f(select(is)...);
        });
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_GATHER_VIEW_HPP
