/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_SHAPE_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_SHAPE_HPP

#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <class Lens, class Strides>
struct shape
{
    using shape_type  = shape;
    using index_array = typename Lens::base_array;
    Lens lens         = {};
    Strides strides   = {};

    constexpr shape() = default;

    constexpr shape(Lens l, Strides s) : lens(l), strides(s) {}

    constexpr auto elements() const { return _c<Lens{}.product()>; }

    constexpr auto element_space() const { return _c<Strides{}.dot(Lens{} - 1) + 1>; }

    constexpr auto packed() const { return not skips() and elements() == element_space(); }
    constexpr auto broadcasted() const { return _c<Strides{}.product() == 0>; }
    constexpr auto transposed() const
    {
        return return_c([] {
            auto lstrides = Strides{};
            if(shape{}.broadcasted())
            {
                index_array s{};
                auto out = copy_if(
                    lstrides.begin(), lstrides.end(), s.begin(), [](auto x) { return x != 0; });
                return not is_sorted(s.begin(), out, greater{});
            }
            else
            {
                return not is_sorted(lstrides.begin(), lstrides.end(), greater{});
            }
        });
    }
    constexpr auto skips() const
    {
        return return_c([] {
            auto lstrides = Strides{};
            return none_of(lstrides.begin(), lstrides.end(), [](auto x) { return x == 1; });
        });
    }

    constexpr auto standard() const { return packed() and not transposed(); }

    constexpr index_int index(index_array x) const { return x.dot(strides); }

    constexpr index_int index(std::initializer_list<index_int> x) const
    {
        index_int idx = 0;
        for(index_int i = 0; i < x.size(); i++)
            idx += *(x.begin() + i) * strides[i];
        return idx;
    }

    constexpr index_int index(index_int i) const
    {
        if(this->standard())
        {
            MIGRAPHX_ASSERT(i == compute_index(i));
            return i;
        }
        else
        {
            return compute_index(i);
        }
    }

    constexpr index_int compute_index(index_int i) const
    {
        const auto rank  = this->lens.size();
        index_int s      = 1;
        index_int result = 0;
        for(index_int j = 0; j < rank; j++)
        {
            const index_int k      = rank - j - 1;
            const index_int stride = this->strides[k];
            const index_int len    = this->lens[k];
            const index_int slen   = s * len;
            const index_int idx    = (i % slen) / s;
            result += stride * idx;
            s = slen;
        }
        return result;
    }

    /// Convert single index into a multi-index
    constexpr index_array multi(index_int idx) const
    {
        index_array result;
        index_int tidx = idx;
        for(diff_int is = result.size() - 1; is > 0; is--)
        {
            result[is] = tidx % lens[is];
            tidx       = tidx / lens[is];
        }
        result[0] = tidx;
        return result;
    }
    /// Convert multi-index into a single index
    constexpr index_int single(index_array idx) const
    {
        if(idx.empty())
            return 0;
        return inner_product(lens.begin() + 1, lens.end(), idx.begin(), idx.back());
    }

    constexpr shape get_shape() const { return *this; }

    template <class Stream>
    friend constexpr const Stream& operator<<(const Stream& ss, const shape& s)
    {
        ss << "{" << s.lens << "}, {" << s.strides << "}";
        return ss;
    }
};

template <class Lens, class Strides>
constexpr shape<Lens, Strides> make_shape(Lens lens, Strides strides)
{
    return {lens, strides};
}

} // namespace migraphx

#endif
