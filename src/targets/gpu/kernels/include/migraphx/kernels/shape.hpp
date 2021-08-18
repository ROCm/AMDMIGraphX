#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_SHAPE_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_SHAPE_HPP

#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <class Lens, class Strides>
struct shape
{
    using index_array = typename Lens::base_array;
    Lens lens         = {};
    Strides strides   = {};

    constexpr shape() = default;

    constexpr shape(Lens l, Strides s) : lens(l), strides(s) {}

    constexpr index_int elements() const { return lens.product(); }

    constexpr index_int element_space() const { return strides.dot(lens - 1) + 1; }

    constexpr bool packed() const { return elements() == element_space(); }
    constexpr bool broadcasted() const { return strides.product() == 0; }
    constexpr bool transposed() const
    {
        if(broadcasted())
        {
            index_array s;
            index_int j = 0;
            for(index_int i = 0; i < s.size(); i++)
            {
                if(strides[i] != 0)
                {
                    s[j] = strides[i];
                    j++;
                }
            }
            return not is_sorted(s.begin(), s.begin() + j, greater{});
        }
        else
        {
            return not is_sorted(strides.begin(), strides.end(), greater{});
        }
    }

    constexpr bool standard() const { return packed() and not transposed(); }

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
            return i;
        else
        {
            const index_int rank = this->lens.size();
            index_int s          = 1;
            index_int result     = 0;
            for(index_int j = 0; j < this->lens.size(); j++)
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
    }

    constexpr index_array multi(index_int idx) const
    {
        index_array result;
        index_int tidx = idx;
        for(std::ptrdiff_t is = result.size() - 1; is > 0; is--)
        {
            result[is] = tidx % lens[is];
            tidx       = tidx / lens[is];
        }
        result[0] = tidx;
        return result;
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
