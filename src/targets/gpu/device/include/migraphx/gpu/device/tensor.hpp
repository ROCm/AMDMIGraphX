#ifndef MIGRAPHX_GUARD_RTGLIB_DEAVICE_TENSOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEAVICE_TENSOR_HPP

#include <hip/hip_runtime.h>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/tensor_view.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

#define MIGRAPHX_DEVICE_CONSTEXPR constexpr __device__ __host__

template <class F>
void visit_tensor_size(std::size_t n, F f)
{
    switch(n)
    {
    case 1:
    {
        f(std::integral_constant<std::size_t, 1>{});
        break;
    }
    case 2:
    {
        f(std::integral_constant<std::size_t, 2>{});
        break;
    }
    case 3:
    {
        f(std::integral_constant<std::size_t, 3>{});
        break;
    }
    case 4:
    {
        f(std::integral_constant<std::size_t, 4>{});
        break;
    }
    case 5:
    {
        f(std::integral_constant<std::size_t, 5>{});
        break;
    }
    default: throw std::runtime_error("Unknown tensor size");
    }
}

template <class T, std::size_t N>
struct hip_array
{
    T d[N];
    MIGRAPHX_DEVICE_CONSTEXPR T& operator[](std::size_t i) { return d[i]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& operator[](std::size_t i) const { return d[i]; }

    MIGRAPHX_DEVICE_CONSTEXPR T* data() { return d; }
    MIGRAPHX_DEVICE_CONSTEXPR const T* data() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR std::integral_constant<std::size_t, N> size() const { return {}; }

    MIGRAPHX_DEVICE_CONSTEXPR T* begin() { return d; }
    MIGRAPHX_DEVICE_CONSTEXPR const T* begin() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR T* end() { return d + size(); }
    MIGRAPHX_DEVICE_CONSTEXPR const T* end() const { return d + size(); }

    MIGRAPHX_DEVICE_CONSTEXPR T dot(const hip_array& x) const
    {
        T result = 0;
        for(std::size_t i = 0; i < N; i++)
            result += x[i] * d[i];
        return result;
    }

    MIGRAPHX_DEVICE_CONSTEXPR T product() const
    {
        T result = 1;
        for(std::size_t i = 0; i < N; i++)
            result *= d[i];
        return result;
    }

    friend MIGRAPHX_DEVICE_CONSTEXPR hip_array operator*(const hip_array& x, const hip_array& y)
    {
        hip_array result;
        for(std::size_t i = 0; i < N; i++)
            result[i] = x[i] * y[i];
        return result;
    }
};

template <class T, std::size_t N>
struct hip_vector
{
    MIGRAPHX_DEVICE_CONSTEXPR hip_vector() = default;
    MIGRAPHX_DEVICE_CONSTEXPR hip_vector(std::size_t s) : len(s) {}
    template <class Iterator>
    __device__ __host__ hip_vector(Iterator start, Iterator last)
    {
        auto it = std::copy(start, last, d);
        len     = std::distance(d, it);
    }

    __device__ __host__ hip_vector(std::initializer_list<T> x)
    {
        auto it = std::copy(x.begin(), x.end(), d);
        len     = x.size();
    }

    MIGRAPHX_DEVICE_CONSTEXPR T& operator[](std::size_t i) { return d[i]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& operator[](std::size_t i) const { return d[i]; }

    MIGRAPHX_DEVICE_CONSTEXPR T& front() { return d[0]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& front() const { return d[0]; }

    MIGRAPHX_DEVICE_CONSTEXPR T& back() { return d[size() - 1]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& back() const { return d[size() - 1]; }

    MIGRAPHX_DEVICE_CONSTEXPR T* data() { return d; }
    MIGRAPHX_DEVICE_CONSTEXPR const T* data() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t size() const { return len; }

    MIGRAPHX_DEVICE_CONSTEXPR T* begin() { return d; }
    MIGRAPHX_DEVICE_CONSTEXPR const T* begin() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR T* end() { return d + size(); }
    MIGRAPHX_DEVICE_CONSTEXPR const T* end() const { return d + size(); }

    template <class U>
    MIGRAPHX_DEVICE_CONSTEXPR void push_back(U&& x)
    {
        d[len] = static_cast<U&&>(x);
        len++;
    }

    private:
    T d[N]          = {};
    std::size_t len = 0;
};

template <std::size_t N, class T>
hip_vector<T, N> to_hip_vector(const std::vector<T>& x)
{
    hip_vector<T, N> result(x.size());
    std::copy(x.begin(), x.end(), result.begin());
    return result;
}

template <std::size_t N>
struct hip_shape
{
    using hip_index                   = hip_array<std::size_t, N>;
    hip_array<std::size_t, N> lens    = {};
    hip_array<std::size_t, N> strides = {};
    bool standard                     = false;

    __device__ __host__ hip_shape() = default;

    hip_shape(const shape& s) : standard(s.standard())
    {
        assert(s.lens().size() == N);
        assert(s.strides().size() == N);
        std::copy(s.lens().begin(), s.lens().end(), lens.begin());
        std::copy(s.strides().begin(), s.strides().end(), strides.begin());
    }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t elements() const { return lens.product(); }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t index(hip_index x) const { return x.dot(strides); }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t index(std::initializer_list<std::size_t> x) const
    {
        std::size_t idx = 0;
        for(std::size_t i = 0; i < x.size(); i++)
            idx += *(x.begin() + i) * strides[i];
        return idx;
    }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t index(std::size_t i) const
    {
        if(this->standard)
            return i;
        else
        {
            const std::size_t rank = this->lens.size();
            std::size_t s          = 1;
            std::size_t result     = 0;
            for(std::size_t j = 0; j < this->lens.size(); j++)
            {
                const std::size_t k      = rank - j - 1;
                const std::size_t stride = this->strides[k];
                const std::size_t len    = this->lens[k];
                const std::size_t slen   = s * len;
                const std::size_t idx    = (i % slen) / s;
                result += stride * idx;
                s = slen;
            }
            return result;
        }
    }

    MIGRAPHX_DEVICE_CONSTEXPR hip_index multi(std::size_t idx) const
    {
        hip_index result;
        std::size_t tidx = idx;
        for(std::size_t is = 0; is < result.size(); is++)
        {
            result[is] = tidx / strides[is];
            tidx       = tidx % strides[is];
        }
        return result;
    }
};
template <class T, std::size_t N>
struct hip_tensor_view
{
    using value_type                      = T;
    __device__ __host__ hip_tensor_view() = default;
    __host__ hip_tensor_view(tensor_view<T> x) : d(x.data()), s(x.get_shape()) {}
    __host__ hip_tensor_view(T* x, const shape& ss) : d(x), s(ss) {}

    MIGRAPHX_DEVICE_CONSTEXPR const hip_shape<N>& get_shape() const { return s; }

    MIGRAPHX_DEVICE_CONSTEXPR std::size_t size() const { return s.elements(); }

    MIGRAPHX_DEVICE_CONSTEXPR value_type* data() const { return d; }

    template <class U>
    MIGRAPHX_DEVICE_CONSTEXPR value_type& operator[](U i) const
    {
        return d[s.index(i)];
    }

    MIGRAPHX_DEVICE_CONSTEXPR value_type* begin() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR value_type* end() const { return d + size(); }

    private:
    value_type* d = nullptr;
    hip_shape<N> s{};
};

template <std::size_t N, class T>
hip_tensor_view<T, N> make_hip_tensor_view(tensor_view<T> x)
{
    return x;
}

template <std::size_t N, std::size_t M, class T>
hip_vector<hip_tensor_view<T, N>, M> make_hip_tensor_views(const std::vector<tensor_view<T>>& x)
{
    hip_vector<hip_tensor_view<T, N>, M> result(x.size());
    std::transform(
        x.begin(), x.end(), result.begin(), [&](auto y) { return make_hip_tensor_view<N>(y); });
    return result;
}

template <class T, class... Ts>
auto hip_visit_all(T&& x, Ts&&... xs)
{
    return [&](auto f) {
        visit_tensor_size(x.get_shape().lens().size(), [&](auto dim) {
            visit_all(x,
                      xs...)([&](auto... vs) { f(make_hip_tensor_view<dim>(device_cast(vs))...); });
        });
    };
}

template <std::size_t N, class T, class... Ts>
auto hip_vec_visit_all(T&& x, Ts&&... xs)
{
    return [&](auto f) {
        visit_tensor_size(x.get_shape().lens().size(), [&](auto dim) {
            visit_all(x, xs...)(
                [&](auto... vs) { f(make_hip_tensor_view<dim>(as_vec<N>(device_cast(vs)))...); });
        });
    };
}

template <class T, class... Ts>
auto hip_pointer_visit_all(T&& x, Ts&&... xs)
{
    return [&](auto f) { visit_all(x, xs...)([&](auto... vs) { f(device_cast(vs.data())...); }); };
}

template <std::size_t N, class T>
auto hip_visit_all(const std::vector<T>& x)
{
    return [&](auto f) {
        visit_tensor_size(x.front().get_shape().lens().size(), [&](auto dim) {
            visit_all(x)([&](auto&& v) { f(make_hip_tensor_views<dim, N>(v)); });
        });
    };
}

template <std::size_t NDim>
using hip_tensor_index = hip_array<std::size_t, NDim>;

template <std::size_t NDim>
struct hip_tensor_descriptor
{
    __device__ __host__ hip_tensor_descriptor() = default;

    hip_tensor_descriptor(const shape& s)
    {
        std::copy(s.lens().begin(), s.lens().end(), lens);
        std::copy(s.strides().begin(), s.strides().end(), strides);
    }

    __device__ __host__ hip_tensor_index<NDim> multi(std::size_t idx) const
    {
        hip_tensor_index<NDim> result{};
        std::size_t tidx = idx;
        for(std::size_t is = 0; is < NDim; is++)
        {
            result[is] = tidx / strides[is];
            tidx       = tidx % strides[is];
        }
        return result;
    }
    __device__ __host__ std::size_t linear(hip_tensor_index<NDim> s) const
    {
        std::size_t idx = 0;
        for(std::size_t i = 0; i < NDim; i++)
            idx += s[i] * strides[i];
        return idx;
    }
    std::size_t lens[NDim]    = {};
    std::size_t strides[NDim] = {};
};

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
