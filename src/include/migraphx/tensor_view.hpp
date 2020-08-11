#ifndef MIGRAPHX_GUARD_TENSOR_VIEW_HPP
#define MIGRAPHX_GUARD_TENSOR_VIEW_HPP

#include <migraphx/shape.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/config.hpp>

#include <iostream>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T>
T as_number(T x)
{
    return x;
}
inline int32_t as_number(int8_t x) { return static_cast<int32_t>(x); }
inline uint32_t as_number(uint8_t x) { return static_cast<uint32_t>(x); }

inline int8_t as_number(bool x) { return static_cast<int8_t>(x); }

template <class T>
struct tensor_view
{
    using value_type = T;
    tensor_view() : m_data(nullptr) {}
    tensor_view(shape s, T* d) : m_data(d), m_shape(std::move(s)) {}

    const shape& get_shape() const { return this->m_shape; }

    bool empty() const { return m_data == nullptr || m_shape.lens().empty(); }

    std::size_t size() const { return m_shape.elements(); }

    T* data() { return this->m_data; }

    const T* data() const { return this->m_data; }

    template <class... Ts, MIGRAPHX_REQUIRES(std::is_integral<Ts>{}...)>
    const T& operator()(Ts... xs) const
    {
        assert(std::vector<std::size_t>{static_cast<std::size_t>(xs)...} < m_shape.lens());
        assert(m_shape.index({static_cast<std::size_t>(xs)...}) < m_shape.bytes() / sizeof(T));
        return m_data[m_shape.index({static_cast<std::size_t>(xs)...})];
    }

    template <class... Ts, MIGRAPHX_REQUIRES(std::is_integral<Ts>{}...)>
    T& operator()(Ts... xs)
    {
        assert(std::vector<std::size_t>{static_cast<std::size_t>(xs)...} < m_shape.lens());
        assert(m_shape.index({static_cast<std::size_t>(xs)...}) < m_shape.bytes() / sizeof(T));
        return m_data[m_shape.index({static_cast<std::size_t>(xs)...})];
    }

    template <class Iterator, MIGRAPHX_REQUIRES(not std::is_integral<Iterator>{})>
    const T& operator()(Iterator start, Iterator last) const
    {
        assert(std::distance(start, last) > 0);
        assert(std::all_of(start, last, [](auto x) { return x >= 0; }));
        return m_data[m_shape.index(start, last)];
    }

    template <class Iterator, MIGRAPHX_REQUIRES(not std::is_integral<Iterator>{})>
    T& operator()(Iterator start, Iterator last)
    {
        assert(std::distance(start, last) > 0);
        assert(std::all_of(start, last, [](auto x) { return x >= 0; }));
        return m_data[m_shape.index(start, last)];
    }

    T& operator[](std::size_t i)
    {
        assert(!this->empty() && i < this->size());
        return m_data[m_shape.index(i)];
    }

    const T& operator[](std::size_t i) const
    {
        assert(!this->empty() && i < this->size());
        return m_data[m_shape.index(i)];
    }

    T& front()
    {
        assert(!this->empty());
        return m_data[0];
    }

    const T& front() const
    {
        assert(!this->empty());
        return m_data[0];
    }

    T& back()
    {
        assert(!this->empty());
        return m_data[m_shape.index(this->size() - 1)];
    }

    const T& back() const
    {
        assert(!this->empty());
        return m_data[m_shape.index(this->size() - 1)];
    }

    // TODO: Add iterators so it can handle nonstandard tensors
    T* begin()
    {
        assert(this->m_shape.standard() or this->empty());
        return m_data;
    }

    T* end()
    {
        assert(this->m_shape.standard() or this->empty());
        if(this->empty())
            return m_data;
        else
            return m_data + this->size();
    }

    const T* begin() const
    {
        assert(this->m_shape.standard() or this->empty());
        return m_data;
    }

    const T* end() const
    {
        assert(this->m_shape.standard() or this->empty());
        if(this->empty())
            return m_data;
        else
            return m_data + this->size();
    }

    template <class U = T>
    std::vector<U> to_vector() const
    {
        return std::vector<U>(this->begin(), this->end());
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor_view<T>& x)
    {
        if(!x.empty())
        {
            os << as_number(x.front());
            for(std::size_t i = 1; i < x.m_shape.elements(); i++)
            {
                os << ", " << as_number(x.m_data[x.m_shape.index(i)]);
            }
        }
        return os;
    }

    private:
    T* m_data;
    shape m_shape;
};

template <>
struct tensor_view<bool>
{
    using value_type = int8_t;
    tensor_view() : m_data(nullptr) {}
    tensor_view(shape s, bool* d) : m_shape(std::move(s)) 
    {
        for (std::size_t i = 0; i < s.elements(); ++i)
        {
            m_data[i] = d[i];
        }
    }

    const shape& get_shape() const { return this->m_shape; }

    bool empty() const { return m_data == nullptr || m_shape.lens().empty(); }

    std::size_t size() const { return m_shape.elements(); }

    int8_t* data() { return this->m_data; }

    const int8_t* data() const { return this->m_data; }

    template <class... Ts, MIGRAPHX_REQUIRES(std::is_integral<Ts>{}...)>
    const int8_t& operator()(Ts... xs) const
    {
        assert(std::vector<std::size_t>{static_cast<std::size_t>(xs)...} < m_shape.lens());
        assert(m_shape.index({static_cast<std::size_t>(xs)...}) < m_shape.bytes() / sizeof(T));
        return m_data[m_shape.index({static_cast<std::size_t>(xs)...})];
    }

    template <class... Ts>
    int8_t& operator()(Ts... xs)
    {
        assert(std::vector<std::size_t>{static_cast<std::size_t>(xs)...} < m_shape.lens());
        assert(m_shape.index({static_cast<std::size_t>(xs)...}) < m_shape.bytes() / sizeof(T));
        return m_data[m_shape.index({static_cast<std::size_t>(xs)...})];
    }

    template <class Iterator, MIGRAPHX_REQUIRES(not std::is_integral<Iterator>{})>
    const int8_t& operator()(Iterator start, Iterator last) const
    {
        assert(std::distance(start, last) > 0);
        assert(std::all_of(start, last, [](auto x) { return x >= 0; }));
        return m_data[m_shape.index(start, last)];
    }

    template <class Iterator, MIGRAPHX_REQUIRES(not std::is_integral<Iterator>{})>
    int8_t& operator()(Iterator start, Iterator last)
    {
        assert(std::distance(start, last) > 0);
        assert(std::all_of(start, last, [](auto x) { return x >= 0; }));
        return m_data[m_shape.index(start, last)];
    }

    int8_t& operator[](std::size_t i)
    {
        assert(!this->empty() && i < this->size());
        return m_data[m_shape.index(i)];
    }

    const int8_t& operator[](std::size_t i) const
    {
        assert(!this->empty() && i < this->size());
        return m_data[m_shape.index(i)];
    }

    int8_t& front()
    {
        assert(!this->empty());
        return m_data[0];
    }

    const int8_t& front() const
    {
        assert(!this->empty());
        return m_data[0];
    }

    int8_t& back()
    {
        assert(!this->empty());
        return m_data[m_shape.index(this->size() - 1)];
    }

    const int8_t& back() const
    {
        assert(!this->empty());
        return m_data[m_shape.index(this->size() - 1)];
    }

    // TODO: Add iterators so it can handle nonstandard tensors
    int8_t* begin()
    {
        assert(this->m_shape.standard() or this->empty());
        return m_data;
    }

    int8_t* end()
    {
        assert(this->m_shape.standard() or this->empty());
        if(this->empty())
            return m_data;
        else
            return m_data + this->size();
    }

    const int8_t* begin() const
    {
        assert(this->m_shape.standard() or this->empty());
        return m_data;
    }

    const int8_t* end() const
    {
        assert(this->m_shape.standard() or this->empty());
        if(this->empty())
            return m_data;
        else
            return m_data + this->size();
    }

    std::vector<int8_t> to_vector() const
    {
        return std::vector<int8_t>(this->begin(), this->end());
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor_view<bool>& x)
    {
        if(!x.empty())
        {
            os << as_number(x.front());
            for(std::size_t i = 1; i < x.m_shape.elements(); i++)
            {
                os << ", " << as_number(x.m_data[x.m_shape.index(i)]);
            }
        }
        return os;
    }

    private:
    int8_t* m_data;
    shape m_shape;
};

template <class T, class U>
bool operator==(const tensor_view<T>& x, const tensor_view<U>& y)
{
    if(x.get_shape() == y.get_shape())
    {
        for(std::size_t i = 0; i < x.get_shape().elements(); i++)
        {
            if(!float_equal(x[i], y[i]))
                return false;
        }
        return true;
    }
    return false;
}

template <class T, class U>
bool operator!=(const tensor_view<T>& x, const tensor_view<U>& y)
{
    return !(x == y);
}

template <class T>
tensor_view<T> make_view(const shape& s, T* data)
{
    return {s, data};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
