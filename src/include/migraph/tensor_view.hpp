#ifndef MIGRAPH_GUARD_TENSOR_VIEW_HPP
#define MIGRAPH_GUARD_TENSOR_VIEW_HPP

#include <migraph/shape.hpp>
#include <migraph/float_equal.hpp>
#include <migraph/requires.hpp>

#include <iostream>

namespace migraph {

template <class T>
struct tensor_view
{
    using value_type = T;
    tensor_view() : m_data(nullptr) {}
    tensor_view(shape s, T* d) : m_data(d), m_shape(s) {}

    const shape& get_shape() const { return this->m_shape; }

    bool empty() const { return m_data == nullptr || m_shape.lens().empty(); }

    std::size_t size() const { return m_shape.elements(); }

    T* data() { return this->m_data; }

    const T* data() const { return this->m_data; }

    template <class... Ts, MIGRAPH_REQUIRES(std::is_integral<Ts>{}...)>
    const T& operator()(Ts... xs) const
    {
        assert(m_shape.index({static_cast<std::size_t>(xs)...}) < m_shape.bytes() / sizeof(T));
        return m_data[m_shape.index({static_cast<std::size_t>(xs)...})];
    }

    template <class... Ts, MIGRAPH_REQUIRES(std::is_integral<Ts>{}...)>
    T& operator()(Ts... xs)
    {
        assert(m_shape.index({static_cast<std::size_t>(xs)...}) < m_shape.bytes() / sizeof(T));
        return m_data[m_shape.index({static_cast<std::size_t>(xs)...})];
    }

    template <class Iterator, MIGRAPH_REQUIRES(not std::is_integral<Iterator>{})>
    const T& operator()(Iterator start, Iterator last) const
    {
        return m_data[m_shape.index(start, last)];
    }

    template <class Iterator, MIGRAPH_REQUIRES(not std::is_integral<Iterator>{})>
    T& operator()(Iterator start, Iterator last)
    {
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
        assert(this->m_shape.standard());
        return m_data;
    }

    T* end()
    {
        assert(this->m_shape.standard());
        if(this->empty())
            return m_data;
        else
            return m_data + this->size();
    }

    const T* begin() const
    {
        assert(this->m_shape.standard());
        return m_data;
    }

    const T* end() const
    {
        assert(this->m_shape.standard());
        if(this->empty())
            return m_data;
        else
            return m_data + this->size();
    }

    friend std::ostream& operator<<(std::ostream& os, const tensor_view<T>& x)
    {
        if(!x.empty())
        {
            os << x.front();
            for(std::size_t i = 1; i < x.m_shape.elements(); i++)
            {
                os << ", " << x.m_data[x.m_shape.index(i)];
            }
        }
        return os;
    }

    private:
    T* m_data;
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
tensor_view<T> make_view(shape s, T* data)
{
    return {s, data};
}

} // namespace migraph

#endif
