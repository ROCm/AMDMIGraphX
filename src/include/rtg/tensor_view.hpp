#ifndef RTG_GUARD_TENSOR_VIEW_HPP
#define RTG_GUARD_TENSOR_VIEW_HPP

#include <rtg/shape.hpp>
#include <rtg/float_equal.hpp>

#include <iostream>

namespace rtg {

template <class T>
struct tensor_view
{
    tensor_view() : m_data(nullptr) {}
    tensor_view(shape s, T* d) : m_data(d), m_shape(s) {}

    const shape& get_shape() const { return this->m_shape; }

    bool empty() const { return m_data == nullptr || m_shape.lens().empty(); }

    std::size_t size() const { return m_shape.elements(); }

    T* data() { return this->m_data; }

    const T* data() const { return this->m_data; }

    template <class... Ts>
    const T& operator()(Ts... xs) const
    {
        return m_data[m_shape.index({xs...})];
    }

    template <class... Ts>
    T& operator()(Ts... xs)
    {
        return m_data[m_shape.index({static_cast<std::size_t>(xs)...})];
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

    // TODO: Add iterators so it can handle nonpacked tensors
    T* begin()
    {
        assert(this->m_shape.packed());
        return m_data;
    }

    T* end()
    {
        assert(this->m_shape.packed());
        if(this->empty())
            return m_data;
        else
            return m_data + this->size();
    }

    const T* begin() const
    {
        assert(this->m_shape.packed());
        return m_data;
    }

    const T* end() const
    {
        assert(this->m_shape.packed());
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

template<class T, class U>
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

template<class T, class U>
bool operator!=(const tensor_view<T>& x, const tensor_view<U>& y) { return !(x == y); }

template <class T>
tensor_view<T> make_view(shape s, T* data)
{
    return {s, data};
}

} // namespace rtg

#endif
