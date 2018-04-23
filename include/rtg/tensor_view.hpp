#ifndef RTG_GUARD_TENSOR_VIEW_HPP
#define RTG_GUARD_TENSOR_VIEW_HPP

#include <rtg/shape.hpp>
#include <rtg/float_equal.hpp>

#include <iostream>

namespace rtg {

template <class T>
struct tensor_view
{
    tensor_view() : data_(nullptr) {}
    tensor_view(shape s, T* d) : data_(d), shape_(s) {}

    const shape& get_shape() const { return this->shape_; }

    bool empty() const { return data_ == nullptr || shape_.lens().empty(); }

    std::size_t size() const { return shape_.elements(); }

    T* data() { return this->data_; }

    const T* data() const { return this->data_; }

    template <class... Ts>
    const T& operator()(Ts... xs) const
    {
        return data_[shape_.index({xs...})];
    }

    template <class... Ts>
    T& operator()(Ts... xs)
    {
        return data_[shape_.index({xs...})];
    }

    T& operator[](std::size_t i)
    {
        assert(!this->empty() && i < this->size());
        return data_[shape_.index(i)];
    }

    const T& operator[](std::size_t i) const
    {
        assert(!this->empty() && i < this->size());
        return data_[shape_.index(i)];
    }

    T& front()
    {
        assert(!this->empty());
        return data_[0];
    }

    const T& front() const
    {
        assert(!this->empty());
        return data_[0];
    }

    T& back()
    {
        assert(!this->empty());
        return data_[shape_.index(this->size() - 1)];
    }

    const T& back() const
    {
        assert(!this->empty());
        return data_[shape_.index(this->size() - 1)];
    }

    // TODO: Add iterators so it can handle nonpacked tensors
    T* begin()
    {
        assert(this->shape_.packed());
        return data_;
    }

    T* end()
    {
        assert(this->shape_.packed());
        if(this->empty())
            return data_;
        else
            return data_ + this->size();
    }

    const T* begin() const
    {
        assert(this->shape_.packed());
        return data_;
    }

    const T* end() const
    {
        assert(this->shape_.packed());
        if(this->empty())
            return data_;
        else
            return data_ + this->size();
    }

    friend bool operator==(const tensor_view<T>& x, const tensor_view<T>& y)
    {
        if(x.shape_ == y.shape_)
        {
            for(std::size_t i = 0; i < x.shape_.elements(); i++)
            {
                if(!float_equal(x[i], y[i]))
                    return false;
            }
            return true;
        }
        return false;
    }

    friend bool operator!=(const tensor_view<T>& x, const tensor_view<T>& y) { return !(x == y); }

    friend std::ostream& operator<<(std::ostream& os, const tensor_view<T>& x)
    {
        if(!x.empty())
        {
            os << x.front();
            for(std::size_t i = 1; i < x.shape_.elements(); i++)
            {
                os << ", " << x.data_[x.shape_.index(i)];
            }
        }
        return os;
    }

    private:
    T* data_;
    shape shape_;
};

template <class T>
tensor_view<T> make_view(shape s, T* data)
{
    return {s, data};
}

} // namespace rtg

#endif
