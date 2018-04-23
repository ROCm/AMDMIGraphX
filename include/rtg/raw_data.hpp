
#ifndef RTG_GUARD_RAW_DATA_HPP
#define RTG_GUARD_RAW_DATA_HPP

#include <rtg/tensor_view.hpp>

namespace rtg {

template <class Derived>
struct raw_data
{
    friend bool operator==(const Derived& x, const Derived& y)
    {
        auto&& xshape = x.get_shape();
        auto&& yshape = y.get_shape();
        bool result   = x.empty() && y.empty();
        if(not result && xshape == yshape)
        {
            auto&& xbuffer = x.data();
            auto&& ybuffer = y.data();
            // TODO: Dont use tensor view for single values
            xshape.visit_type([&](auto as) {
                auto xview = make_view(xshape, as.from(xbuffer));
                auto yview = make_view(yshape, as.from(ybuffer));
                result     = xview == yview;
            });
        }
        return result;
    }

    friend bool operator!=(const Derived& x, const Derived& y) { return !(x == y); }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const Derived& d)
    {
        d.visit([&](auto x) { os << x; });
        return os;
    }

    template <class Visitor>
    void visit_at(Visitor v, std::size_t n = 0) const
    {
        auto&& s      = static_cast<const Derived&>(*this).get_shape();
        auto&& buffer = static_cast<const Derived&>(*this).data();
        s.visit_type([&](auto as) { v(*(as.from(buffer) + s.index(n))); });
    }

    template <class Visitor>
    void visit(Visitor v) const
    {
        auto&& s      = static_cast<const Derived&>(*this).get_shape();
        auto&& buffer = static_cast<const Derived&>(*this).data();
        s.visit_type([&](auto as) { v(make_view(s, as.from(buffer))); });
    }

    bool single() const
    {
        auto&& s = static_cast<const Derived&>(*this).get_shape();
        return s.elements() == 1;
    }

    template <class T>
    T at(std::size_t n = 0) const
    {
        T result;
        this->visit_at([&](auto x) { result = x; }, n);
        return result;
    }
};

} // namespace rtg

#endif
