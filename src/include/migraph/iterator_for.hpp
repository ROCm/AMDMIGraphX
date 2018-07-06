#ifndef MIGRAPH_GUARD_RTGLIB_ITERATOR_FOR_HPP
#define MIGRAPH_GUARD_RTGLIB_ITERATOR_FOR_HPP

namespace migraph {

template <class T> 
    struct iterator_for_range {
    T* base;
    using base_iterator = decltype(base->begin());

    struct iterator {
        base_iterator i;
        base_iterator operator * () { return i; }
        base_iterator operator ++ () { return ++i; }
        bool operator != (const iterator& rhs) { return i != rhs.i; }
    };

    iterator begin() { return {base->begin()}; }
    iterator end() { return {base->end()}; }
};
template <class T>
iterator_for_range<T> iterator_for(T& x) 
{
    return {&x};
}

} // namespace migraph

#endif
