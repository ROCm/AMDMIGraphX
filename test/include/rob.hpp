#ifndef MIGRAPH_GUARD_ROB_HPP
#define MIGRAPH_GUARD_ROB_HPP

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

// Used to access private member variables
template <class Tag>
struct stowed
{
     static typename Tag::type value;
}; 
template <class Tag> 
typename Tag::type stowed<Tag>::value;

template <class Tag, typename Tag::type x>
struct stow_private
{
     stow_private() { stowed<Tag>::value = x; }
     static stow_private instance;
};
template <class Tag, typename Tag::type x> 
stow_private<Tag,x> stow_private<Tag,x>::instance;

template<class C, class T>
struct mem_data_ptr { typedef T(C::*type); };

#define MIGRAPH_ROB(name, Type, C, mem) \
struct name ## _tag \
: mem_data_ptr<C, Type> \
{}; \
template struct stow_private<name ## _tag,&C::mem>; \
template<class T> \
auto& name(T&& x) \
{ \
    return x.*stowed<name ## _tag>::value; \
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
