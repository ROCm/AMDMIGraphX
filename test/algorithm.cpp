#include <migraphx/algorithm.hpp>
#include <forward_list>
#include <list>
#include <functional>
#include <test.hpp>

#define FORWARD_CONTAINER_TEST_CASE(name, type)        \
    template <class Container>                         \
    void name();                                       \
    TEST_CASE_REGISTER(name<std::vector<type>>);       \
    TEST_CASE_REGISTER(name<std::list<type>>);         \
    TEST_CASE_REGISTER(name<std::forward_list<type>>); \
    template <class Container>                         \
    void name()

template <class Container, class Iterator>
auto erase_iterator(Container& c, Iterator pos, Iterator last) -> decltype(c.erase_after(pos, last))
{
    auto n  = std::distance(c.begin(), pos);
    auto it = n == 0 ? c.before_begin() : std::next(c.begin(), n - 1);
    return c.erase_after(it, last);
}

template <class Container, class Iterator>
auto erase_iterator(Container& c, Iterator pos, Iterator last) -> decltype(c.erase(pos, last))
{
    return c.erase(pos, last);
}

FORWARD_CONTAINER_TEST_CASE(adjacent_remove_if1, int)
{
    Container v = {0, 1, 1, 1, 4, 2, 2, 4, 2};
    erase_iterator(v, migraphx::adjacent_remove_if(v.begin(), v.end(), std::equal_to<>{}), v.end());
    EXPECT(v == Container{0, 1, 4, 2, 4, 2});
}

FORWARD_CONTAINER_TEST_CASE(adjacent_remove_if2, int)
{
    Container v = {0, 1, 1, 1, 4, 2, 2, 4, 2, 5, 5};
    erase_iterator(v, migraphx::adjacent_remove_if(v.begin(), v.end(), std::equal_to<>{}), v.end());
    EXPECT(v == Container{0, 1, 4, 2, 4, 2, 5});
}

FORWARD_CONTAINER_TEST_CASE(adjacent_remove_if3, int)
{
    Container v = {0, 1, 1, 1, 4, 2, 2, 4, 2, 5, 5, 6};
    erase_iterator(v, migraphx::adjacent_remove_if(v.begin(), v.end(), std::equal_to<>{}), v.end());
    EXPECT(v == Container{0, 1, 4, 2, 4, 2, 5, 6});
}

FORWARD_CONTAINER_TEST_CASE(adjacent_remove_if_non_equivalence, int)
{
    Container v = {0, 1, 1, 1, 4, 2, 2, 3, 4, 2, 5, 5, 6};
    auto pred   = [](int a, int b) { return (b - a) == 1; };
    erase_iterator(v, migraphx::adjacent_remove_if(v.begin(), v.end(), pred), v.end());
    EXPECT(v == Container{1, 1, 1, 4, 2, 4, 2, 5, 6});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
