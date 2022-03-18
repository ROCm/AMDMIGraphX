#include <migraphx/migraphx.hpp>
#include "test.hpp"

struct array2 : migraphx::array_base<array2>
{
    std::vector<int> v;
    array2() = default;
    array2(std::initializer_list<int> x) : v(x)
    {}
    std::size_t size() const
    {
        return v.size();
    }
    int operator[](std::size_t i) const
    {
        return v[i];
    }
};

TEST_CASE(iterators)
{
    array2 a = {1, 2, 3};
    EXPECT(bool{std::equal(a.begin(), a.end(), a.v.begin())});
}

TEST_CASE(front_back)
{
    array2 a = {1, 2, 3};
    EXPECT(a.front() == 1);
    EXPECT(a.back() == 3);
}

TEST_CASE(empty)
{
    array2 a = {1, 2, 3};
    EXPECT(not a.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
