
#include <migraphx/transform_view.hpp>
#include <list>
#include <forward_list>
#include <vector>

#include <test.hpp>

TEST_CASE(basic_transform) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto view = migraphx::views::transform(vec, [](int x) { return x * x; });

    auto it = view.begin();
    EXPECT(*it == 1); ++it;
    EXPECT(*it == 4); ++it;
    EXPECT(*it == 9); ++it;
    EXPECT(*it == 16); ++it;
    EXPECT(*it == 25);
}

TEST_CASE(transform_with_reference) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto view = migraphx::views::transform(vec, [](int& x) -> int& { return x; });

    auto it = view.begin();
    EXPECT(*it == 1); ++it;
    EXPECT(*it == 2); ++it;
    EXPECT(*it == 3); ++it;
    EXPECT(*it == 4); ++it;
    EXPECT(*it == 5);

    // Modify the original vector through the view
    *view.begin() = 10;
    EXPECT(vec[0] == 10);
}

TEST_CASE(empty_range) {
    std::vector<int> vec;
    auto view = migraphx::views::transform(vec, [](int x) { return x * x; });

    EXPECT(bool{view.begin() == view.end()});
}

TEST_CASE(non_random_access_iterator) {
    std::list<int> lst = {1, 2, 3, 4, 5};
    auto view = migraphx::views::transform(lst, [](int x) { return x * 2; });

    auto it = view.begin();
    EXPECT(*it == 2); ++it;
    EXPECT(*it == 4); ++it;
    EXPECT(*it == 6); ++it;
    EXPECT(*it == 8); ++it;
    EXPECT(*it == 10);
}

TEST_CASE(non_random_access_iterator_with_reference) {
    std::list<int> lst = {1, 2, 3, 4, 5};
    auto view = migraphx::views::transform(lst, [](int& x) -> int& { return x; });

    auto it = view.begin();
    EXPECT(*it == 1); ++it;
    EXPECT(*it == 2); ++it;
    EXPECT(*it == 3); ++it;
    EXPECT(*it == 4); ++it;
    EXPECT(*it == 5);

    // Modify the original list through the view
    *view.begin() = 10;
    EXPECT(lst.front() == 10);
}

TEST_CASE(forward_iterator) {
    std::forward_list<int> flst = {1, 2, 3, 4, 5};
    auto view = migraphx::views::transform(flst, [](int x) { return x + 1; });

    auto it = view.begin();
    EXPECT(*it == 2); ++it;
    EXPECT(*it == 3); ++it;
    EXPECT(*it == 4); ++it;
    EXPECT(*it == 5); ++it;
    EXPECT(*it == 6);
    
    auto it2 = view.begin();
    std::advance(it2, 3);
    EXPECT(*it2 == 5);
}

TEST_CASE(forward_iterator_with_reference) {
    std::forward_list<int> flst = {1, 2, 3, 4, 5};
    auto view = migraphx::views::transform(flst, [](int& x) -> int& { return x; });

    auto it = view.begin();
    EXPECT(*it == 1); ++it;
    EXPECT(*it == 2); ++it;
    EXPECT(*it == 3); ++it;
    EXPECT(*it == 4); ++it;
    EXPECT(*it == 5);

    // Modify the original forward_list through the view
    *view.begin() = 10;
    EXPECT(flst.front() == 10);
}

TEST_CASE(transform_view_element_comparison) {
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {1, 2, 3, 4, 5};
    std::vector<int> vec3 = {5, 4, 3, 2, 1};

    auto squared = [](int x) { return x * x; };

    auto view1 = migraphx::views::transform(vec1, squared);
    auto view2 = migraphx::views::transform(vec2, squared);
    auto view3 = migraphx::views::transform(vec3, squared);

    EXPECT(bool{view1 == view2}); // Same elements
    EXPECT(bool{view1 != view3}); // Different elements
    EXPECT(bool{view1 < view3});  // Lexicographical comparison
    EXPECT(bool{view1 <= view3});
    EXPECT(bool{view3 > view1});
    EXPECT(bool{view3 >= view1});
}

struct non_comparable {
    int value;

    friend bool operator==(const non_comparable& lhs, const non_comparable& rhs) {
        return lhs.value == rhs.value;
    }

    friend bool operator!=(const non_comparable& lhs, const non_comparable& rhs) {
        return !(lhs == rhs);
    }
};

TEST_CASE(transform_view_non_comparable_elements) {

    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {2, 3, 4};
    auto as_non_comparable = [](int x) -> non_comparable { return {x}; };
    auto view = migraphx::views::transform(vec1, as_non_comparable);
    auto view2 = migraphx::views::transform(vec1, as_non_comparable);
    auto view3 = migraphx::views::transform(vec2, as_non_comparable);

    EXPECT(bool{view == view2});
    EXPECT(bool{view != view3});
}

TEST_CASE(operator_arrow_in_loop_reference) {
    struct T { int val; };
    std::vector<T> data{{1},{2},{3}};
    auto view = migraphx::views::transform(data, [](T& t)->T& { return t; });
    int sum = 0;
    for (auto it = view.begin(); it != view.end(); ++it) {
        sum += it->val;
    }
    EXPECT(sum == 6);
}

TEST_CASE(operator_arrow_in_loop_value) {
    struct T { int val; };
    std::vector<T> data{{1},{2},{3}};
    auto view = migraphx::views::transform(data, [](const T& t) { return T{t.val * 2}; });
    std::vector<int> out;
    for (auto it = view.begin(); it != view.end(); ++it) {
        out.push_back(it->val);
    }
    EXPECT(out.size() == 3);
    EXPECT(out[0] == 2);
    EXPECT(out[1] == 4);
    EXPECT(out[2] == 6);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
