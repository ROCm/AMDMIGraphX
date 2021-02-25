#include <migraphx/serialize.hpp>
#include <migraphx/functional.hpp>
#include <test.hpp>

#include <numeric>

struct empty_type
{
};
struct reflectable_type
{
    enum simple_enum
    {
        simple1,
        simple2,
        simple3
    };
    enum class class_enum
    {
        class1,
        class2,
        class3
    };
    std::vector<std::size_t> ints = {};
    std::string name              = "";
    float fvalue                  = 0.0;
    empty_type et{};
    simple_enum se = simple1;
    class_enum ce  = class_enum::class1;

    struct nested_type
    {
        int value;
        template <class Self, class F>
        static auto reflect(Self& self, F f)
        {
            return migraphx::pack(f(self.value, "value"));
        }
    };
    std::vector<nested_type> nested_types = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.ints, "ints"),
                              f(self.name, "name"),
                              f(self.fvalue, "fvalue"),
                              f(self.et, "et"),
                              f(self.se, "se"),
                              f(self.ce, "ce"),
                              f(self.nested_types, "nested_types"));
    }
};

TEST_CASE(serialize_reflectable_type)
{
    reflectable_type t1{{1, 2},
                        "hello",
                        1.0,
                        {},
                        reflectable_type::simple1,
                        reflectable_type::class_enum::class2,
                        {{1}, {2}}};
    migraphx::value v1  = migraphx::to_value(t1);
    reflectable_type t2 = migraphx::from_value<reflectable_type>(v1);
    migraphx::value v2  = migraphx::to_value(t2);
    migraphx::value v3  = migraphx::to_value(reflectable_type{});

    EXPECT(v1 == v2);
    EXPECT(v1 != v3);
    EXPECT(v2 != v3);
}

TEST_CASE(serialize_empty_array)
{
    std::vector<std::size_t> ints = {};
    migraphx::value v             = migraphx::to_value(ints);
    EXPECT(v.is_array());
    EXPECT(v.empty());
    v.push_back(1);
    EXPECT(v.size() == 1);
    EXPECT(v.front().to<int>() == 1);
}

struct empty_struct
{
    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return migraphx::pack();
    }
};

TEST_CASE(serialize_empty_struct)
{
    empty_struct es{};
    migraphx::value v = migraphx::to_value(es);
    EXPECT(v.is_object());
    EXPECT(v.empty());
    v["a"] = 1;
    EXPECT(v.size() == 1);
    EXPECT(v.at("a").to<int>() == 1);
}

TEST_CASE(from_value_binary)
{
    std::vector<std::uint8_t> data(10);
    std::iota(data.begin(), data.end(), 0);

    migraphx::value v = migraphx::value::binary{data};

    auto out = migraphx::from_value<migraphx::value::binary>(v);
    EXPECT(out == data);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
