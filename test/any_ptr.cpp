#include <migraphx/any_ptr.hpp>
#include <test.hpp>

TEST_CASE(test_int_id)
{
    int i               = 1;
    migraphx::any_ptr p = &i;
    EXPECT(p.get<int*>() == &i);
    EXPECT(p.get(migraphx::get_type_name(i)) == &i);
    EXPECT(p.unsafe_get() == &i);
    EXPECT(test::throws([&] { p.get<float*>(); }));
    EXPECT(test::throws([&] { p.get(migraphx::get_type_name(&i)); }));
}

TEST_CASE(test_int_name)
{
    int i    = 1;
    void* vp = &i;
    migraphx::any_ptr p{vp, migraphx::get_type_name(i)};
    EXPECT(p.get<int*>() == &i);
    EXPECT(p.get(migraphx::get_type_name(i)) == &i);
    EXPECT(p.unsafe_get() == &i);
    EXPECT(test::throws([&] { p.get<float*>(); }));
    EXPECT(test::throws([&] { p.get(migraphx::get_type_name(&i)); }));
    EXPECT(test::throws([&] { p.get(migraphx::get_type_name(float{})); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
