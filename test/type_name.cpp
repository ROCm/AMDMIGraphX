
#include <migraphx/type_name.hpp>
#include "test.hpp"

struct global_class
{
    struct inner_class
    {
    };
};

namespace foo {

struct ns_class
{
    struct inner_class
    {
    };
};
} // namespace foo

int main()
{
    EXPECT(migraphx::get_type_name<global_class>() == "global_class");
    EXPECT(migraphx::get_type_name<global_class::inner_class>() == "global_class::inner_class");
    EXPECT(migraphx::get_type_name<foo::ns_class>() == "foo::ns_class");
    EXPECT(migraphx::get_type_name<foo::ns_class::inner_class>() == "foo::ns_class::inner_class");
}
