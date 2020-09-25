#include <migraphx/convert_to_json.hpp>
#include <test.hpp>

TEST_CASE(key_num)
{
    std::string str = "{abc:{key:1}}";
    auto jstr = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":1}}");
}

TEST_CASE(key_null)
{
    std::string str = "{abc:{key:null}}";
    auto jstr = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":null}}");
}

TEST_CASE(key_nan)
{
    std::string str = "{abc:{key:nan}}";
    auto jstr = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":nan}}");
}

TEST_CASE(quote_key_num)
{
    std::string str = "{\"abc\":{\"key\":1}}";
    auto jstr = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\":{\"key\":1}}");
}

TEST_CASE(quote_with_space_key_num)
{
    std::string str = "{\"abc key\":{\"key\":1}}";
    auto jstr = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc key\":{\"key\":1}}");
}


TEST_CASE(key_value_num_space)
{
    std::string str = "{abc    :    {    key    :    1}}";
    auto jstr = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\"    :    {    \"key\"    :    1}}");
}

TEST_CASE(key_value_str)
{
    std::string str = "{abc : {key : value}}";
    auto jstr = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\" : {\"key\" : \"value\"}}");
}

TEST_CASE(key_space_value)
{
    std::string str = "{abc    : [key, value]}";
    auto jstr = migraphx::convert_to_json(str);
    EXPECT(jstr == "{\"abc\"    : [\"key\", \"value\"]}");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
