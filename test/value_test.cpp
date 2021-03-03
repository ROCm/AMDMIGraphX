#include <migraphx/value.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>

enum class enum_type
{
    a,
    b,
    c
};

TEST_CASE(value_default_construct)
{
    migraphx::value v;
    EXPECT(v.is_null());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_null)
{
    migraphx::value v = nullptr;
    EXPECT(v.is_null());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_assign_null)
{
    migraphx::value v;
    v = nullptr;
    EXPECT(v.is_null());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_int1)
{
    EXPECT(migraphx::value(1).is_int64());
    migraphx::value v(1);
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_int2)
{
    migraphx::value v = 1;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_string)
{
    migraphx::value v = "one";
    EXPECT(v.is_string());
    EXPECT(v.get_string() == "one");
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_float)
{
    migraphx::value v = 1.0;
    EXPECT(v.is_float());
    EXPECT(migraphx::float_equal(v.get_float(), 1.0));
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_bool)
{
    migraphx::value v = true;
    EXPECT(v.is_bool());
    EXPECT(v.get_bool() == true);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_enum1)
{
    migraphx::value v = enum_type::a;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == static_cast<std::uint64_t>(enum_type::a));
    EXPECT(bool{v.to<enum_type>() == enum_type::a});
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_enum2)
{
    migraphx::value v = enum_type::b;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == static_cast<std::uint64_t>(enum_type::b));
    EXPECT(bool{v.to<enum_type>() == enum_type::b});
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_enum3)
{
    migraphx::value v = enum_type::c;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == static_cast<std::uint64_t>(enum_type::c));
    EXPECT(bool{v.to<enum_type>() == enum_type::c});
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_empty_object)
{
    migraphx::value v = migraphx::value::object{};
    EXPECT(v.is_object());
    EXPECT(v.get_object().empty());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_empty_array)
{
    migraphx::value v = migraphx::value::array{};
    EXPECT(v.is_array());
    EXPECT(v.get_array().empty());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_assign_int)
{
    migraphx::value v;
    v = 0;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 0);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_copy_construct)
{
    migraphx::value v1(1);
    migraphx::value v2 = v1; // NOLINT
    EXPECT(v1 == v2);
}

TEST_CASE(value_copy_assign)
{
    migraphx::value v1(1);
    migraphx::value v2;
    v2 = v1;
    EXPECT(v1 == v2);
}

TEST_CASE(value_reassign)
{
    migraphx::value v1(1);
    migraphx::value v2 = v1;
    v1                 = 2;
    EXPECT(v1 != v2);
}

TEST_CASE(value_copy_assign_key)
{
    migraphx::value v1("key", 1);
    migraphx::value v2;
    v2 = v1;
    EXPECT(v2.get_key() == "key");
    EXPECT(v1 == v2);
}

TEST_CASE(value_copy_assign_keyless)
{
    migraphx::value v1(1);
    migraphx::value v2("key", nullptr);
    v2 = v1;
    EXPECT(v2.get_key() == "key");
    EXPECT(v1 != v2);
    EXPECT(v1.without_key() == v2.without_key());
}

TEST_CASE(value_construct_array)
{
    migraphx::value v = {1, 2, 3};
    EXPECT(v.is_array());
    EXPECT(v.get_array().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front() == migraphx::value(1));
    EXPECT(v[1] == migraphx::value(2));
    EXPECT(v.at(1) == migraphx::value(2));
    EXPECT(v.back() == migraphx::value(3));
    EXPECT(test::throws([&] { v.at("???"); }));
    [=] {
        EXPECT(v.data() != nullptr);
        EXPECT(v.front().is_int64());
        EXPECT(v.front() == migraphx::value(1));
        EXPECT(v[1] == migraphx::value(2));
        EXPECT(v.at(1) == migraphx::value(2));
        EXPECT(v.back() == migraphx::value(3));
    }();
}

TEST_CASE(value_insert_array)
{
    migraphx::value v;
    v.insert(v.end(), 1);
    v.insert(v.end(), 2);
    v.insert(v.end(), 3);
    EXPECT(v.is_array());
    EXPECT(v.get_array().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front() == migraphx::value(1));
    EXPECT(v[1] == migraphx::value(2));
    EXPECT(v.at(1) == migraphx::value(2));
    EXPECT(v.back() == migraphx::value(3));
}

TEST_CASE(value_key_array)
{
    std::vector<migraphx::value> values = {1, 2, 3};
    migraphx::value v("key", values);
    EXPECT(v.is_array());
    EXPECT(v.get_key() == "key");
    EXPECT(v.get_array().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front() == migraphx::value(1));
    EXPECT(v[1] == migraphx::value(2));
    EXPECT(v.at(1) == migraphx::value(2));
    EXPECT(v.back() == migraphx::value(3));
}

TEST_CASE(value_key_array_empty)
{
    std::vector<migraphx::value> values{};
    migraphx::value v("key", values);
    EXPECT(v.is_array());
    EXPECT(v.get_key() == "key");
    EXPECT(v.get_array().size() == 0);
    EXPECT(v.size() == 0);
    EXPECT(v.empty());
}

TEST_CASE(value_construct_key_int1)
{
    migraphx::value v("one", 1);
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key() == "one");
}

TEST_CASE(value_construct_key_int2)
{
    migraphx::value v = {"one", 1};
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key() == "one");
}

TEST_CASE(value_construct_key_pair)
{
    migraphx::value v = std::make_pair("one", 1);
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key() == "one");
}

TEST_CASE(value_construct_object)
{
    migraphx::value v = {{"one", 1}, {"two", migraphx::value(2)}, {"three", 3}};
    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_key_object)
{
    std::unordered_map<std::string, migraphx::value> values = {
        {"one", 1}, {"two", migraphx::value(2)}, {"three", 3}};
    migraphx::value v("key", values);
    EXPECT(v.get_key() == "key");
    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_key_object_empty)
{
    std::unordered_map<std::string, migraphx::value> values{};
    migraphx::value v("key", values);
    EXPECT(v.get_key() == "key");
    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 0);
    EXPECT(v.size() == 0);
    EXPECT(v.empty());
    EXPECT(not v.contains("one"));
}

TEST_CASE(value_bracket_object)
{
    migraphx::value v;
    v["one"]   = 1;
    v["two"]   = migraphx::value(2);
    v["three"] = 3;

    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");
}

TEST_CASE(value_insert_object)
{
    migraphx::value v;
    v.insert({"one", 1});
    v.insert({"two", migraphx::value(2)});
    v.insert({"three", 3});
    EXPECT(v.is_object());
    EXPECT(v.get_object().size() == 3);
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_emplace_object)
{
    migraphx::value v;
    v.emplace("one", 1);
    v.emplace("two", migraphx::value(2));
    v.emplace("three", 3);
    EXPECT(v.is_object());
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_construct_object_string_value)
{
    migraphx::value v = {{"one", "onev"}, {"two", "twov"}};
    EXPECT(v.is_object());
    EXPECT(v.size() == 2);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.at("one").is_string());
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("one").get_string() == "onev");
    EXPECT(v.at("two").is_string());
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("two").get_string() == "twov");
}

TEST_CASE(value_construct_object_string_mixed_value)
{
    migraphx::value v = {{"one", "onev"}, {"two", 2}};
    EXPECT(v.is_object());
    EXPECT(v.size() == 2);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.at("one").is_string());
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("one").get_string() == "onev");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("two").get_int64() == 2);
}

TEST_CASE(value_compare)
{
    EXPECT(migraphx::value(1) == migraphx::value(1));
    EXPECT(migraphx::value("key", 1) == migraphx::value("key", 1));
    EXPECT(migraphx::value(1) != migraphx::value(2));
    EXPECT(migraphx::value("key", 1) != migraphx::value("key", 2));
    EXPECT(migraphx::value("key1", 1) != migraphx::value("key2", 1));
    EXPECT(migraphx::value(1) < migraphx::value(2));
    EXPECT(migraphx::value(1) <= migraphx::value(2));
    EXPECT(migraphx::value(1) <= migraphx::value(1));
    EXPECT(migraphx::value(2) > migraphx::value(1));
    EXPECT(migraphx::value(2) >= migraphx::value(1));
    EXPECT(migraphx::value(1) >= migraphx::value(1));
}

TEST_CASE(value_to_from_string)
{
    migraphx::value v = "1";
    EXPECT(v.to<std::string>() == "1");
    EXPECT(v.to<int>() == 1);
    EXPECT(migraphx::float_equal(v.to<float>(), 1.0));
}

TEST_CASE(value_to_from_int)
{
    migraphx::value v = 1;
    EXPECT(v.to<std::string>() == "1");
    EXPECT(v.to<int>() == 1);
    EXPECT(migraphx::float_equal(v.to<float>(), 1.0));
}

TEST_CASE(value_to_from_float)
{
    migraphx::value v = 1.5;
    EXPECT(v.to<std::string>() == "1.5");
    EXPECT(v.to<int>() == 1);
    EXPECT(migraphx::float_equal(v.to<float>(), 1.5));
}

TEST_CASE(value_to_from_pair)
{
    migraphx::value v = {"one", 1};
    EXPECT(bool{v.to<std::pair<std::string, std::string>>() ==
                std::pair<std::string, std::string>("one", "1")});
    EXPECT(bool{v.to<std::pair<std::string, int>>() == std::pair<std::string, int>("one", 1)});
    EXPECT(
        bool{v.to<std::pair<std::string, float>>() == std::pair<std::string, float>("one", 1.0)});
}

TEST_CASE(value_to_struct)
{
    migraphx::value v = 1;
    struct local
    {
        int i   = 0;
        local() = default;
        local(int ii) : i(ii) {}
    };
    EXPECT(v.to<local>().i == 1);
}

TEST_CASE(value_to_error1)
{
    migraphx::value v = {1, 2, 3};
    EXPECT(test::throws([&] { v.to<int>(); }));
}

TEST_CASE(value_to_error2)
{
    migraphx::value v = 1;
    struct local
    {
    };
    EXPECT(test::throws([&] { v.to<local>(); }));
}

TEST_CASE(value_to_error_parse)
{
    migraphx::value v = "abc";
    EXPECT(test::throws([&] { v.to<int>(); }));
}

TEST_CASE(value_to_vector)
{
    migraphx::value v  = {1, 2, 3};
    std::vector<int> a = {1, 2, 3};
    EXPECT(v.to_vector<int>() == a);
}

TEST_CASE(not_array)
{
    migraphx::value v = 1;
    EXPECT(v.size() == 0);
    EXPECT(not v.contains("???"));
    EXPECT(test::throws([&] { v.at(0); }));
    EXPECT(test::throws([&] { v.at("???"); }));
    EXPECT(v.data() == nullptr);
    [=] {
        EXPECT(test::throws([&] { v.at(0); }));
        EXPECT(test::throws([&] { v.at("???"); }));
        EXPECT(v.data() == nullptr);
    }();
}

TEST_CASE(print)
{
    std::stringstream ss;
    migraphx::value v = {1, {{"one", 1}, {"two", 2}}, {1, 2}, {}};
    ss << v;
    EXPECT(ss.str() == "{1, {one: 1, two: 2}, {1, 2}, null}");
}

TEST_CASE(value_clear)
{
    migraphx::value values = {1, 2, 3};
    EXPECT(values.is_array());
    EXPECT(values.size() == 3);
    values.clear();
    EXPECT(values.empty());

    values.push_back(3);
    EXPECT(values.size() == 1);
    EXPECT(values.at(0).to<int>() == 3);
}

TEST_CASE(value_clear_non_array)
{
    migraphx::value values = 1.0;
    EXPECT(test::throws([&] { values.clear(); }));
}

TEST_CASE(value_clear_object)
{
    migraphx::value values = {{"a", 1}, {"b", 2}};
    EXPECT(values.is_object());
    EXPECT(values.size() == 2);
    values.clear();
    EXPECT(values.empty());

    values["c"] = 3;
    EXPECT(values.size() == 1);
    EXPECT(values.at("c").to<int>() == 3);
}

TEST_CASE(value_clear_empty_array)
{
    migraphx::value values = migraphx::value::array{};
    EXPECT(values.empty());
    values.clear();
    EXPECT(values.empty());
}

TEST_CASE(value_clear_empty_object)
{
    migraphx::value values = migraphx::value::object{};
    EXPECT(values.empty());
    values.clear();
    EXPECT(values.empty());
}

TEST_CASE(value_resize)
{
    migraphx::value values = {1, 2, 3};
    EXPECT(values.is_array());
    EXPECT(values.size() == 3);
    values.resize(5);
    EXPECT(values.size() == 5);

    EXPECT(values.at(3).is_null());
    EXPECT(values.at(4).is_null());
}

TEST_CASE(value_resize_with_value)
{
    migraphx::value values = {1, 2, 3};
    EXPECT(values.is_array());
    EXPECT(values.size() == 3);
    values.resize(5, 7);
    EXPECT(values.size() == 5);

    EXPECT(values.at(3).to<int>() == 7);
    EXPECT(values.at(4).to<int>() == 7);
}

TEST_CASE(value_resize_empty_array)
{
    migraphx::value values = migraphx::value::array{};
    EXPECT(values.is_array());
    EXPECT(values.empty());
    values.resize(3);
    EXPECT(values.size() == 3);

    EXPECT(values.at(0).is_null());
    EXPECT(values.at(1).is_null());
    EXPECT(values.at(2).is_null());
}

TEST_CASE(value_resize_object)
{
    migraphx::value values = migraphx::value::object{};
    EXPECT(values.is_object());
    EXPECT(test::throws([&] { values.resize(4); }));
}

TEST_CASE(value_resize_n_object)
{
    migraphx::value values = migraphx::value::object{};
    EXPECT(values.is_object());
    EXPECT(test::throws([&] { values.resize(4, ""); }));
}

TEST_CASE(value_assign_construct_from_vector)
{
    std::vector<int> v     = {1, 2, 3};
    migraphx::value values = v;
    EXPECT(values.to_vector<int>() == v);
}

TEST_CASE(value_construct_from_vector)
{
    std::vector<int> v = {1, 2, 3};
    migraphx::value values(v);
    EXPECT(values.to_vector<int>() == v);
}

TEST_CASE(value_assign_from_vector)
{
    std::vector<int> v = {1, 2, 3};
    migraphx::value values{};
    values = v;
    EXPECT(values.to_vector<int>() == v);
}

TEST_CASE(value_init_from_vector)
{
    std::vector<int> v     = {1, 2, 3};
    migraphx::value values = {{"a", v}};
    EXPECT(values.at("a").to_vector<int>() == v);
}

TEST_CASE(value_binary_default)
{
    migraphx::value v;
    v = migraphx::value::binary{};
    EXPECT(v.is_binary());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_binary)
{
    migraphx::value v;
    std::vector<std::uint8_t> data(20);
    std::iota(data.begin(), data.end(), 0);
    v = migraphx::value::binary{data};
    EXPECT(v.is_binary());
    EXPECT(v.get_binary().size() == data.size());
    EXPECT(v.get_binary() == data);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_binary_object)
{
    std::vector<std::uint8_t> data(20);
    std::iota(data.begin(), data.end(), 0);
    migraphx::value v = {{"data", migraphx::value::binary{data}}};

    EXPECT(v["data"].is_binary());
    EXPECT(v["data"].get_binary().size() == data.size());
    EXPECT(v["data"].get_binary() == data);
}

TEST_CASE(value_binary_object_conv)
{
    std::vector<std::int8_t> data(20);
    std::iota(data.begin(), data.end(), 0);
    migraphx::value v = {{"data", migraphx::value::binary{data}}};

    EXPECT(v["data"].is_binary());
    EXPECT(v["data"].get_binary().size() == data.size());
    EXPECT(migraphx::equal(v["data"].get_binary(), data));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
