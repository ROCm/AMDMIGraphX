#include <test.hpp>

bool glob_match(const std::string& input, const std::string& pattern)
{
    return test::glob_match(input.begin(), input.end(), pattern.begin(), pattern.end());
}

TEST_CASE(globbing)
{
    EXPECT(not glob_match("ab", "a"));
    EXPECT(not glob_match("ba", "a"));
    EXPECT(not glob_match("bac", "a"));
    EXPECT(glob_match("ab", "ab"));

    // Star loop
    EXPECT(glob_match("/foo/bar/baz/blig/fig/blig", "/foo/*/blig"));
    EXPECT(glob_match("/foo/bar/baz/xlig/fig/blig", "/foo/*/blig"));
    EXPECT(glob_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "a*a*a*a*a*a*a*a*b"));
    EXPECT(glob_match("aabaabaab", "a*"));
    EXPECT(glob_match("aabaabaab", "a*b*ab"));
    EXPECT(glob_match("aabaabaab", "a*baab"));
    EXPECT(glob_match("aabaabaab", "aa*"));
    EXPECT(glob_match("aabaabaab", "aaba*"));
    EXPECT(glob_match("aabaabqqbaab", "a*baab"));
    EXPECT(glob_match("aabaabqqbaab", "a*baab"));
    EXPECT(glob_match("abcdd", "*d"));
    EXPECT(glob_match("abcdd", "*d*"));
    EXPECT(glob_match("daaadabadmanda", "da*da*da*"));
    EXPECT(glob_match("mississippi", "m*issip*"));

    // Repeated star
    EXPECT(glob_match("aabaabqqbaab", "a****baab"));
    EXPECT(glob_match("abcdd", "***d"));
    EXPECT(glob_match("abcdd", "***d****"));

    // Single wildcard
    EXPECT(glob_match("abc", "a?c"));

    // Special characters
    EXPECT(glob_match("test.foo[gpu]", "test.foo[gpu]"));
    EXPECT(glob_match("test.foo[gpu]", "test.foo[*]"));
    EXPECT(glob_match("test.foo[gpu]", "*[*"));

    EXPECT(glob_match("test.foo(gpu)", "test.foo(gpu)"));
    EXPECT(glob_match("test.foo(gpu)", "test.foo(*)"));
    EXPECT(glob_match("test.foo(gpu)", "*(*"));

    EXPECT(not glob_match("test.foog", "test.foo[gpu]"));
    EXPECT(not glob_match("test.foogpu", "test.foo[gpu]"));
    EXPECT(not glob_match("test_foo", "test.foo"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
