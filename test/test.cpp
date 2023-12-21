#include <test.hpp>
#include <regex>

bool glob_match(const std::string& input, const std::string& pattern)
{
    return test::glob_match(input.begin(), input.end(), pattern.begin(), pattern.end());
}

TEST_CASE(globbing)
{
    for(int i = 0; i < 1000; i++)
    {
        EXPECT(not glob_match("ab", "a"));
        EXPECT(not glob_match("ba", "a"));
        EXPECT(not glob_match("bac", "a"));
        EXPECT(glob_match("ab", "ab"));

        // Star loop
        EXPECT(glob_match("/foo/bar/baz/blig/fig/blig", "/foo/*/blig"));
        EXPECT(glob_match("/foo/bar/baz/xlig/fig/blig", "/foo/*/blig"));
        EXPECT(glob_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "a*a*a*a*a*a*a*a*b"));
        EXPECT(glob_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
                          "a*a*a*a*a*a*a*a**a*a*a*a*b"));
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
}

inline void
replace_string_inplace(std::string& subject, const std::string& search, const std::string& replace)
{
    size_t pos = 0;
    while((pos = subject.find(search, pos)) != std::string::npos)
    {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
}

bool regex_match(const std::string& input, const std::string& pattern)
{
#if 0
    // replace_string_inplace(pattern, "\\", "\\\\");
    std::string grep_command = "echo '" + input + "' | grep -q -E '" + pattern + "'";
    // std::cout << grep_command << std::endl;
    auto result = std::system(grep_command.c_str());
    return result == 0;
#else
    std::regex r(pattern);
    return std::regex_search(input, r);
#endif
}

TEST_CASE(regex)
{
    for(int i = 0; i < 1000; i++)
    {
        EXPECT(not regex_match("ab", "^a$"));
        EXPECT(not regex_match("ba", "^a$"));
        EXPECT(not regex_match("bac", "^a$"));
        EXPECT(regex_match("ab", "^ab$"));

        // Star loop
        EXPECT(regex_match("/foo/bar/baz/blig/fig/blig", "^/foo/.*/blig$"));
        EXPECT(regex_match("/foo/bar/baz/xlig/fig/blig", "^/foo/.*/blig$"));
        EXPECT(
            regex_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "^a.*a.*a.*a.*a.*a.*a.*a.*b$"));
        EXPECT(regex_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
                           "^a.*a.*a.*a.*a.*a.*a.*a.*.*a.*a.*a.*a.*b$"));
        EXPECT(regex_match("aabaabaab", "^a.*$"));
        EXPECT(regex_match("aabaabaab", "^a.*b.*ab$"));
        EXPECT(regex_match("aabaabaab", "^a.*baab$"));
        EXPECT(regex_match("aabaabaab", "^aa.*$"));
        EXPECT(regex_match("aabaabaab", "^aaba.*$"));
        EXPECT(regex_match("aabaabqqbaab", "^a.*baab$"));
        EXPECT(regex_match("aabaabqqbaab", "^a.*baab$"));
        EXPECT(regex_match("abcdd", "^.*d$"));
        EXPECT(regex_match("abcdd", "^.*d.*$"));
        EXPECT(regex_match("daaadabadmanda", "^da.*da.*da.*$"));
        EXPECT(regex_match("mississippi", "^m.*issip.*$"));

        // Repeated star
        EXPECT(regex_match("aabaabqqbaab", "^a.*.*.*.*baab$"));
        EXPECT(regex_match("abcdd", "^.*.*.*d$"));
        EXPECT(regex_match("abcdd", "^.*.*.*d.*.*.*.*$"));

        // Single wildcard
        EXPECT(regex_match("abc", "^a.c$"));

        // Special characters
        EXPECT(regex_match("test.foo[gpu]", "^test\\.foo\\[gpu\\]$"));
        EXPECT(regex_match("test.foo[gpu]", "^test\\.foo\\[.*\\]$"));
        EXPECT(regex_match("test.foo[gpu]", "^.*\\[.*$"));

        EXPECT(regex_match("test.foo(gpu)", "^test\\.foo\\(gpu\\)$"));
        EXPECT(regex_match("test.foo(gpu)", "^test\\.foo\\(.*\\)$"));
        EXPECT(regex_match("test.foo(gpu)", "^.*\\(.*$"));

        EXPECT(not regex_match("test.foog", "^test\\.foo\\[gpu\\]$"));
        EXPECT(not regex_match("test.foogpu", "^test\\.foo\\[gpu\\]$"));
        EXPECT(not regex_match("test_foo", "^test\\.foo$"));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
