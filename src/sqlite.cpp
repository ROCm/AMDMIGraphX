#include <migraphx/sqlite.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/errors.hpp>
#include <sqlite3.h>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using sqlite3_ptr = MIGRAPHX_MANAGE_PTR(sqlite3*, sqlite3_close);

struct sqlite_impl
{
    sqlite3* get() const { return ptr.get(); }
    void open(const fs::path& p, int flags)
    {
        sqlite3* ptr_tmp = nullptr;
        int rc = sqlite3_open_v2(p.string().c_str(), &ptr_tmp, flags, nullptr);
        ptr = sqlite3_ptr{ptr_tmp};
        if (rc != 0)
            MIGRAPHX_THROW("error opening " + p.string() + ": " + error_message());
    }

    template<class F>
    void exec(const char*sql, F f)
    {
        auto callback = [](void* obj, auto... xs) -> int {
            try
            {
                auto g = static_cast<const F*>(obj);
                (*g)(xs...);
                return 0;
            }
            catch(...)
            {
                return -1;
            }
        };
        int rc = sqlite3_exec(get(), sql, callback, &f, nullptr);
        if (rc != 0)
            MIGRAPHX_THROW(error_message());
    }

    std::string error_message() const
    {
        std::string msg = "sqlite3: ";
        return msg + sqlite3_errmsg(get());
    }
    sqlite3_ptr ptr;
};

sqlite sqlite::read(const fs::path& p)
{
    sqlite r;
    r.impl = std::make_shared<sqlite_impl>();
    r.impl->open(p, SQLITE_OPEN_READONLY);
    return r;
}

sqlite sqlite::write(const fs::path& p)
{
    sqlite r;
    r.impl = std::make_shared<sqlite_impl>();
    r.impl->open(p, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
    return r;
}

std::vector<std::unordered_map<std::string, std::string>> sqlite::execute(const std::string& s)
{
    std::vector<std::unordered_map<std::string, std::string>> result;
    impl->exec(s.c_str(), [&](int n, char** texts, char** names) {
        std::unordered_map<std::string, std::string> row;
        row.reserve(n);
        std::transform(names, names+n, texts, std::inserter(row, row.begin()), [&](const char* name, const char* text) {
            return std::make_pair(name, text);
        });
        result.push_back(row);
    });
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
