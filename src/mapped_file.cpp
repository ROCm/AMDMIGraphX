/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/mapped_file.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/manage_ptr.hpp>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>

#ifdef _WIN32
// cppcheck-suppress definePrefix
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// Paths of the live mappings in this process, counted since one file may be mapped several times
struct mapping_registry
{
    std::mutex mutex;
    std::unordered_map<std::string, std::size_t> counts;
};

mapping_registry& registry()
{
    static mapping_registry r;
    return r;
}

// Distinct spellings of one file map to the same key; a file that does not exist has none
std::string registry_key(const fs::path& p)
{
    std::error_code ec;
    auto c = fs::canonical(p, ec);
    if(ec)
        return {};
    return c.string();
}

} // namespace

#ifdef _WIN32

using handle_ptr = MIGRAPHX_MANAGE_PTR(HANDLE, CloseHandle);

static std::string last_error() { return std::to_string(GetLastError()); }

struct mapped_file_impl
{
    explicit mapped_file_impl(const fs::path& p) : path(p)
    {
        HANDLE f = CreateFileW(p.wstring().c_str(),
                               GENERIC_READ,
                               FILE_SHARE_READ,
                               nullptr,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL,
                               nullptr);
        if(f == INVALID_HANDLE_VALUE)
            MIGRAPHX_THROW("Failure opening file: " + p.string() + " (" + last_error() + ")");
        handle_ptr file{f};
        LARGE_INTEGER file_size{};
        if(GetFileSizeEx(file.get(), &file_size) == 0)
            MIGRAPHX_THROW("Failure reading size of file: " + p.string() + " (" + last_error() +
                           ")");
        size = static_cast<std::size_t>(file_size.QuadPart);
        if(size == 0)
            return;
        // The view keeps the file mapped after both handles are closed
        handle_ptr mapping{CreateFileMappingW(file.get(), nullptr, PAGE_READONLY, 0, 0, nullptr)};
        if(mapping == nullptr)
            MIGRAPHX_THROW("Failure mapping file: " + p.string() + " (" + last_error() + ")");
        data = static_cast<const char*>(MapViewOfFile(mapping.get(), FILE_MAP_READ, 0, 0, 0));
        if(data == nullptr)
            MIGRAPHX_THROW("Failure mapping file: " + p.string() + " (" + last_error() + ")");
    }

    mapped_file_impl(const mapped_file_impl&)            = delete;
    mapped_file_impl& operator=(const mapped_file_impl&) = delete;

    ~mapped_file_impl()
    {
        if(data != nullptr)
            UnmapViewOfFile(data);
    }

    fs::path path;
    std::string key;
    const char* data = nullptr;
    std::size_t size = 0;
};

#else

struct mapped_file_impl
{
    explicit mapped_file_impl(const fs::path& p) : path(p)
    {
        int fd = open(p.string().c_str(), O_RDONLY | O_CLOEXEC); // NOLINT
        if(fd < 0)
            MIGRAPHX_THROW("Failure opening file: " + p.string());
        struct stat st = {};
        if(fstat(fd, &st) != 0)
        {
            close(fd);
            MIGRAPHX_THROW("Failure reading size of file: " + p.string());
        }
        size = static_cast<std::size_t>(st.st_size);
        if(size == 0)
        {
            close(fd);
            return;
        }
        void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0); // NOLINT
        // The mapping stays valid after the descriptor is closed
        close(fd);
        if(m == MAP_FAILED) // NOLINT
            MIGRAPHX_THROW("Failure mapping file: " + p.string());
        data = static_cast<const char*>(m);
    }

    mapped_file_impl(const mapped_file_impl&)            = delete;
    mapped_file_impl& operator=(const mapped_file_impl&) = delete;

    ~mapped_file_impl()
    {
        if(data != nullptr)
            munmap(const_cast<char*>(data), size); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    }

    fs::path path;
    std::string key;
    const char* data = nullptr;
    std::size_t size = 0;
};

#endif

// The key is kept from construction since the file may be renamed or removed while mapped. The
// lock is held while mapping so replace_unless_mapped cannot replace the file in between.
mapped_file::mapped_file(const fs::path& p)
{
    std::lock_guard<std::mutex> lock(registry().mutex);
    impl      = std::make_unique<mapped_file_impl>(p);
    impl->key = registry_key(p);
    registry().counts[impl->key]++;
}

mapped_file::~mapped_file()
{
    std::lock_guard<std::mutex> lock(registry().mutex);
    auto it = registry().counts.find(impl->key);
    if(it == registry().counts.end())
        return;
    if(--it->second == 0)
        registry().counts.erase(it);
}

const char* mapped_file::data() const { return impl->data; }

std::size_t mapped_file::size() const { return impl->size; }

const fs::path& mapped_file::path() const { return impl->path; }

bool mapped_file::is_mapped(const fs::path& p)
{
    auto key = registry_key(p);
    if(key.empty())
        return false;
    std::lock_guard<std::mutex> lock(registry().mutex);
    return registry().counts.count(key) > 0;
}

void mapped_file::replace_unless_mapped(const fs::path& p, const std::function<void()>& replace)
{
    std::lock_guard<std::mutex> lock(registry().mutex);
    auto key = registry_key(p);
    if(not key.empty() and registry().counts.count(key) > 0)
        MIGRAPHX_THROW("Cannot overwrite " + p.string() + " while a loaded program maps it");
    replace();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
