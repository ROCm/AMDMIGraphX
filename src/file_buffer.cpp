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
#include <migraphx/file_buffer.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/fileutils.hpp>
#include <fstream>
#include <iostream>
#include <utility>

#ifdef _WIN32
#include <migraphx/manage_ptr.hpp>
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

template <class T>
static T generic_read_file(const fs::path& filename, size_t offset = 0, size_t nbytes = 0)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if(not is.is_open())
        MIGRAPHX_THROW("Failure opening file: " + filename);
    std::size_t file_size = is.tellg();
    if(offset > file_size)
        MIGRAPHX_THROW("offset is larger than file size");
    // if nbytes is not set, read the remaining bytes after offset
    if(nbytes == 0)
        nbytes = file_size - offset;
    else if(nbytes > file_size - offset)
        MIGRAPHX_THROW("Reading " + std::to_string(nbytes) + " bytes at offset " +
                       std::to_string(offset) + " exceeds size of file: " + filename);
    if(nbytes < 1)
        MIGRAPHX_THROW("Invalid size for: " + filename);
    is.seekg(offset, std::ios::beg);

    T buffer(nbytes, 0);
    if(not is.read(&buffer[0], nbytes))
        MIGRAPHX_THROW("Error reading file: " + filename);
    return buffer;
}

std::vector<char> read_buffer(const fs::path& filename, size_t offset, size_t nbytes)
{
    return generic_read_file<std::vector<char>>(filename, offset, nbytes);
}

std::string read_string(const fs::path& filename)
{
    return generic_read_file<std::string>(filename);
}

file_view::file_view(std::shared_ptr<const char> ptr, std::size_t size)
    : m_ptr(std::move(ptr)), m_size(size)
{
}

const char* file_view::data() const { return m_ptr.get(); }

std::size_t file_view::size() const { return m_size; }

const char* file_view::begin() const { return data(); }

const char* file_view::end() const { return data() + size(); }

std::shared_ptr<const char> file_view::ptr() const { return m_ptr; }

// For a file that cannot be mapped, such as one on a filesystem without mapping support
static file_view read_file_view(const fs::path& filename)
{
    auto buffer = std::make_shared<const std::vector<char>>(read_buffer(filename));
    return {std::shared_ptr<const char>{buffer, buffer->data()}, buffer->size()};
}

#ifdef _WIN32

using handle_ptr = MIGRAPHX_MANAGE_PTR(HANDLE, CloseHandle);

file_view map_file(const fs::path& filename)
{
    HANDLE handle = CreateFileW(filename.c_str(),
                                GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr,
                                OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL,
                                nullptr);
    if(handle == INVALID_HANDLE_VALUE)
        MIGRAPHX_THROW("Failure opening file: " + filename);
    handle_ptr file{handle};
    LARGE_INTEGER file_size = {};
    if(GetFileSizeEx(file.get(), &file_size) == 0)
        MIGRAPHX_THROW("Failure reading size of file: " + filename);
    std::size_t size = file_size.QuadPart;
    // An empty file cannot be mapped
    if(size == 0)
        return {};
    handle_ptr mapping{CreateFileMappingW(file.get(), nullptr, PAGE_READONLY, 0, 0, nullptr)};
    if(mapping == nullptr)
        return read_file_view(filename);
    // The view keeps the file open, so it outlives both handles
    void* mem = MapViewOfFile(mapping.get(), FILE_MAP_READ, 0, 0, 0);
    if(mem == nullptr)
        return read_file_view(filename);
    std::shared_ptr<void> view{mem, [](void* p) { UnmapViewOfFile(p); }};
    return {std::shared_ptr<const char>{view, static_cast<const char*>(mem)}, size};
}

#else

struct file_descriptor
{
    int fd = -1;

    explicit file_descriptor(int f) : fd(f) {}
    file_descriptor(const file_descriptor&)            = delete;
    file_descriptor& operator=(const file_descriptor&) = delete;
    ~file_descriptor()
    {
        if(fd >= 0)
            close(fd);
    }
};

file_view map_file(const fs::path& filename)
{
    file_descriptor file{open(filename.c_str(), O_RDONLY | O_CLOEXEC)};
    if(file.fd < 0)
        MIGRAPHX_THROW("Failure opening file: " + filename);
    struct stat info = {};
    if(fstat(file.fd, &info) != 0)
        MIGRAPHX_THROW("Failure reading size of file: " + filename);
    if(not S_ISREG(info.st_mode))
        MIGRAPHX_THROW("Not a regular file: " + filename);
    std::size_t size = info.st_size;
    // An empty file cannot be mapped
    if(size == 0)
        return {};
    // The mapping keeps the file open, so it outlives the close of file
    void* mem = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if(mem == MAP_FAILED)
        return read_file_view(filename);
    std::shared_ptr<void> mapping{mem, [size](void* p) { munmap(p, size); }};
    return {std::shared_ptr<const char>{mapping, static_cast<const char*>(mem)}, size};
}

#endif

void write_string(const fs::path& filename, const std::string& buffer)
{
    write_buffer(filename, buffer.data(), buffer.size());
}

void write_buffer(const fs::path& filename, const char* buffer, std::size_t size)
{
    std::ofstream os(filename, std::ios::out | std::ios::binary);
    if(os.fail())
        MIGRAPHX_THROW("Failure opening file: " + filename);
    os.write(buffer, size);
    if(os.bad())
        MIGRAPHX_THROW("Error writing file: " + filename);
}

void write_buffer(const fs::path& filename, const std::vector<char>& buffer)
{
    write_buffer(filename, buffer.data(), buffer.size());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
