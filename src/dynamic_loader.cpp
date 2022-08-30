/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/tmp_dir.hpp>
#include <utility>

#include <dlfcn.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct dynamic_loader_impl
{
    dynamic_loader_impl() = default;
    dynamic_loader_impl(const fs::path& p, std::shared_ptr<tmp_dir> t = nullptr)
        : handle(dlopen(p.string().c_str(), RTLD_LAZY), &dlclose), temp(std::move(t))
    {
    }

    static std::shared_ptr<dynamic_loader_impl> from_buffer(const char* image, std::size_t size)
    {
        auto t = std::make_shared<tmp_dir>("dloader");
        auto f = t->path / "libtmp.so";
        write_buffer(f.string(), image, size);
        return std::make_shared<dynamic_loader_impl>(f, t);
    }

    std::shared_ptr<void> handle  = nullptr;
    std::shared_ptr<tmp_dir> temp = nullptr;
};

dynamic_loader::dynamic_loader(const fs::path& p) : impl(std::make_shared<dynamic_loader_impl>(p))
{
}

dynamic_loader::dynamic_loader(const char* image, std::size_t size)
    : impl(dynamic_loader_impl::from_buffer(image, size))
{
}

dynamic_loader::dynamic_loader(const std::vector<char>& buffer)
    : impl(dynamic_loader_impl::from_buffer(buffer.data(), buffer.size()))
{
}

std::shared_ptr<void> dynamic_loader::get_symbol(const std::string& name) const
{
    dlerror();
    void* symbol = dlsym(impl->handle.get(), name.c_str());
    if(symbol == nullptr)
        MIGRAPHX_THROW("Symbol not found: " + name);
    return {impl, symbol};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
