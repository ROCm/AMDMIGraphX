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
#include <migraphx/msgpack.hpp>
#include <migraphx/serialize.hpp>
#include <msgpack.hpp>
#include <variant>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct msgpack_chunk
{
    std::vector<value> chunks;

    value as_value() const
    {
        if(chunks.empty())
            return {};
        const value& v = chunks.front();
        if(v.is_array() or v.is_object())
        {
            std::vector<value> values = v.is_array() ? v.get_array() : v.get_object();
            std::for_each(chunks.begin() + 1, chunks.end(), [&](const auto& chunk) {
                values.insert(values.end(), chunk.begin(), chunk.end());
            });
            return values;
        }
        else if(v.is_binary())
        {
            value::binary data = v.get_binary();
            std::for_each(chunks.begin() + 1, chunks.end(), [&](const auto& chunk) {
                const value::binary& b = chunk.get_binary();
                data.insert(data.end(), b.begin(), b.end());
            });
            return data;
        }
        MIGRAPHX_THROW("Incorrect chunking");
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
{
    namespace adaptor {

    template <>
    struct convert<migraphx::value>
    {
        const msgpack::object& operator()(const msgpack::object& o, migraphx::value& v) const
        {
            switch(o.type)
            {
            case msgpack::type::NIL: {
                v = nullptr;
                break;
            }
            case msgpack::type::BOOLEAN: {
                v = o.as<bool>();
                break;
            }
            case msgpack::type::POSITIVE_INTEGER: {
                v = o.as<std::uint64_t>();
                break;
            }
            case msgpack::type::NEGATIVE_INTEGER: {
                v = o.as<std::int64_t>();
                break;
            }
            case msgpack::type::FLOAT32:
            case msgpack::type::FLOAT64: {
                v = o.as<double>();
                break;
            }
            case msgpack::type::STR: {
                v = o.as<std::string>();
                break;
            }
            case msgpack::type::ARRAY: {
                if(o.via.array.size == 0)
                {
                    v = migraphx::value::array{};
                    break;
                }
                std::variant<migraphx::value::array,
                             migraphx::value::binary,
                             migraphx::value::object>
                    r;
                switch(o.via.array.ptr->type)
                {
                case msgpack::type::BIN: {
                    r = migraphx::value::binary{};
                    break;
                }
                case msgpack::type::ARRAY: {
                    r = migraphx::value::array{};
                    break;
                }
                case msgpack::type::MAP: {
                    r = migraphx::value::object{};
                    break;
                }
                default: MIGRAPHX_THROW("Incorrect chunking");
                }
            }
                std::for_each(
                    o.via.array.ptr,
                    o.via.array.ptr + o.via.array.size,
                    [&](const msgpack::object& sa) {
                        std::visit(
                            overload(
                                [&](migraphx::value::binary& bin) {
                                    bin.insert(
                                        bin.end(), o.via.bin.ptr, o.via.bin.ptr + o.via.bin.size);
                                },
                                [&](migraphx::value::array& arr) {
                                    std::for_each(sa.via.array.ptr,
                                                  sa.via.array.ptr + sa.via.array.size,
                                                  [&](const msgpack::object& so) {
                                                      arr.push_back(so.as<migraphx::value>());
                                                  });
                                },
                                [&](migraphx::value::object& obj) {
                                    std::for_each(sa.via.map.ptr,
                                                  sa.via.map.ptr + sa.via.map.size,
                                                  [&](const msgpack::object_kv& p) {
                                                      obj[p.key.as<std::string>()] =
                                                          p.val.as<migraphx::value>();
                                                  });
                                }),
                            r);
                    });
                std::visit([&](const auto& x) { v = x; }, r);
                break;
            }
        case msgpack::type::MAP:
        case msgpack::type::BIN: {
            MIGRAPHX_THROW("Unexpected msgpack type");
            break;
        }
            case msgpack::type::EXT: {
                MIGRAPHX_THROW("msgpack EXT type not supported.");
            }
            }
            return o;
        }
    };

    template <>
    struct pack<migraphx::value::binary>
    {
        template <class Stream>
        packer<Stream>& operator()(msgpack::packer<Stream>& o,
                                   const migraphx::value::binary& x) const
        {
            const auto* data = reinterpret_cast<const char*>(x.data());
            auto size        = x.size();
            o.pack_bin(size);
            o.pack_bin_body(data, size);
            return o;
        }
    };

    template <>
    struct pack<migraphx::value>
    {
        template <class Stream>
        void write(msgpack::packer<Stream>& o, const std::nullptr_t&) const
        {
            o.pack_nil();
        }
        template <class Stream, class T>
        void write(msgpack::packer<Stream>& o, const T& x) const
        {
            o.pack(x);
        }
        template <class Stream>
        void write(msgpack::packer<Stream>& o, const std::vector<migraphx::value>& v) const
        {
            if(v.empty())
            {
                o.pack_array(0);
                return;
            }
            if(not v.front().get_key().empty())
            {
                o.pack_map(v.size());
                for(auto&& x : v)
                {
                    o.pack(x.get_key());
                    o.pack(x.without_key());
                }
            }
            else
            {
                o.pack_array(v.size());
                for(auto&& x : v)
                {
                    o.pack(x);
                }
            }
        }
        template <class Stream>
        packer<Stream>& operator()(msgpack::packer<Stream>& o, const migraphx::value& v) const
        {
            v.visit_value([&](auto&& x) { this->write(o, x); });
            return o;
        }
    };

    } // namespace adaptor
} // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
} // namespace msgpack

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct vector_stream
{
    std::vector<char> buffer{};
    vector_stream& write(const char* b, std::size_t n)
    {
        buffer.insert(buffer.end(), b, b + n);
        return *this;
    }
};

struct writer_stream
{
    std::function<void(const char*, std::size_t)> writer;
    writer_stream& write(const char* b, std::size_t n)
    {
        writer(b, n);
        return *this;
    }
};

void to_msgpack(const value& v, std::function<void(const char*, std::size_t)> writer)
{
    writer_stream ws{std::move(writer)};
    msgpack::pack(ws, v);
}

std::vector<char> to_msgpack(const value& v)
{
    vector_stream vs;
    msgpack::pack(vs, v);
    return vs.buffer;
}
value from_msgpack(const char* buffer, std::size_t size)
{
    msgpack::object_handle oh = msgpack::unpack(buffer, size);
    return oh.get().as<value>();
}
value from_msgpack(const std::vector<char>& buffer)
{
    return from_msgpack(buffer.data(), buffer.size());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
