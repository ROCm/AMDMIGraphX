/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstring>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Leave an extra byte for error checking
constexpr std::size_t msgpack_size_limit = std::numeric_limits<uint32_t>::max() - 1;

template <class Range>
static std::size_t msgpack_chunk_size(const Range& r)
{
    return 1 + (std::max<size_t>(1, r.size()) - 1) / msgpack_size_limit;
}

template <class Iterator, class F>
static void msgpack_chunk_for_each(Iterator start, Iterator last, F f)
{
    while(std::distance(start, last) > msgpack_size_limit)
    {
        auto next = std::next(start, msgpack_size_limit);
        f(start, next);
        start = next;
    }
    f(start, last);
}

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
            /* Intentional Fall Through This is due to value encoding floats as double. */
            case msgpack::type::FLOAT64: {
                v = o.as<double>();
                break;
            }
            case msgpack::type::STR: {
                v = o.as<std::string>();
                break;
            }
            case msgpack::type::BIN: {
                // For backwards compatibility
                v = migraphx::value::binary{o.via.bin.ptr, o.via.bin.size};
                break;
            }
            case msgpack::type::ARRAY: {
                if(o.via.array.size != 0 and o.via.array.ptr->type == msgpack::type::BIN)
                {
                    auto bin = migraphx::value::binary{};
                    std::for_each(
                        o.via.array.ptr,
                        o.via.array.ptr + o.via.array.size,
                        [&](const msgpack::object& so) {
                            bin.insert(bin.end(), so.via.bin.ptr, so.via.bin.ptr + so.via.bin.size);
                        });
                    v = bin;
                }
                else
                {
                    migraphx::value r = migraphx::value::array{};
                    std::for_each(
                        o.via.array.ptr,
                        o.via.array.ptr + o.via.array.size,
                        [&](const msgpack::object& so) { r.push_back(so.as<migraphx::value>()); });
                    v = r;
                }
                break;
            }
            case msgpack::type::MAP: {
                migraphx::value r = migraphx::value::object{};
                std::for_each(o.via.map.ptr,
                              o.via.map.ptr + o.via.map.size,
                              [&](const msgpack::object_kv& p) {
                                  r[p.key.as<std::string>()] = p.val.as<migraphx::value>();
                              });
                v = r;
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
            o.pack_array(migraphx::msgpack_chunk_size(x));
            migraphx::msgpack_chunk_for_each(
                data, data + size, [&](const char* start, const char* last) {
                    o.pack_bin(last - start);
                    o.pack_bin_body(start, last - start);
                });
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
            if(v.size() > migraphx::msgpack_size_limit)
                MIGRAPHX_THROW("Size is too large for msgpack");
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

namespace {

struct msgpack_cursor
{
    const unsigned char* pos;
    const unsigned char* last;

    std::size_t remaining() const { return static_cast<std::size_t>(last - pos); }

    // Returns the next n bytes and moves past them, or nullptr if fewer are left.
    const unsigned char* take(std::size_t n)
    {
        if(n > remaining())
            return nullptr;
        const auto* result = pos;
        pos += n;
        return result;
    }
};

enum class msgpack_kind
{
    nil,
    boolean,
    signed_int,
    unsigned_int,
    floating,
    str,
    bin,
    array,
    map
};

struct msgpack_item
{
    msgpack_kind kind = msgpack_kind::nil;
    // Element count for array and map, byte length for str and bin
    std::uint64_t size = 0;
    std::int64_t i     = 0;
    std::uint64_t u    = 0;
    double d           = 0;
    bool b             = false;
    // Body of str and bin, which the cursor has already moved past
    const unsigned char* data = nullptr;
};

optional<std::uint64_t> read_big_endian(msgpack_cursor& c, std::size_t n)
{
    const auto* bytes = c.take(n);
    if(bytes == nullptr)
        return nullopt;
    return std::accumulate(bytes,
                           bytes + n,
                           std::uint64_t{0},
                           [](std::uint64_t acc, unsigned char b) { return (acc << 8U) | b; });
}

// Signed integers are classified like the decoder does: a non-negative one is unsigned.
void set_signed(msgpack_item& item, std::int64_t i)
{
    if(i < 0)
    {
        item.kind = msgpack_kind::signed_int;
        item.i    = i;
    }
    else
    {
        item.kind = msgpack_kind::unsigned_int;
        item.u    = static_cast<std::uint64_t>(i);
    }
}

// Reads the body of a str or bin of n bytes, which the cursor moves past.
optional<msgpack_item> read_body(msgpack_cursor& c, std::uint64_t n, msgpack_kind kind)
{
    if(n > c.remaining())
        return nullopt;
    msgpack_item item;
    item.kind = kind;
    item.size = n;
    item.data = c.take(static_cast<std::size_t>(n));
    return item;
}

// Reads a size that is stored in `width` bytes, and the body if the kind has one.
optional<msgpack_item> read_sized(msgpack_cursor& c, std::size_t width, msgpack_kind kind)
{
    auto n = read_big_endian(c, width);
    if(not n)
        return nullopt;
    if(kind == msgpack_kind::str or kind == msgpack_kind::bin)
        return read_body(c, *n, kind);
    msgpack_item item;
    item.kind = kind;
    item.size = *n;
    return item;
}

// Reads one item header, and the body of a str or bin. The elements of an array or map are not
// read. Returns nullopt for anything that is truncated or that the decoder does not support.
optional<msgpack_item> read_item(msgpack_cursor& c)
{
    const auto* tag = c.take(1);
    if(tag == nullptr)
        return nullopt;
    const unsigned char t = *tag;
    msgpack_item item;
    if(t <= 0x7f)
    {
        item.kind = msgpack_kind::unsigned_int;
        item.u    = t;
        return item;
    }
    if(t >= 0xe0)
    {
        set_signed(item, static_cast<std::int8_t>(t));
        return item;
    }
    if(t >= 0xa0 and t <= 0xbf)
        return read_body(c, t & 0x1fU, msgpack_kind::str);
    if(t >= 0x90 and t <= 0x9f)
    {
        item.kind = msgpack_kind::array;
        item.size = t & 0x0fU;
        return item;
    }
    if(t >= 0x80 and t <= 0x8f)
    {
        item.kind = msgpack_kind::map;
        item.size = t & 0x0fU;
        return item;
    }
    switch(t)
    {
    case 0xc0: item.kind = msgpack_kind::nil; return item;
    case 0xc2:
    case 0xc3:
        item.kind = msgpack_kind::boolean;
        item.b    = t == 0xc3;
        return item;
    case 0xc4: return read_sized(c, 1, msgpack_kind::bin);
    case 0xc5: return read_sized(c, 2, msgpack_kind::bin);
    case 0xc6: return read_sized(c, 4, msgpack_kind::bin);
    case 0xd9: return read_sized(c, 1, msgpack_kind::str);
    case 0xda: return read_sized(c, 2, msgpack_kind::str);
    case 0xdb: return read_sized(c, 4, msgpack_kind::str);
    case 0xdc: return read_sized(c, 2, msgpack_kind::array);
    case 0xdd: return read_sized(c, 4, msgpack_kind::array);
    case 0xde: return read_sized(c, 2, msgpack_kind::map);
    case 0xdf: return read_sized(c, 4, msgpack_kind::map);
    case 0xca: {
        auto bits = read_big_endian(c, 4);
        if(not bits)
            return nullopt;
        auto raw = static_cast<std::uint32_t>(*bits);
        float f  = 0;
        std::memcpy(&f, &raw, sizeof(f));
        item.kind = msgpack_kind::floating;
        item.d    = f;
        return item;
    }
    case 0xcb: {
        auto bits = read_big_endian(c, 8);
        if(not bits)
            return nullopt;
        double d = 0;
        std::memcpy(&d, &*bits, sizeof(d));
        item.kind = msgpack_kind::floating;
        item.d    = d;
        return item;
    }
    case 0xcc:
    case 0xcd:
    case 0xce:
    case 0xcf: {
        auto bits = read_big_endian(c, std::size_t{1} << (t - 0xccU));
        if(not bits)
            return nullopt;
        item.kind = msgpack_kind::unsigned_int;
        item.u    = *bits;
        return item;
    }
    case 0xd0:
    case 0xd1:
    case 0xd2:
    case 0xd3: {
        auto bits = read_big_endian(c, std::size_t{1} << (t - 0xd0U));
        if(not bits)
            return nullopt;
        switch(t)
        {
        case 0xd0: set_signed(item, static_cast<std::int8_t>(*bits)); break;
        case 0xd1: set_signed(item, static_cast<std::int16_t>(*bits)); break;
        case 0xd2: set_signed(item, static_cast<std::int32_t>(*bits)); break;
        default: set_signed(item, static_cast<std::int64_t>(*bits)); break;
        }
        return item;
    }
    // 0xc1 is never used, and the ext types are not supported by the decoder
    default: return nullopt;
    }
}

// Moves past one value, including everything inside an array or map. It counts the items still
// to skip instead of recursing, so the depth of the nesting does not matter.
bool skip_value(msgpack_cursor& c)
{
    std::uint64_t pending = 1;
    while(pending > 0)
    {
        // Every pending item takes at least one byte
        if(pending > c.remaining())
            return false;
        auto item = read_item(c);
        if(not item)
            return false;
        pending--;
        if(item->kind == msgpack_kind::array)
            pending += item->size;
        else if(item->kind == msgpack_kind::map)
            pending += 2 * item->size;
    }
    return true;
}

value scalar_to_value(const msgpack_item& item)
{
    // Anything that is not a scalar becomes an empty array, which no version check accepts.
    value result = value::array{};
    switch(item.kind)
    {
    case msgpack_kind::nil: result = nullptr; break;
    case msgpack_kind::boolean: result = item.b; break;
    case msgpack_kind::signed_int: result = item.i; break;
    case msgpack_kind::unsigned_int: result = item.u; break;
    case msgpack_kind::floating: result = item.d; break;
    case msgpack_kind::str:
        result = std::string(reinterpret_cast<const char*>(item.data),
                             static_cast<std::size_t>(item.size));
        break;
    case msgpack_kind::bin:
    case msgpack_kind::array:
    case msgpack_kind::map: break;
    }
    return result;
}

} // namespace

optional<value> msgpack_peek_field(const char* buffer, std::size_t size, const std::string& key)
{
    if(buffer == nullptr)
        return nullopt;
    msgpack_cursor c{reinterpret_cast<const unsigned char*>(buffer),
                     reinterpret_cast<const unsigned char*>(buffer) + size};
    auto top = read_item(c);
    // Each entry takes at least two bytes
    if(not top or top->kind != msgpack_kind::map or top->size > c.remaining() / 2)
        return nullopt;
    std::uint64_t remaining_entries = top->size;
    while(remaining_entries > 0)
    {
        remaining_entries--;
        auto entry_key = read_item(c);
        if(not entry_key or entry_key->kind != msgpack_kind::str)
            return nullopt;
        if(entry_key->size == key.size() and
           std::memcmp(entry_key->data, key.data(), key.size()) == 0)
        {
            auto field = read_item(c);
            if(not field)
                return nullopt;
            return scalar_to_value(*field);
        }
        if(not skip_value(c))
            return nullopt;
    }
    return nullopt;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
