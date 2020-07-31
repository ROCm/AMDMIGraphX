#include <migraphx/msgpack.hpp>
#include <migraphx/serialize.hpp>
#include <msgpack.hpp>

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
namespace adaptor {

template<>
struct convert<migraphx::value> {
    const msgpack::object& operator()(const msgpack::object& o, migraphx::value& v) const {
        switch(o.type) {
            case msgpack::type::NIL:
            {
                v = nullptr;
                break;    
            }
            case msgpack::type::BOOLEAN:
            {
                v = o.as<bool>();
                break;    
            }
            case msgpack::type::POSITIVE_INTEGER:
            {
                v = o.as<std::uint64_t>();
                break;    
            }
            case msgpack::type::NEGATIVE_INTEGER:
            {
                v = o.as<std::int64_t>();
                break;    
            }
            case msgpack::type::FLOAT32:
            case msgpack::type::FLOAT64:
            {
                v = o.as<double>();
                break;    
            }
            case msgpack::type::STR:
            {
                v = o.as<std::string>();
                break;    
            }
            case msgpack::type::BIN:
            {
                throw msgpack::type_error();
            }
            case msgpack::type::ARRAY:
            {
                migraphx::value r;
                std::for_each(o.via.array.ptr, o.via.array.ptr+o.via.array.size, [&](auto&& so) {
                    r.push_back(so.as<migraphx::value>());
                });
                v = r;
                break;    
            }
            case msgpack::type::MAP:
            {
                migraphx::value r;
                std::for_each(o.via.map.ptr, o.via.map.ptr+o.via.map.size, [&](auto&& p) {
                    r[p.key.as<std::string>()] = p.val.as<migraphx::value>();
                });
                v = r;
                break;    
            }
            case msgpack::type::EXT:
            {
                throw msgpack::type_error();    
            }
        }
        return o;
    }
};

template<>
struct pack<migraphx::value> {
    template <class Stream>
    void write(msgpack::packer<Stream>& o, const std::nullptr_t&) const {
        o.pack_nil();
    }
    template <class Stream, class T>
    void write(msgpack::packer<Stream>& o, const T& x) const {
        o.pack(x);
    }
    template <class Stream>
    void write(msgpack::packer<Stream>& o, const std::vector<migraphx::value>& v) const {
        if (v.empty())
        {
            o.pack_array(0);
            return;
        }
        if (not v.front().get_key().empty())
        {
            o.pack_map(v.size());
            for(auto&& x:v)
            {
                o.pack(x.get_key());
                o.pack(x.without_key());
            }
        }
        else
        {
            o.pack_array(v.size());
            for(auto&& x:v)
            {
                o.pack(x);
            }
        }
    }
    template <class Stream>
    packer<Stream>& operator()(msgpack::packer<Stream>& o, const migraphx::value& v) const {
        v.visit([&](auto&& x) {
            this->write(o, x);
        });
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
        buffer.insert(buffer.end(), b, b+n);
        return *this;
    }
};

std::vector<char> to_msgpack(const value& v)
{
    vector_stream vs;
    msgpack::pack(vs, v);
    return vs.buffer;
}
value from_msgpack(const std::vector<char>& buffer)
{
    msgpack::object_handle oh = msgpack::unpack(buffer.data(), buffer.size());
    return oh.get().as<value>();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
