#ifndef MIGRAPHX_GUARD_KERNELS_BUFFER_ADDRESS_HPP
#define MIGRAPHX_GUARD_KERNELS_BUFFER_ADDRESS_HPP

#include <migraphx/kernels/types.hpp>

#ifndef MIGRAPHX_HAS_BUFFER_ADDR
#define MIGRAPHX_HAS_BUFFER_ADDR 1
#endif

namespace migraphx {

#if MIGRAPHX_HAS_BUFFER_ADDR
template <typename T>
__device__ vec<int32_t, 4> make_wave_buffer_resource(T* p_wave, index_int bytes)
{
    union resource
    {
        // using ptr = const void*;
        vec<int32_t, 4> content;
        const void* address[2];
        int32_t chunk[4];
    };
    resource result;
    result.address[0] = p_wave;
    result.chunk[2] = bytes;
    result.chunk[3] = 0x00020000; // 0x31014000 for navi
    return result.content;
}

template<class T>
struct raw_buffer_load;

#define MIGRAPHX_BUFFER_ADDR_VISIT_TYPES(m) \
m(i8, int8_t) \
m(i16, int16_t) \
m(f16, half) \
m(f32, float)

#define MIGRAPHX_RAW_BUFFER_LOAD(llvmtype, ...) \
    __device__ __VA_ARGS__ \
    llvm_amdgcn_raw_buffer_load_ ## llvmtype(vec<int32_t, 4> srsrc, \
                                     int32_t voffset, \
                                     int32_t soffset, \
                                     int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load." #llvmtype); \
    template<> \
    struct raw_buffer_load<__VA_ARGS__> { \
        static __device__ __VA_ARGS__ apply(vec<int32_t, 4> srsrc, \
                                     int32_t voffset, \
                                     int32_t soffset, \
                                     int32_t glc_slc) { \
            return llvm_amdgcn_raw_buffer_load_ ## llvmtype(srsrc, voffset, soffset, glc_slc); \
        } \
    };

#define MIGRAPHX_RAW_BUFFER_LOAD_VEC(llvmtype, cpptype) \
    MIGRAPHX_RAW_BUFFER_LOAD(llvmtype, cpptype) \
    MIGRAPHX_RAW_BUFFER_LOAD(v2 ## llvmtype, vec<cpptype, 2>) \
    MIGRAPHX_RAW_BUFFER_LOAD(v4 ## llvmtype, vec<cpptype, 4>)

MIGRAPHX_BUFFER_ADDR_VISIT_TYPES(MIGRAPHX_RAW_BUFFER_LOAD_VEC)

#define MIGRAPHX_RAW_BUFFER_STORE(llvmtype, ...) \
    __device__ void \
    llvm_amdgcn_raw_buffer_store_ ## llvmtype(__VA_ARGS__ vdata, vec<int32_t, 4> srsrc, \
                                     int32_t voffset, \
                                     int32_t soffset, \
                                     int32_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store." #llvmtype); \
        __device__ void raw_buffer_store(__VA_ARGS__ vdata, vec<int32_t, 4> srsrc, \
                                     int32_t voffset, \
                                     int32_t soffset, \
                                     int32_t glc_slc) { \
            llvm_amdgcn_raw_buffer_store_ ## llvmtype(vdata, srsrc, voffset, soffset, glc_slc); \
        }

#define MIGRAPHX_RAW_BUFFER_STORE_VEC(llvmtype, cpptype) \
    MIGRAPHX_RAW_BUFFER_STORE(llvmtype, cpptype) \
    MIGRAPHX_RAW_BUFFER_STORE(v2 ## llvmtype, vec<cpptype, 2>) \
    MIGRAPHX_RAW_BUFFER_STORE(v4 ## llvmtype, vec<cpptype, 4>)

MIGRAPHX_BUFFER_ADDR_VISIT_TYPES(MIGRAPHX_RAW_BUFFER_STORE_VEC)

template<class T, index_int N>
struct raw_buffer_load<vec<T, N>>
{
    static __device__ vec<T, N> apply(vec<int32_t, 4> srsrc,
                                     int32_t voffset,
                                     int32_t soffset,
                                     int32_t glc_slc)
    {
        static_assert(N % 2 == 0, "Invalid vector size");
        union type
        {
            vec<T, N> data;
            vec<T, N/2> reg[2];
        };
        type result;
        auto offset = sizeof(T) * (N/2);
        result.reg[0] = raw_buffer_load<vec<T, N/2>>::apply(srsrc, voffset+offset, soffset, glc_slc);
        result.reg[1] = raw_buffer_load<vec<T, N/2>>::apply(srsrc, voffset, soffset, glc_slc);
        return result.data;
    }
};

template<class T, index_int N>
__device__ void raw_buffer_store(vec<T, N> vdata, vec<int32_t, 4> srsrc,
                                     int32_t voffset,
                                     int32_t soffset,
                                     int32_t glc_slc) {
    union type
    {
        vec<T, N> data;
        vec<T, N/2> reg[2];
    };
    type x;
    x.data = vdata;
    auto offset = sizeof(T) * (N/2);
    raw_buffer_store(x.reg[0], srsrc, voffset, soffset, glc_slc);
    raw_buffer_store(x.reg[1], srsrc, voffset+offset, soffset, glc_slc);
}

template<class T>
__device__ T buffer_load(const T* p, index_int offset, index_int size, address_space::global)
{
    auto resource = make_wave_buffer_resource(p, size * sizeof(T));
    return raw_buffer_load<T>::apply(resource, offset * sizeof(T), 0, 0);
}

template<class T>
__device__ void buffer_store(T data, T* p, index_int offset, index_int size, address_space::global)
{
    auto resource = make_wave_buffer_resource(p, size * sizeof(T));
    return raw_buffer_store(data, resource, offset * sizeof(T), 0, 0);
}

#endif

template<class T, class AddressSpace>
__device__ T buffer_load(const T* p, index_int offset, index_int, AddressSpace)
{
    return p[offset];
}

template<class T, class AddressSpace>
__device__ void buffer_store(T data, T* p, index_int offset, index_int, AddressSpace)
{
    p[offset] = data;
}


} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_BUFFER_ADDRESS_HPP
