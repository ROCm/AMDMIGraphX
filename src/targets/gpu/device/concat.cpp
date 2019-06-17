#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/concat.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

#if 0
argument concat(hipStream_t stream,
                const migraphx::shape&,
                std::vector<migraphx::argument> args_vec,
                std::vector<std::size_t> offsets_vec)
{
    static constexpr const std::size_t limit = 6;
    if(offsets_vec.size() > limit)
        MIGRAPHX_THROW("Too many arguments to concat");
    std::size_t nelements =
        std::max_element(args_vec.begin(),
                         std::prev(args_vec.end()),
                         by(std::less<>{}, [&](auto&& x) { return x.get_shape().elements(); }))
            ->get_shape()
            .elements();
    auto offsets = to_hip_vector<limit>(offsets_vec);
    hip_visit_all<limit + 1>(args_vec)([&](auto args) {
        auto output  = args.back();
        auto ninputs = args.size() - 1;
        gs_launch(stream, nelements)([=](auto i) {
            for(std::size_t j = 0; j < ninputs; j++)
            {
                auto&& arg = args[j];
                if(i >= arg.size())
                    continue;
                auto idx = output.get_shape().index(arg.get_shape().multi(i));
                output.data()[idx + offsets[j]] = arg.data()[i];
            }
        });
    });
    return args_vec.back();
}

#else

argument concat(hipStream_t stream,
                const migraphx::shape&,
                std::vector<migraphx::argument> args,
                std::vector<std::size_t> offsets)
{
    auto ninputs = args.size() - 1;
    for(std::size_t j = 0; j < ninputs; j++)
    {
        auto&& arg            = args[j];
        std::size_t nelements = arg.get_shape().elements();
        auto offset           = offsets[j];
        hip_visit_all(args.back(), arg)([&](auto output, auto input) {
            gs_launch(stream, nelements)([=](auto i) {
                auto idx                    = output.get_shape().index(input.get_shape().multi(i));
                output.data()[idx + offset] = input.data()[i];
            });
        });
    }
    return args.back();
}

// argument concat(hipStream_t stream,
//                 const migraphx::shape& output_shape,
//                 std::vector<migraphx::argument> args,
//                 std::vector<std::size_t> offsets)
// {
//     for(std::size_t l = 0; l < args.size() - 1; l++)
//     {
//         auto argl             = args[l];
//         std::size_t nelements = argl.get_shape().elements();
//         visit_all(args.back(), argl)([&](auto output, auto input) {
//             visit_tensor_size(output_shape.lens().size(), [&](auto ndim) {
//                 auto* outptr      = output.data() + offsets[l];
//                 const auto* inptr = input.data();
//                 hip_tensor_descriptor<ndim> desc_input(input.get_shape());
//                 hip_tensor_descriptor<ndim> desc_output(output.get_shape());
//                 gs_launch(stream, nelements)(
//                     [=](auto i) { outptr[desc_output.linear(desc_input.multi(i))] = inptr[i]; });
//             });
//         });
//     }
//     return args.back();
// }
#endif
} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
