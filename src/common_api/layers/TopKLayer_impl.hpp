#ifndef TOPKLAYER_IMPL_HPP
#define TOPKLAYER_IMPL_HPP

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class TopKLayer_impl : public ITopKLayer, public apiv::VTopKLayer, virtual public Layer_impl
    {
    public:
        TopKLayer_impl() noexcept;
        TopKLayer_impl(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~TopKLayer_impl() override;

        // public API
        void setOperation(TopKOperation op) noexcept override;
        TopKOperation getOperation() const noexcept override;
        void setK(int32_t k) noexcept override;
        int32_t getK() const noexcept override;
        void setReduceAxes(uint32_t reduceAxes) noexcept override;
        uint32_t getReduceAxes() const noexcept override;

        void build() noexcept override;

    private:
        TopKOperation mOp{TopKOperation::kMAX};
        int32_t mK{1};
        uint32_t mReduceAxes{0};
    };
} // namespace nvinfer1 

#endif // TOPKLAYER_IMPL_HPP
