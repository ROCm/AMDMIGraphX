#ifndef ELEMENT_WISE_LAYER_IMPL_HPP
#define ELEMENT_WISE_LAYER_IMPL_HPP

#include "Layer_impl.hpp"

namespace nvinfer1
{
    class ElementWiseLayer_impl : public IElementWiseLayer, public apiv::VElementWiseLayer, virtual public Layer_impl
    {
    public:
        ElementWiseLayer_impl() noexcept;
        ElementWiseLayer_impl(ITensor& input1, ITensor& input2, ElementWiseOperation op, const std::shared_ptr<migraphx::program>& program) noexcept;
        ~ElementWiseLayer_impl() override;

        // public API
        void setOperation(ElementWiseOperation op) noexcept override;
        ElementWiseOperation getOperation() const noexcept override;

        void build() noexcept override;

    private:
        ElementWiseOperation mOp;
    };
}

#endif // ELEMENT_WISE_LAYER_IMPL_HPP
