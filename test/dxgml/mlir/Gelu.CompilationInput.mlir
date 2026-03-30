dxgml.module {
        dxgml.entry_point @main_graph(%arg0: !dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
        {
            %sqrt2_const = dxgml_op.constant(#dxgml.constant_resource<_ : !dxgml.tensor<1x512x3000x!dxgml.float16>>)
            %one_const = dxgml_op.constant(#dxgml.constant_resource<__1 : !dxgml.tensor<1x512x3000x!dxgml.float16>>)
            %half_const = dxgml_op.constant(#dxgml.constant_resource<__2 : !dxgml.tensor<1x512x3000x!dxgml.float16>>)
            %div = dxgml_op.divide(%arg0, %sqrt2_const) : (!dxgml.tensor<1x512x3000x!dxgml.float16>, !dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
            %erf = dxgml_op.erf(%div) : (!dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
            %add = dxgml_op.add(%erf, %one_const) : (!dxgml.tensor<1x512x3000x!dxgml.float16>, !dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
            %mul_input = dxgml_op.multiply(%arg0, %add) : (!dxgml.tensor<1x512x3000x!dxgml.float16>, !dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
            %mul_half = dxgml_op.multiply(%mul_input, %half_const) : (!dxgml.tensor<1x512x3000x!dxgml.float16>, !dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
                        dxgml.return %mul_half : !dxgml.tensor<1x512x3000x!dxgml.float16>
        }
}
{-#
dialect_resources: {
    dxgml: {
        _: "0x08000000A83D",
        __1: "0x08000000003C",
        __2: "0x080000000038"
    }
}
#-}