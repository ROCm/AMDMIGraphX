// Gelu.CompilationInput.mlir
// Minimal DxGML fixture: GELU activation = x * 0.5 * (1 + erf(x / sqrt(2)))
// Input:  arg0 half[1,512,3000]
// Output: half[1,512,3000]
//
// Instruction order (matches test expectations):
//   [0] arg0
//   [1] sqrt2_const  (divisor)
//   [2] one_const    (addend)
//   [3] half_const   (multiplier 0.5)
//   [4] div(arg0, sqrt2_const)
//   [5] erf(div_result)
//   [6] add(erf_result, one_const)   -- note: test checks add before mul
//   [7] mul(arg0, add_result)
//   [8] mul(prev_mul, half_const)
//   [9] return

module {
  dxgml.module @gelu {
    dxgml.entry_point @forward(
        %arg0 : !dxgml.tensor<1x512x3000x!dxgml.float16>
    ) -> !dxgml.tensor<1x512x3000x!dxgml.float16> {
      %sqrt2_const = dxgml_op.constant(#dxgml.constant_resource<gelu.sqrt2 : !dxgml.tensor<1x512x3000x!dxgml.float16>>)
      %one_const   = dxgml_op.constant(#dxgml.constant_resource<gelu.one   : !dxgml.tensor<1x512x3000x!dxgml.float16>>)
      %half_const  = dxgml_op.constant(#dxgml.constant_resource<gelu.half  : !dxgml.tensor<1x512x3000x!dxgml.float16>>)
      %div    = dxgml_op.divide(%arg0, %sqrt2_const)  : (!dxgml.tensor<1x512x3000x!dxgml.float16>, !dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
      %erf    = dxgml_op.erf(%div)                    : (!dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
      %add    = dxgml_op.add(%erf, %one_const)        : (!dxgml.tensor<1x512x3000x!dxgml.float16>, !dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
      %mul1   = dxgml_op.multiply(%arg0, %add)        : (!dxgml.tensor<1x512x3000x!dxgml.float16>, !dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
      %mul2   = dxgml_op.multiply(%mul1, %half_const) : (!dxgml.tensor<1x512x3000x!dxgml.float16>, !dxgml.tensor<1x512x3000x!dxgml.float16>) -> !dxgml.tensor<1x512x3000x!dxgml.float16>
      dxgml.return %mul2 : !dxgml.tensor<1x512x3000x!dxgml.float16>
    }
  }
}
