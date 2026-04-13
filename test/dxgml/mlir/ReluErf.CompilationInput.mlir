// ReluErf.CompilationInput.mlir
// Minimal DxGML fixture: relu -> erf
// Input:  arg0 half[1,32,1080,1920]
// Output: half[1,32,1080,1920]

module {
  dxgml.module @relu_erf {
    dxgml.entry_point @forward(
        %arg0 : !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    ) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16> {
      %relu = dxgml_op.relu(%arg0) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
      %erf  = dxgml_op.erf(%relu)  : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
      dxgml.return %erf : !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    }
  }
}
