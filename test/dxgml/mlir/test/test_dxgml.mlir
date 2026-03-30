dxgml.module @test_module {
  dxgml.entry_point @test(%arg0: !dxgml.tensor<1x4x!dxgml.float16>) -> !dxgml.tensor<1x4x!dxgml.float16> {
    dxgml.return %arg0 : !dxgml.tensor<1x4x!dxgml.float16>
  }
}
