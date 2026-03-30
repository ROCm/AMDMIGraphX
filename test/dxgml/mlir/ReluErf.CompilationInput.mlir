dxgml.module {
  dxgml.entry_point @Relu_Erf(%arg0: !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
  {
    %0 = dxgml_op.relu(%arg0) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    %1 = dxgml_op.erf(%0) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    
    dxgml.return %1 : !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
  }
}

