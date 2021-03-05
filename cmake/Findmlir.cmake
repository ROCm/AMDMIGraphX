# This file allows users to call find_package(MLIR) and pick up our targets.

set(HAS_MLIR_MIOPEN_SUPPORT False)

if(DEFINED ENV{MLIR_SOURCE_PATH})
  set(MLIR_SOURCE_PATH   "$ENV{MLIR_SOURCE_PATH}")
  set(MLIR_BUILD_PATH    "$ENV{MLIR_BUILD_PATH}" )

  # For mlir_dialect
  set(mlir_INCLUDE_DIRS
    ${MLIR_SOURCE_PATH}/mlir/include
    ${MLIR_SOURCE_PATH}/mlir/tools/mlir-miopen-lib
    ${MLIR_SOURCE_PATH}/llvm/include
    ${MLIR_BUILD_PATH}/include
    ${MLIR_BUILD_PATH}/tools/mlir/include
    )
  set(mlir_LIBRARY_DIR ${MLIR_BUILD_PATH}/lib)
  set(mlir_LIBRARIES ${mlir_LIBRARY_DIR}/libMLIRMIOpenThin.so)

  if(EXISTS ${mlir_LIBRARIES})
    set(HAS_MLIR_MIOPEN_SUPPORT True)
  endif()

endif()
