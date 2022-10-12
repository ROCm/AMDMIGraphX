#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MKLDNN::mkldnn" for configuration "Release"
set_property(TARGET MKLDNN::mkldnn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MKLDNN::mkldnn PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libdnnl.so.1.7"
  IMPORTED_SONAME_RELEASE "libdnnl.so.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS MKLDNN::mkldnn )
list(APPEND _IMPORT_CHECK_FILES_FOR_MKLDNN::mkldnn "${_IMPORT_PREFIX}/lib/libdnnl.so.1.7" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
