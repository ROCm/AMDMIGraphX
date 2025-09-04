#####################################################################################
# Blaze Integration Example CMakeLists.txt
#####################################################################################

cmake_minimum_required(VERSION 3.15)
project(blaze_integration_example)

# Find MIGraphX
find_package(migraphx REQUIRED)

# Create the example executable
add_executable(blaze_integration_example blaze_integration_example.cpp)

# Link MIGraphX library
target_link_libraries(blaze_integration_example migraphx::migraphx)

# If Blaze is enabled, link Blaze library
if(MIGRAPHX_USE_BLAZE)
    find_package(blaze REQUIRED)
    target_link_libraries(blaze_integration_example blaze::blaze)
    target_compile_definitions(blaze_integration_example PRIVATE MIGRAPHX_USE_BLAZE=1)
else()
    target_compile_definitions(blaze_integration_example PRIVATE MIGRAPHX_USE_BLAZE=0)
endif()

# Set C++ standard
set_property(TARGET blaze_integration_example PROPERTY CXX_STANDARD 17)
