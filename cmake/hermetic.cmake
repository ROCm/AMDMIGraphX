#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
#
# Toolchain file for hermetic builds: CMake never reaches the network on its
# own. FetchContent does not download or update anything (its source dirs must
# already exist) and ExternalProject does not re-fetch (update) content it has
# already downloaded.
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/hermetic.cmake ...
#   rbuild prepare -d depend -t cmake/hermetic.cmake
#   tools/install_prereqs.sh --toolchain cmake/hermetic.cmake
#
# rbuild/cget include this file from the cget.cmake toolchain they write, so
# every dependency build and the MIGraphX build itself pick up these settings.

# Cache entries so option() in FetchContent.cmake keeps them under every CMP0077
# setting. An explicit -D on the command line still wins.
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL
    "Hermetic build: never download or update FetchContent sources")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Hermetic build: never update FetchContent sources")

# ExternalProject reads EP_UPDATE_DISCONNECTED as an inherited directory
# property, which chains up to the global scope from every directory.
set_property(GLOBAL PROPERTY EP_UPDATE_DISCONNECTED ON)
