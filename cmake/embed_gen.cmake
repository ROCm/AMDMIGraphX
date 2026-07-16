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

# Turns a single data file into a C++ source with a byte array plus its length.
# Invoked at build time (one process per file) so embedding runs in parallel and
# only reruns when an input changes. A single linear regex does the hex->array
# conversion, avoiding any per-byte string copying.

file(READ "${EMBED_INPUT}" HEX_STRING HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "static_cast<char>(0x\\1)," ARRAY_VALUES "${HEX_STRING}")

file(WRITE "${EMBED_OUTPUT}" "\
#include <cstddef>
extern const char _binary_${EMBED_SYMBOL}_start[] = { ${ARRAY_VALUES} };
extern const size_t _binary_${EMBED_SYMBOL}_length = sizeof(_binary_${EMBED_SYMBOL}_start);
")
