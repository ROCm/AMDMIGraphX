#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
import migraphx


def test_create_shape():
    s = migraphx.shape(lens=[1, 64, 3, 3])
    assert s.standard()
    assert s.packed()
    assert s.lens() == [1, 64, 3, 3]


def test_create_shape_broadcast():
    s = migraphx.shape(lens=[1, 64, 3, 3], strides=[0, 1, 0, 0])
    assert s.broadcasted()
    assert s.lens() == [1, 64, 3, 3]
    assert s.strides() == [0, 1, 0, 0]


def test_create_shape_type():
    s = migraphx.shape(type='int64_t')
    assert s.type_string() == 'int64_type'
    assert s.type_size() == 8
    s = migraphx.shape(type='uint8_t')
    assert s.type_string() == "uint8_type"
    assert s.type_size() == 1
    s = migraphx.shape(type='float')
    assert s.type_size() == 4


if __name__ == "__main__":
    test_create_shape()
    test_create_shape_broadcast()
    test_create_shape_type()
