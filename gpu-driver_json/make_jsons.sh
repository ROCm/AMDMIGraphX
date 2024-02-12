#!/bin/bash
hjson -j half_erf_gelu.hjson > half_erf_gelu.json
hjson -j half_sigmoid_gelu.hjson > half_sigmoid_gelu.json
hjson -j half_tanh_exp_gelu.hjson > half_tanh_exp_gelu.json
hjson -j half_tanh_pow_gelu.hjson > half_tanh_pow_gelu.json
hjson -j erf_gelu.hjson > erf_gelu.json
hjson -j sigmoid_gelu.hjson > sigmoid_gelu.json
hjson -j tanh_exp_gelu.hjson > tanh_exp_gelu.json
hjson -j tanh_pow_gelu.hjson > tanh_pow_gelu.json
