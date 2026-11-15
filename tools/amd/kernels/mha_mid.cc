//===- mha_mid.cc - IRON's attention kernels with exp2_mid.h's exponential -*- C++ -*-===//
//
// IRON's linalg/mha.cc compiled after exp2_mid.h, which takes the softmax's exp2_bf16 over
// (tools/amd/iron_mha.py, ImageMHA's exp "mid").
//===----------------------------------------------------------------------===//
#include "exp2_mid.h"

#include "linalg/mha.cc"
