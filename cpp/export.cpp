#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include <onnx/onnx_pb.h>
#include <onnx/optimizer/optimize.h>

#include "dqx_helper.h"

#include "caffe2ncnn.h"
#include "onnx2ncnn.h"

struct WasmBuffer {
  unsigned char *output_buffer1 = nullptr;
  unsigned char *output_buffer2 = nullptr;
  size_t output_buffer_size1 = 0;
  size_t output_buffer_size2 = 0;

  void freeBuffers() {
    freeBuffer1();
    freeBuffer2();
  }
  void freeBuffer1() {
    if (output_buffer1 != nullptr) {
      free(output_buffer1);
      output_buffer1 = nullptr;
    }
  }
  void freeBuffer2() {
    if (output_buffer2 != nullptr) {
      free(output_buffer2);
      output_buffer2 = nullptr;
    }
  }
  void setBuffer1(const std::vector<char> &vec) {
    output_buffer1 = static_cast<unsigned char *>(malloc(vec.size()));
    memcpy(output_buffer1, vec.data(), vec.size());
    output_buffer_size1 = vec.size();
  }
  void setBuffer2(const std::vector<char> &vec) {
    output_buffer2 = static_cast<unsigned char *>(malloc(vec.size()));
    memcpy(output_buffer2, vec.data(), vec.size());
    output_buffer_size2 = vec.size();
  }
  void setBuffer1(const std::string &str) {
    output_buffer1 = static_cast<unsigned char *>(malloc(str.size()));
    memcpy(output_buffer1, str.c_str(), str.size());
    output_buffer_size1 = str.size();
  }
  void setBuffer2(const std::string &str) {
    output_buffer2 = static_cast<unsigned char *>(malloc(str.size()));
    memcpy(output_buffer2, str.c_str(), str.size());
    output_buffer_size2 = str.size();
  }
};

extern "C" {

WasmBuffer *create_exporter() {
  WasmBuffer *ctx;

  ctx = static_cast<WasmBuffer *>(malloc(sizeof(WasmBuffer)));

  return ctx;
}

void free_exporter(WasmBuffer *ctx) {
  if (ctx != NULL) {
    ctx->freeBuffers();
    free(ctx);
    ctx = NULL;
  }
}

unsigned char *get_buffer1(WasmBuffer *ctx) { return ctx->output_buffer1; }

size_t get_buffer_size1(WasmBuffer *ctx) { return ctx->output_buffer_size1; }

unsigned char *get_buffer2(WasmBuffer *ctx) { return ctx->output_buffer2; }

size_t get_buffer_size2(WasmBuffer *ctx) { return ctx->output_buffer_size2; }

bool onnx2ncnn_export(WasmBuffer *ctx, const unsigned char *buffer,
                      const size_t bufferlen) {
  std::string buf_str(reinterpret_cast<const char *>(buffer), bufferlen);
  const auto expected_res = onnx2ncnn(buf_str);
  if (!expected_res) {
    std::cout << expected_res.error() << std::endl;
    ctx->setBuffer1(expected_res.error());
    return false;
  }
  const auto res = expected_res.value();
  const auto pv = res.first;
  const auto bv = res.second;
  PNT(pv.size(), bv.size());
  ctx->setBuffer1(pv);
  ctx->setBuffer2(bv);

  return true;
}

bool caffe2ncnn_export(WasmBuffer *ctx, const unsigned char *prototxt_buffer,
                       const size_t prototxt_bufferlen,
                       const unsigned char *caffemodel_buffer,
                       const size_t caffemodel_bufferlen) {
  const std::string prototxt_str(
      reinterpret_cast<const char *>(prototxt_buffer), prototxt_bufferlen);
  const std::string caffemodel_str(
      reinterpret_cast<const char *>(caffemodel_buffer), caffemodel_bufferlen);
  const auto expected_res = caffe2ncnn(prototxt_str, caffemodel_str);
  if (!expected_res) {
    std::cout << expected_res.error() << std::endl;
    ctx->setBuffer1(expected_res.error());
    return false;
  }
  const auto res = expected_res.value();
  const auto pv = res.first;
  const auto bv = res.second;
  PNT(pv.size(), bv.size());
  ctx->setBuffer1(pv);
  ctx->setBuffer2(bv);
  return true;
}
}
