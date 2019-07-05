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
  unsigned char *output_buffer1;
  unsigned char *output_buffer2;
  size_t output_buffer_size1;
  size_t output_buffer_size2;
};

extern "C" {

WasmBuffer *create_exporter() {
  WasmBuffer *ctx;

  ctx = static_cast<WasmBuffer *>(malloc(sizeof(WasmBuffer)));

  return ctx;
}

void free_exporter(WasmBuffer *ctx) {
  if (ctx != NULL) {
    if (ctx->output_buffer1 != NULL) {
      free(ctx->output_buffer1);
      ctx->output_buffer1 = NULL;
    }
    if (ctx->output_buffer2 != NULL) {
      free(ctx->output_buffer2);
      ctx->output_buffer2 = NULL;
    }
    free(ctx);
    ctx = NULL;
  }
}

unsigned char *get_buffer1(WasmBuffer *ctx) { return ctx->output_buffer1; }

size_t get_buffer_size1(WasmBuffer *ctx) { return ctx->output_buffer_size1; }

unsigned char *get_buffer2(WasmBuffer *ctx) { return ctx->output_buffer2; }

size_t get_buffer_size2(WasmBuffer *ctx) { return ctx->output_buffer_size2; }

void onnx2ncnn_export(WasmBuffer *ctx, unsigned char *buffer, size_t bufferlen) {
    PNT(bufferlen);
  std::string buf_str(reinterpret_cast<char *>(buffer), bufferlen);
  const auto res = onnx2ncnn(buf_str).value();
  const auto pv = res.first;
  const auto bv = res.second;
  PNT(pv.size(), bv.size());
  ctx->output_buffer1 = static_cast<unsigned char*>(malloc(pv.size()));
  memcpy(ctx->output_buffer1, pv.data(), pv.size());
  ctx->output_buffer_size1 = pv.size();
  ctx->output_buffer2 = static_cast<unsigned char*>(malloc(bv.size()));
  memcpy(ctx->output_buffer2, bv.data(), bv.size());
  ctx->output_buffer_size2 = bv.size();
}
}
