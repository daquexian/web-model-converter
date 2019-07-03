#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include <onnx/onnx_pb.h>
#include <onnx/optimizer/optimize.h>

#include "dqx_helper.h"

struct WasmBuffer {
  unsigned char *output_buffer;
  size_t output_buffer_size;
};

WasmBuffer *create_exporter(unsigned char *input_buffer,
                            size_t input_buffer_size) {
  WasmBuffer *ctx;

  ctx = static_cast<WasmBuffer *>(malloc(sizeof(WasmBuffer)));

  return ctx;
}

void free_exporter(WasmBuffer *ctx) {
  if (ctx != NULL) {
    if (ctx->output_buffer != NULL) {
      free(ctx->output_buffer);
      ctx->output_buffer = NULL;
    }
    free(ctx);
    ctx = NULL;
  }
}

unsigned char *get_buffer(WasmBuffer *ctx) { return ctx->output_buffer; }

size_t get_buffer_size(WasmBuffer *ctx) { return ctx->output_buffer_size; }

extern "C" {

void list(WasmBuffer *ctx, unsigned char *buffer, size_t bufferlen) {
  ONNX_NAMESPACE::ModelProto model_proto;
  std::string buf_str(reinterpret_cast<char *>(buffer), bufferlen);
  PNT(model_proto.ParseFromString(buf_str));

  for (const auto &tensor : model_proto.graph().initializer()) {
    PNT(tensor.name());
  }
}
}
