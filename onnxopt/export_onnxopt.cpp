#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <string>
#include "dqx_helper.h"
#include "wmc_utils.h"
#include <onnx/checker.h>
#include <onnxoptimizer/optimize.h>


#define FOR(i, range) for (auto i = decltype(range)(0); i < range; i++)

extern "C" {

WasmBuffer *create_exporter() {
  WasmBuffer *ctx;

  ctx = static_cast<WasmBuffer *>(malloc(sizeof(WasmBuffer)));
  ctx->output_buffer_size1 = 0;
  ctx->output_buffer_size2 = 0;
  ctx->output_buffer_size3 = 0;

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

unsigned char *get_buffer3(WasmBuffer *ctx) { return ctx->output_buffer3; }

size_t get_buffer_size3(WasmBuffer *ctx) { return ctx->output_buffer_size3; }

void add_initer_to_inputs(onnx::ModelProto &model) {
  std::vector<std::string> input_names;
  for (const auto &x : model.graph().input()) {
    input_names.push_back(x.name());
  }
  for (const auto &x : model.graph().initializer()) {
    if (std::find(input_names.begin(), input_names.end(), x.name()) ==
        input_names.end()) {
      auto *value_info = model.mutable_graph()->add_input();
      value_info->set_name(x.name());
      onnx::TypeProto *type = value_info->mutable_type();
      auto *tensor = type->mutable_tensor_type();
      tensor->set_elem_type(x.data_type());
      auto *shape = tensor->mutable_shape();
      for (const auto &dim : x.dims()) {
        onnx::TensorShapeProto::Dimension *new_dim = shape->add_dim();
        new_dim->set_dim_value(dim);
      }
    }
  }
}

bool onnxoptimize_export2(WasmBuffer *ctx, unsigned char *buf,
                         const size_t len) {
  try {
    onnx::ModelProto opt_model;
    {
      onnx::ModelProto model;
      bool s1 = model.ParseFromArray(buf, len);
      free(buf);
      if (!s1) {
        ctx->setBuffer3("parsing ONNX model fails");
        return false;
      }
      add_initer_to_inputs(model);
      opt_model = ONNX_NAMESPACE::optimization::OptimizeFixed(
          model,
          {"eliminate_deadend"});
    }
    onnx::checker::check_model(opt_model);
    auto byte_size = opt_model.ByteSizeLong();
    void *buf = malloc(byte_size);
    bool s2 = opt_model.SerializeToArray(buf, byte_size);
    if (!s2) {
      ctx->setBuffer3("serialing ONNX model fails");
      return false;
    }
    ctx->setBuffer1(buf, byte_size);
    std::cout << "ok!" << std::endl;
    std::cout << "byte_size: " << byte_size << std::endl;
    return true;
  } catch (onnx::checker::ValidationError &e) {
    ctx->setBuffer3("The optimized onnx model is broken.");
    return false;
  } catch (std::exception &e) {
    ctx->setBuffer3(e.what());
    return false;
  }
}

}

