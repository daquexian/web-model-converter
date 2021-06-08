#include <onnxruntime/cmake/external/onnx/onnx/onnx_pb.h>
#include <onnxruntime/cmake/external/onnx/onnx/optimizer/optimize.h>
#include <onnxruntime/cmake/external/onnx/onnx/shape_inference/implementation.h>
#include <onnxruntime/cmake/external/onnx/onnx/checker.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include <onnxruntime/test.h>

#include "onnx2tnn.h"

#include "dqx_helper.h"
#include "tengine/core/include/tengine_c_api.h"

#define FOR(i, range) for (auto i = decltype(range)(0); i < range; i++)

struct WasmBuffer {
  unsigned char *output_buffer1 = nullptr;
  unsigned char *output_buffer2 = nullptr;
  unsigned char *output_buffer3 = nullptr;
  size_t output_buffer_size1 = 0;
  size_t output_buffer_size2 = 0;
  size_t output_buffer_size3 = 0;

  void freeBuffers() {
    freeBuffer1();
    freeBuffer2();
    freeBuffer3();
  }
  void freeBuffer1() {
    if (output_buffer1 != nullptr) {
      free(output_buffer1);
      output_buffer1 = nullptr;
      output_buffer_size1 = 0;
    }
  }
  void freeBuffer2() {
    if (output_buffer2 != nullptr) {
      free(output_buffer2);
      output_buffer2 = nullptr;
      output_buffer_size2 = 0;
    }
  }
  void freeBuffer3() {
    if (output_buffer3 != nullptr) {
      free(output_buffer3);
      output_buffer3 = nullptr;
      output_buffer_size3 = 0;
    }
  }
  void setBuffer1(Buffer buf) { setBuffer1(buf.first, buf.second); }
  void setBuffer1(void *buf, const size_t buflen) {
    // we own the buf
    output_buffer1 = static_cast<unsigned char *>(buf);
    output_buffer_size1 = buflen;
  }
  void setBuffer1(const std::string &str) {
    output_buffer1 = static_cast<unsigned char *>(malloc(str.size()));
    memcpy(output_buffer1, str.c_str(), str.size());
    output_buffer_size1 = str.size();
  }
  void setBuffer2(Buffer buf) { setBuffer2(buf.first, buf.second); }
  void setBuffer2(void *buf, const size_t buflen) {
    // we own the buf
    output_buffer2 = static_cast<unsigned char *>(buf);
    output_buffer_size2 = buflen;
  }
  void setBuffer2(const std::string &str) {
    output_buffer2 = static_cast<unsigned char *>(malloc(str.size()));
    memcpy(output_buffer2, str.c_str(), str.size());
    output_buffer_size2 = str.size();
  }
  void setBuffer3(const std::string &str) {
    output_buffer3 = static_cast<unsigned char *>(malloc(str.size()));
    memcpy(output_buffer3, str.c_str(), str.size());
    output_buffer_size3 = str.size();
  }
};

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

// ------ onnx

// for x in model.graph.initializer:
//     input_names = [x.name for x in model.graph.input]
//     if x.name not in input_names:
//         shape = onnx.TensorShapeProto()
//         for dim in x.dims:
//             shape.dim.extend([onnx.TensorShapeProto.Dimension(dim_value=dim)])
//         model.graph.input.extend(
//             [onnx.ValueInfoProto(name=x.name,
//                                  type=onnx.TypeProto(tensor_type=onnx.TypeProto.Tensor(elem_type=x.data_type,
//                                                                                        shape=shape)))])
// return model

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

int check_static_input_size_export(WasmBuffer *ctx, unsigned char *buf,
                                   const size_t len) {
  try {
    onnx::ModelProto model;
    bool s1 = model.ParseFromArray(buf, len);
    if (!s1) {
      ctx->setBuffer3("parsing ONNX model fails");
      return -1;
    }
    for (const auto &x : model.graph().input()) {
      for (const auto &initer : model.graph().initializer()) {
        if (x.name() == initer.name()) {
          continue;
        }
      }
      if (!CheckStaticInputShape(model, x.name())) {
        if (GetInputNames(model).size() > 1) {
          ctx->setBuffer3("Multiple inputs and dynamic input size");
          return -2;
        } else {
          return 1;
        }
      }
    }
    return 2;
  } catch (std::exception &e) {
    ctx->setBuffer3(e.what());
    return -1;
  }
}

bool onnxsimplify_export(WasmBuffer *ctx, unsigned char *buf, const size_t len,
                         const bool optimize, const int32_t *input_shape,
                         const size_t input_shape_len) {
  try {
    onnx::ModelProto opt_model;
    bool check;
    {
      onnx::ModelProto model;
      bool s1 = model.ParseFromArray(buf, len);
      free(buf);
      if (!s1) {
        ctx->setBuffer3("parsing ONNX model fails");
        return false;
      }
      add_initer_to_inputs(model);
      MyTensorShapeMap input_map;
      const std::string input_name = GetInputNames(model)[0];
      if (input_shape_len > 0) {
        MyTensorShape shape;
        FOR(i, input_shape_len) { shape.push_back(input_shape[i]); }
        for (const auto &x : shape) {
          std::cout << __LINE__ << " " << x << std::endl;
        }
        input_map[input_name] = shape;
      }
      std::cout << __LINE__ << " " << input_map.size() << std::endl;
      if (input_map.size() > 0) {
        std::cout << __LINE__ << " " << input_map[input_name].size()
                  << std::endl;
        for (const auto &x : input_map[input_name]) {
          std::cout << __LINE__ << " " << x << std::endl;
        }
      }

      std::cout << "simplify begin" << std::endl;
      opt_model = Simplify(model, optimize, input_map);
      std::cout << "simplify end" << std::endl;
      try {
        check = Check(opt_model, model, input_map);
      } catch (const std::exception &e) {
        std::cout << "check exception: " << e.what() << std::endl;
        check = false;
      }
      std::cout << "check end" << std::endl;
      if (check) {
        std::cout << "check ok" << std::endl;
      } else {
        std::cout << "check failed" << std::endl;
      }
    }
    auto byte_size = opt_model.ByteSizeLong();
    void *buf = malloc(byte_size);
    bool s2 = opt_model.SerializeToArray(buf, byte_size);
    if (!s2) {
      ctx->setBuffer3("serialing ONNX model fails");
      return false;
    }
    ctx->setBuffer1(buf, byte_size);
    if (!check) {
      ctx->setBuffer3(
          "The result is different after simplifying, sometimes it is "
          "something wrong in onnx simplifier, but sometimes it is just "
          "numerical error, please be careful to use the simplified model.");
    }
    return true;
  } catch (std::exception &e) {
    ctx->setBuffer3(e.what());
    return false;
  }
}

bool onnx2tnn_export(WasmBuffer *ctx, void *buffer, const size_t bufferlen) {
  std::cout << bufferlen << std::endl;
  std::cout << __LINE__ << std::endl;
  Onnx2TNN converter(&buffer, bufferlen);
  const auto expected_res = converter.Convert();
  std::cout << __LINE__ << std::endl;
  if (!expected_res) {
    std::cout << expected_res.error() << std::endl;
    ctx->setBuffer3(expected_res.error());
    return false;
  }
  std::cout << __LINE__ << std::endl;
  const auto res = expected_res.value();
  std::cout << __LINE__ << std::endl;
  const auto pv = std::get<0>(res);
  const auto str_file_model = std::get<1>(res);
  size_t model_size = str_file_model.size();
  void* ptr_data = (void*) str_file_model.data();
  auto bv = std::make_pair(ptr_data, model_size);
  const auto error_msg = std::get<2>(res);
  PNT(pv.second, bv.second, error_msg);
  ctx->setBuffer1(pv);
  ctx->setBuffer2(bv);
  ctx->setBuffer3(error_msg);

  return true;
}

}
