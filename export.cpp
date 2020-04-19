#include <onnx/onnx_pb.h>
#include <onnx/optimizer/optimize.h>
#include <onnx/onnx/shape_inference/implementation.h>

#include <MNN/tools/converter/include/PostConverter.hpp>
#include <MNN/tools/converter/include/caffeConverter.hpp>
#include <MNN/tools/converter/include/onnxConverter.hpp>
#include <MNN/tools/converter/include/tensorflowConverter.hpp>
#include <MNN/tools/converter/include/writeFb.hpp>

#include <tengine/core/include/tengine_c_api.h>
#include <tengine/core/include/tengine_c_helper.hpp>
#include <tengine/core/include/exec_context.hpp>
#include <tengine/core/include/graph_executor.hpp>
#include <tengine/tools/plugin/serializer/onnx/onnx_serializer.hpp>
#include <tengine/tools/plugin/serializer/caffe/caffe_serializer.hpp>
#include <tengine/serializer/include/tm_serializer.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "caffe2ncnn.h"
#include "ncnn/tools/mxnet/mxnet2ncnn.h"
#include "ncnn/tools/ncnnoptimize.h"
#include "dqx_helper.h"
#include "onnx2ncnn.h"
#include "tengine/core/include/tengine_c_api.h"

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

bool onnx2ncnn_export(WasmBuffer *ctx, const unsigned char *buffer,
                      const size_t bufferlen) {
  std::cout << bufferlen << std::endl;
  std::string buf_str(reinterpret_cast<const char *>(buffer), bufferlen);
  std::cout << __LINE__ << std::endl;
  const auto expected_res = onnx2ncnn(buf_str);
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
  const auto bv = std::get<1>(res);
  const auto error_msg = std::get<2>(res);
  PNT(pv.size(), bv.size(), error_msg);
  ctx->setBuffer1(pv);
  ctx->setBuffer2(bv);
  ctx->setBuffer3(error_msg);

  return true;
}

bool mxnet2ncnn_export(WasmBuffer *ctx, const unsigned char *nodes_buffer,
                       const size_t nodes_bufferlen,
                       const unsigned char *params_buffer,
                       const size_t params_bufferlen) {
  const std::string nodes_str(
      reinterpret_cast<const char *>(nodes_buffer), nodes_bufferlen);
  const std::string params_str(
      reinterpret_cast<const char *>(params_buffer), params_bufferlen);
  const auto expected_res = mxnet2ncnn(nodes_str, params_str);
  if (!expected_res) {
    std::cout << expected_res.error() << std::endl;
    ctx->setBuffer3(expected_res.error());
    return false;
  }
  const auto res = expected_res.value();
  const auto pv = std::get<0>(res);
  const auto bv = std::get<1>(res);
  const auto error_msg = std::get<2>(res);
  PNT(pv.size(), bv.size(), error_msg);
  ctx->setBuffer1(pv);
  ctx->setBuffer2(bv);
  ctx->setBuffer3(error_msg);
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
    ctx->setBuffer3(expected_res.error());
    return false;
  }
  const auto res = expected_res.value();
  const auto pv = std::get<0>(res);
  const auto bv = std::get<1>(res);
  const auto error_msg = std::get<2>(res);
  PNT(pv.size(), bv.size(), error_msg);
  ctx->setBuffer1(pv);
  ctx->setBuffer2(bv);
  ctx->setBuffer3(error_msg);
  return true;
}

bool caffe2mnn_export(WasmBuffer *ctx, const unsigned char *prototxt_buffer,
                      const size_t prototxt_bufferlen,
                      const unsigned char *caffemodel_buffer,
                      const size_t caffemodel_bufferlen) {
  try {
    const std::string prototxt_str(
        reinterpret_cast<const char *>(prototxt_buffer), prototxt_bufferlen);
    const std::string caffemodel_str(
        reinterpret_cast<const char *>(caffemodel_buffer),
        caffemodel_bufferlen);
    std::unique_ptr<MNN::NetT> netT =
        std::unique_ptr<MNN::NetT>(new MNN::NetT());
    const auto retcode = caffe2MNNNet(prototxt_str, caffemodel_str, "", netT);
    if (retcode != 0) {
      ctx->setBuffer3("Unknown problem");
      return false;
    }
    bool forTraining = false;
    netT = optimizeNet(netT, forTraining);
    const auto res = writeFb(netT, false, false);

    PNT(res.size());
    ctx->setBuffer1(res);
    return true;
  } catch (std::exception &e) {
    ctx->setBuffer3(e.what());
    return false;
  }
}

bool onnx2mnn_export(WasmBuffer *ctx, const unsigned char *buffer,
                     const size_t bufferlen) {
  try {
    std::string buf_str(reinterpret_cast<const char *>(buffer), bufferlen);
    std::unique_ptr<MNN::NetT> netT =
        std::unique_ptr<MNN::NetT>(new MNN::NetT());
    const auto retcode = onnx2MNNNet(buf_str, "", netT);
    if (retcode != 0) {
      ctx->setBuffer3("Unknown problem");
      return false;
    }
    bool forTraining = false;
    netT = optimizeNet(netT, forTraining);
    const auto res = writeFb(netT, false, false);

    PNT(res.size());
    ctx->setBuffer1(res);
    return true;
  } catch (std::exception &e) {
    ctx->setBuffer3(e.what());
    return false;
  }
}

bool tf2mnn_export(WasmBuffer *ctx, const unsigned char *buffer,
                   const size_t bufferlen) {
  try {
    std::string buf_str(reinterpret_cast<const char *>(buffer), bufferlen);
    std::unique_ptr<MNN::NetT> netT =
        std::unique_ptr<MNN::NetT>(new MNN::NetT());
    const auto retcode = tensorflow2MNNNet(buf_str, "", netT);
    if (retcode != 0) {
      ctx->setBuffer3("Unknown problem");
      return false;
    }
    bool forTraining = false;
    netT = optimizeNet(netT, forTraining);
    const auto res = writeFb(netT, false, false);

    PNT(res.size());
    ctx->setBuffer1(res);
    return true;
  } catch (std::exception &e) {
    ctx->setBuffer3(e.what());
    return false;
  }
}

// ------ ncnn

bool ncnnoptimize_export(WasmBuffer *ctx, 
        const unsigned char *param_buf, const size_t param_len,
        const unsigned char *bin_buf, const size_t bin_len,
        bool fp16) {
  const auto expected_res = ncnnoptimize(param_buf, bin_buf, fp16 ? 65536 : 0);
  if (!expected_res) {
    std::cout << expected_res.error() << std::endl;
    ctx->setBuffer3(expected_res.error());
    return false;
  }
  const auto res = expected_res.value();
  const auto pv = std::get<0>(res);
  const auto bv = std::get<1>(res);
  PNT(pv.size(), bv.size());
  ctx->setBuffer1(pv);
  ctx->setBuffer2(bv);
  // FIXME: set buf3
  return true;
}

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
        if (std::find(input_names.begin(), input_names.end(), x.name()) == input_names.end()) {
            auto *value_info = model.mutable_graph()->add_input();
            value_info->set_name(x.name());
            onnx::TypeProto *type= value_info->mutable_type();
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

bool onnxoptimize_export(WasmBuffer *ctx,
        const unsigned char *buf, const size_t len
        ) {
    try {
    onnx::ModelProto model;
    bool s1 = model.ParseFromString(std::string(reinterpret_cast<const char *>(buf), len));
    if (!s1) {
        ctx->setBuffer3("parsing ONNX model fails");
        return false;
    }
    add_initer_to_inputs(model);
    const auto opt_model = ONNX_NAMESPACE::optimization::OptimizeFixed(model, {"eliminate_deadend", "eliminate_identity", "eliminate_nop_dropout",
                                            "eliminate_nop_monotone_argmax", "eliminate_nop_pad",
                                            "extract_constant_to_initializer", "eliminate_unused_initializer",
                                            "eliminate_nop_transpose", "fuse_add_bias_into_conv", 
                                            "fuse_consecutive_concats",
                                            "fuse_consecutive_log_softmax",
                                            "fuse_consecutive_reduce_unsqueeze", "fuse_consecutive_squeezes",
                                            "fuse_consecutive_transposes", "fuse_matmul_add_bias_into_gemm",
                                            "fuse_pad_into_conv", "fuse_transpose_into_gemm",
                                            "fuse_bn_into_conv"
                                            });
    std::string opt_str;
    bool s2 = opt_model.SerializeToString(&opt_str);
    if (!s2) {
        ctx->setBuffer3("serialing ONNX model fails");
        return false;
    }
    ctx->setBuffer1(opt_str);
    return true;
    } catch (std::exception &e) {
        ctx->setBuffer3(e.what());
        return false;
    }
}

bool onnx_shape_infer_export(WasmBuffer *ctx,
        const unsigned char *buf, const size_t len
        ) {
    try {
    onnx::ModelProto model;
    bool s1 = model.ParseFromString(std::string(reinterpret_cast<const char *>(buf), len));
    if (!s1) {
        ctx->setBuffer3("parsing ONNX model fails");
        return false;
    }
    ONNX_NAMESPACE::shape_inference::InferShapes(model);
    const auto &shaped_model = model;
    std::string shaped_str;
    bool s2 = shaped_model.SerializeToString(&shaped_str);
    if (!s2) {
        ctx->setBuffer3("serialing ONNX model fails");
        return false;
    }
    ctx->setBuffer1(shaped_str);
    return true;
    } catch (std::exception &e) {
        ctx->setBuffer3(e.what());
        return false;
    }
}
// ------ tengine

// 这么写和普通的头文件写法没有区别，我只是懒
extern "C" int onnx_plugin_init(void);
extern "C" int caffe_plugin_init(void);

std::string log_output;
void log_func(const char *s) {
    log_output += s;
}

bool tengine_converter_inited = false;

struct PointerDeleter
{
    void operator()(void *p) { 
        std::cout << "delete!" << std::endl;
        free(p); 
    }
};

bool onnx2tengine_export(WasmBuffer *ctx, const unsigned char *buffer,
                         const size_t bufferlen) {
  using namespace TEngine;
  try {
    log_output = "";
    SET_LOG_OUTPUT(&log_func);
    if (!tengine_converter_inited) {
        TEngineConfig::Set("exec.engine", "generic", true);
        InitPluginForConverter();
        onnx_plugin_init();
        tengine_converter_inited = true;
    }

    const std::string model_name = "test1";
    const std::string graph_name = "test2";
    SerializerPtr tmp;

    if(!SerializerManager::SafeGet("onnx", tmp)) {
        ctx->setBuffer1("onnx serializer is not registered");
        return false;
    }
    auto *exec_context = (ExecContext *)create_context(model_name.c_str(), 0);
    std::cout << __LINE__ << std::endl;
    graph_t graph = create_graph(exec_context, "onnx:m", reinterpret_cast<const char *>(buffer), bufferlen);
    std::cout << __LINE__ << std::endl;
    if (!graph) {
        ctx->setBuffer3("Error: " + log_output);
        return false;
    }
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);
    Graph* g = executor->GetOptimizedGraph();
    std::cout << __LINE__ << std::endl;
    std::vector<void*> addr_list;
    std::vector<int> size_list;
    TmSerializer saver;
    bool save_res = saver.SaveModel(addr_list, size_list, g);
    std::unique_ptr<char, PointerDeleter> addr_deleter{static_cast<char*>(addr_list[0])};
    std::cout << __LINE__ << std::endl;
    char *tmp2 = static_cast<char*>(addr_list[0]);

    std::string res(tmp2, size_list[0]);
    std::cout << __LINE__ << std::endl;
    ctx->setBuffer1(res);
    destroy_graph(graph);
    std::cout << __LINE__ << std::endl;
    release_tengine();
    std::cout << __LINE__ << std::endl;
    return true;
  } catch (std::exception &e) {
    ctx->setBuffer3(e.what());
    return false;
  }
}

bool caffe2tengine_export(WasmBuffer *ctx, 
        const unsigned char *buffer1, 
        const size_t bufferlen1,
        const unsigned char *buffer2, 
        const size_t bufferlen2
        ) {
  using namespace TEngine;
  try {
    log_output = "";
    SET_LOG_OUTPUT(&log_func);
    if (!tengine_converter_inited) {
        TEngineConfig::Set("exec.engine", "generic", true);
        InitPluginForConverter();
        onnx_plugin_init();
        caffe_plugin_init();
        tengine_converter_inited = true;
    }

    const std::string model_name = "test1";
    const std::string graph_name = "test2";
    SerializerPtr tmp;

    if(!SerializerManager::SafeGet("caffe", tmp)) {
        ctx->setBuffer3("caffe serializer is not registered");
        return false;
    }
    auto *exec_context = (ExecContext *)create_context(model_name.c_str(), 0);
    std::cout << __LINE__ << std::endl;
    graph_t graph = create_graph(exec_context, "caffe:m", reinterpret_cast<const char *>(buffer1), bufferlen1, reinterpret_cast<const char *>(buffer2), bufferlen2);
    std::cout << __LINE__ << std::endl;
    if (!graph) {
        ctx->setBuffer3("Error: " + log_output);
        return false;
    }
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);
    Graph* g = executor->GetOptimizedGraph();
    std::cout << __LINE__ << std::endl;
    std::vector<void*> addr_list;
    std::vector<int> size_list;
    TmSerializer saver;
    bool save_res = saver.SaveModel(addr_list, size_list, g);
    std::unique_ptr<char, PointerDeleter> addr_deleter{static_cast<char*>(addr_list[0])};
    std::cout << __LINE__ << std::endl;
    char *tmp2 = static_cast<char*>(addr_list[0]);

    std::string res(tmp2, size_list[0]);
    std::cout << __LINE__ << std::endl;
    ctx->setBuffer1(res);
    destroy_graph(graph);
    std::cout << __LINE__ << std::endl;
    release_tengine();
    std::cout << __LINE__ << std::endl;
    return true;
  } catch (std::exception &e) {
    ctx->setBuffer3(e.what());
    return false;
  }
}
}
