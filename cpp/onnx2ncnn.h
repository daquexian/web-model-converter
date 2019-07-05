#include <expected.hpp>
#include <wmc_utils.h>

tl::expected<NcnnModel, std::string> onnx2ncnn(const std::string &model_str);

