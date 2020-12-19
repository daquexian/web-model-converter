    const messages = {
      en: {
        title1: "Online model conversion",
        title2: "Work out of the box",
        // title1: "Under maintenance",
        // title2: "will recover soon",
          title_tip: "The conversion is performed on your PC. Your model will never be uploaded to the cloud",
          choose_output_format: "Choose output format:",
          choose_input_format: "Choose input format:",
          onnx_tip: "Perform shape inference or optimization by onnx optimizer",
          ncnn_tip: "Use ncnnoptimize to optimize the ncnn model and get better speed",
          mlir_tip: "Only tf2 mlir dialect is supported. The mlir filename appeared in ncnn param is \"-\" (Refer to https://zhuanlan.zhihu.com/p/152535430)",
          onnx_opt_tip: "onnx optimizer is not maintained by onnx team, you can uncheck it if it fails",
          onnx_sim_checkbox: "Simplify the onnx model by onnx-simplifier",
          onnx_opt_checkbox: "Optimize the onnx model by onnx optimizer",
          onnx_shape_checkbox: "Generate model with shape information",
          ncnn_opt_checkbox: "Optimize the ncnn model by ncnnoptimize",
          ncnn_fp16_checkbox: "Generate fp16 model",
          select_button: "Select",
          convert_button: "Convert",
          input_shape_label: "Input shape:",
          converting_text: "Converting...",
          convert_ok_text: "Convert successfully!",
          author_tip: "This webpage itself is create by me, while the conversion is powered by the code from each framework.",
          no_loaded_warning: "Still loading.. Please wait for a mement and try again",
          onnxsim_multiple_dynamic_error: "The model has multiple inputs, among which one input has dynamic shape. This case is not supported.",
          onnxsim_dynamic_msg: 'The model has dynamic-shape input. If the model is only intended to work at a certain input shape, please enter the space-splitted input shape (you might want to check out <a href="https://lutzroeder.github.io/netron/" target="_blank">netron</a> to determine the NCHW or NHWC order) and try again. If not, you might want to use the onnx optmizer instead.',
          ncnn_error_tip: 'Some errors occurs when converting, but sometimes you can still download the model and <a href="https://zhuanlan.zhihu.com/p/93017149" target="_blank">fix manually</a> <br/>',
          unknown_error: 'Unknown error: "{0}", you can try to contact {1} developers',
          unknown_error2: 'Unknown error: "{0}", please contact daquexian',
          please_choose_one_model: "Please select {0} model",
          please_choose_two_models: "Please select {0} and {1} files",
      },
      zh: {
        title1: "省去编译转换工具的时间",
        title2: "开箱即用，一键转换",
        // title1: "维护中（很快会恢复）",
        // title2: "请暂时使用命令行工具版本",
          title_tip: "转换是完全由浏览器本身在本地进行的，您的模型不会被上传",
          choose_output_format: "选择目标格式：",
          choose_input_format: "选择输入格式：",
          onnx_tip: "使用 onnx simplifier 和 optimizer 对 onnx 模型进行优化，或对 onnx 模型做 shape inference",
          ncnn_tip: "使用 ncnnoptimize 产生优化后的 ncnn 模型，可以提升速度",
          mlir_tip: "只支持 tf2 mlir dialect。用于标识的 mlir 文件名为 \"-\"（使用方法参考 https://zhuanlan.zhihu.com/p/152535430）",
          onnx_opt_tip: "onnx optimizer 已经不被 onnx 团队维护了，如果使用它的时候出现问题可以取消勾选",
          onnx_sim_checkbox: "使用 onnx simplifier 优化模型",
          onnx_opt_checkbox: "使用 onnx optimizer 优化模型",
          onnx_shape_checkbox: "产生有 shape 信息的模型",
          ncnn_opt_checkbox: "使用 ncnnoptimize 优化模型",
          ncnn_fp16_checkbox: "产生 fp16 模型",
          select_button: "选择",
          convert_button: "转换",
          input_shape_label: "输入大小：",
          converting_text: "转换中……",
          convert_ok_text: "转换成功！",
          author_tip: "网页本身是我做的，模型转换部分仍基本使用了原各个框架转换工具的代码，感谢这些框架的作者们~",
          no_loaded_warning: "程序还没有加载完成，请等待一会儿～",
          onnxsim_multiple_dynamic_error: "模型有多个输入并且其中有动态大小的输入，这种情况还不支持",
          onnxsim_dynamic_msg: '模型有动态大小的输入，如果这个模型只想在某一种输入大小下工作，请在文本框里填写以空格分隔的输入大小（请注意各维度的顺序，如 NCHW 和 NHWC，如果不能确定可在 <a href="https://lutzroeder.github.io/netron/" target="_blank">netron</a> 里查看）并重新尝试一次，否则可以仅使用 onnx optimizer 进行优化',
          ncnn_error_tip: '转换遇到了一些错误，不过某些情况下你可以下载模型并 <a href="https://zhuanlan.zhihu.com/p/93017149" target="_blank">手工修复</a> <br/>',
          unknown_error: "遇到了未知错误：“{0}”，可以尝试向 {1} 开发者反馈",
          unknown_error2: "遇到了未知错误：“{0}”，请联系 daquexian",
          please_choose_one_model: "请选择 {0} 模型",
          please_choose_two_models: "请选择 {0} 和 {1} 文件",
      }
    }

    var getNavigatorLanguages = function() {
      if (typeof navigator === 'object') {
        var t = 'anguage', n = navigator, f;
        f = n['l' + t + 's'];
        return f && f.length ? f : (t = n['l' + t] ||
          n['browserL' + t] ||
          n['userL' + t]) ? [ t ] : t;
      }
    };

    const langs = getNavigatorLanguages();
    locale = 'en'
    langs.forEach((x) => { if (x.substring(0, 2) == "zh") { locale = 'zh'; } });

    const i18n = new VueI18n({
      locale:locale, // 设置地区
      messages, // 设置地区信息
    })

    var vm = new Vue({
        i18n,
        el: '#app',
        data: {
            inputFormat: 'onnx',
            outputFormat: 'tengine',
            fileList: [],
            selectDisabled: false,
            convertDisabled: true,
            hasResult: false,
            // Note: In ncnn, there are also converted models 
            // (which can be manually edited) even if the conversion fails.
            // so "convertSuccess" and "hasConvertedModel" are separated.
            convertSuccess: false,
            hasConvertedModel: false,
            ncnnoptFp16: false,
            ncnnConvertWithOpt: true,
            onnxSim: true,
            onnxOpt: true,
            onnxInferShape: true,
            latestFilename: 'model',
            showWaiting: false,
            wasmDownloaded: false,
            showShapeInputBox: false,
            shapeTxtFromUser: '',
        },
        computed: {
            dqxlimit: function () {
                let limit_dict = {
                    'onnx': 1,
                    'caffe': 2,
                    'tf': 1,
                    'mlir': 1,
                    'ncnn': 2,
                    'mxnet': 2,
                    'darknet': 2,
                    'tflite': 1,
                };
                return limit_dict[this.inputFormat];
            },
            uploadTip: function () {
                const msg_dict = {
                    'onnx': this.$t('please_choose_one_model', ['onnx']),
                    'caffe': this.$t('please_choose_two_models', ['prototxt', 'caffemodel']),
                    'tf': this.$t('please_choose_one_model', ['pb']),
                    'ncnn': this.$t('please_choose_two_models', ['param', 'bin']),
                    'mxnet': this.$t('please_choose_two_models', ['json', 'params']),
                    'darknet':this.$t('please_choose_two_models', ['cfg', 'weight']),
                    'tflite': this.$t('please_choose_one_model', ['tflite']),
                    'mlir': this.$t('please_choose_one_model', ['mlir']),
                };
                return msg_dict[this.inputFormat];
            },
            showUrls: function () {return this.hasResult && this.hasConvertedModel},
            showErrorMsg: function () {return this.hasResult && !this.convertSuccess},
            showSuccessMsg: function () {return this.hasResult && this.convertSuccess},
        },
        watch: {
            onnxSim: function (newValue, oldValue) {
                if (newValue) {
                    onnxInferShapeBak = this.onnxInferShape;
                    this.onnxInferShape = true;
                } else {
                    this.onnxInferShape = onnxInferShapeBak;
                }
            },
            inputFormat: function (newValue, oldValue) {
                console.log(newValue);
                console.log(this.fileList);
                this.fileList = [];
                this.convertDisabled = this.fileList.length != this.dqxlimit;
                this.selectDisabled = this.fileList.length == this.dqxlimit;
                this.hasResult = false;
                this.showShapeInputBox = false;
            },
            outputFormat: function (newValue, oldValue) {
                this.fileList = [];
                this.hasResult = false;
                this.showShapeInputBox = false;
                this.convertDisabled = this.fileList.length != this.dqxlimit;
                this.selectDisabled = this.fileList.length == this.dqxlimit;
                if ((this.inputFormat == "tf" && (newValue == "tengine" || newValue == "ncnn" || newValue == "onnx")) ||
                (this.inputFormat == "mxnet" && (newValue != "ncnn" && newValue != "tengine")) ||
                (this.inputFormat == "mlir" && newValue != "ncnn") ||
                (this.inputFormat == "ncnn" && newValue != "ncnn") ||
                (this.inputFormat == "darknet" && newValue != "tengine") ||
                (this.inputFormat == "tflite" && newValue != "tengine")
                ) {
                    this.inputFormat = "onnx";
                }
            },
        },
        methods: {
            submitUpload() {
                if (!this.wasmDownloaded) {
                    this.$message({
                      message: this.$t('no_loaded_warning'),
                      type: 'warning'
                    });
                    return;
                }
                this.hasResult = false;
                this.convertDisabled = true;
                this.showWaiting = true;
                const onnx_func_dict = {
                    'onnx': (uint8_arrs) => {
                        if (this.onnxSim) {
                            return onnxsim_js(uint8_arrs, this.onnxOpt);
                        }
                        if (this.onnxOpt && this.onnxInferShape) {
                            return onnxopt_and_shape_js(uint8_arrs);
                        }
                        if (this.onnxOpt) {
                            return onnxoptimize_js(uint8_arrs);
                        }
                        if (this.onnxInferShape) {
                            return onnx_shape_infer_js(uint8_arrs);
                        }
                    },
                };
                const ncnn_func_dict = {
                    'mlir': (uint8_arrs) => { return mlir2ncnn_js(uint8_arrs, this.ncnnConvertWithOpt, this.ncnnoptFp16) },
                    'onnx': (uint8_arrs) => { return onnx2ncnn_js(uint8_arrs, this.onnxOpt, this.ncnnConvertWithOpt, this.ncnnoptFp16) },
                    'caffe': (uint8_arrs) => { return caffe2ncnn_js(uint8_arrs, this.ncnnConvertWithOpt, this.ncnnoptFp16); },
                    'mxnet': (uint8_arrs) => { return mxnet2ncnn_js(uint8_arrs, this.ncnnConvertWithOpt, this.ncnnoptFp16); },
                    'ncnn': (uint8_arrs) => { return ncnnoptimize_js(uint8_arrs, this.ncnnoptFp16) },
                    // 'caffe': caffe2mnn_js
                };
                const mnn_func_dict = {
                    'onnx': (uint8_arrs) => { return onnx2mnn_js(uint8_arrs, this.onnxOpt); },
                    'caffe': caffe2mnn_js,
                    'tf': tf2mnn_js,
                };
                const tengine_func_dict = {
                    'onnx': (uint8_arrs) => { return onnx2tengine_js(uint8_arrs, this.onnxOpt); },
                    'caffe': caffe2tengine_js,
                    'tf': tf2tengine_js,
                    'mxnet': mxnet2tengine_js,
                    'darknet': darknet2tengine_js,
                    'tflite': tflite2tengine_js,
                }
                const tnn_func_dict = {
                    'onnx': (uint8_arrs) => { return onnx2tnn_js(uint8_arrs, this.onnxOpt) },
                };
                const func_dict = {
                    'ncnn': ncnn_func_dict,
                    'mnn': mnn_func_dict,
                    'tengine': tengine_func_dict,
                    'tnn': tnn_func_dict,
                    'onnx': onnx_func_dict,
                }
                try {
                   const files = this.$refs.select.uploadFiles.map(file => file.raw);
                   const func = func_dict[this.outputFormat][this.inputFormat];
                   files_to_uint8_arrs(files).then((uint8_arrs) => {
                   ret = func(uint8_arrs);
                    this.convertDisabled = false;
                    this.showWaiting = false;
                    if (ret[0] == "dynamic") {
                        if (ret[1] == "multi") {
                            this.convertSuccess = false;
                            this.errorMsg = this.$t('onnxsim_multiple_dynamic_error');
                            this.hasResult = true;
                            return;
                        } else if (ret[1] == "single") {
                            this.showShapeInputBox = true;
                            this.convertSuccess = false;
                            this.hasResult = true;
                            this.errorMsg = this.$t('onnxsim_dynamic_msg');
                            return;
                        }
                    }
                    this.hasResult = true;
                    const hasModel = ret[0];
                    this.hasConvertedModel = hasModel;
                    this.convertSuccess = hasModel;
                    if (hasModel) {
                       ret[1][0] = createObjectURL(ret[1][0]);
                       ret[1][1] = createObjectURL(ret[1][1]);
                        if (this.outputFormat == 'ncnn') {
                            [this.paramUrl, this.binUrl, errorMsg] = ret[1];
                            console.log("js err: " + errorMsg);
                            this.paramFilename = 'ncnn_model.param';
                            this.binFilename = 'ncnn_model.bin';
                            this.paramFilename = latestFilename + (this.ncnnConvertWithOpt ? '-opt' : '') + (this.ncnnoptFp16 ? "-fp16" : "") + '.param'
                            this.binFilename = latestFilename + (this.ncnnConvertWithOpt ? '-opt' : '') + (this.ncnnoptFp16 ? "-fp16" : "") + '.bin'
                            console.log(this.binFilename);
                            if (!(errorMsg === "")) {
                                this.errorMsg = this.$t("ncnn_error_tip") + errorMsg;
                                this.convertSuccess = false;
                            }
                        } else if (this.outputFormat == 'mnn') {
                            [this.paramUrl] = ret[1];
                            this.paramFilename = latestFilename + '.mnn'
                        } else if (this.outputFormat == 'tengine') {
                            [this.paramUrl] = ret[1];
                            this.paramFilename = latestFilename + '.tmfile'
                        } else if (this.outputFormat == 'onnx') {
                            [this.paramUrl] = ret[1];
                            this.paramFilename = latestFilename + (this.onnxSim ? '-sim' : (this.onnxOpt ? '-opt' : '') + (this.onnxInferShape ? "-shape" : "")) + '.onnx'
                        } else if (this.outputFormat == 'tnn') {
                            [this.paramUrl, this.binUrl] = ret[1];
                            this.paramFilename = 'tnn_model.tnnproto';
                            this.binFilename = 'tnn_model.tnnmodel';
                            this.paramFilename = latestFilename + (this.onnxOpt ? '.opt' : '') + '.tnnproto'
                            this.binFilename = latestFilename + (this.onnxOpt ? '.opt' : '') + '.tnnmodel'
                            console.log(this.binFilename);
                        }
                    } else {
                        this.errorMsg = ret[1];
                        console.log(this.errorMsg);
                    }
                }).catch(err => {
                    this.hasResult = true;
                    const hasModel = false;
                    this.hasConvertedModel = hasModel;
                    this.convertSuccess = hasModel;
                    this.convertDisabled = false;
                    this.showWaiting = false;
                    this.errorMsg = this.$t('unknown_error', [err, this.outputFormat]);
                })
                } catch (err) {
                    this.hasResult = true;
                    const hasModel = false;
                    this.hasConvertedModel = hasModel;
                    this.convertSuccess = hasModel;
                    this.convertDisabled = false;
                    this.showWaiting = false;
                    this.errorMsg = this.$t('unknown_error2', [err]);
                }
            },
            onSelectInputFormat(label) {
                console.log(label);
            },
            beforeUpload(file, fileList) {
                this.selectDisabled = fileList.length == this.dqxlimit;
                this.convertDisabled = fileList.length != this.dqxlimit;
                console.log(file.name);
                latestFilename = baseName(file.name);
            },
            handleRemove(file, fileList) {
                this.selectDisabled = fileList.length == this.dqxlimit;
                this.convertDisabled = fileList.length != this.dqxlimit;
                this.hasResult = false;
                this.showShapeInputBox = false;
            },
        }
    })

