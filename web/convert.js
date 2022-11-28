function baseName(str) {
  var base = new String(str).substring(str.lastIndexOf('/') + 1);
  if (base.lastIndexOf(".") != -1)
    base = base.substring(0, base.lastIndexOf("."));
  return base;
}

function getConvertedModelsAndErrorMsg(mdl, ctx) {
  let _get_buffer1 = mdl.cwrap('get_buffer1', "number", ["number"])
  let _get_buffer2 = mdl.cwrap('get_buffer2', "number", ["number"])
  let _get_buffer3 = mdl.cwrap('get_buffer3', "number", ["number"])
  let _get_buffer_size1 = mdl.cwrap('get_buffer_size1', "number", ["number"])
  let _get_buffer_size2 = mdl.cwrap('get_buffer_size2', "number", ["number"])
  let _get_buffer_size3 = mdl.cwrap('get_buffer_size3', "number", ["number"])
  bufferOffset1 = _get_buffer1(ctx);
  bufferSize1 = _get_buffer_size1(ctx);
  console.log("size1 " + bufferSize1);
  output1 = new Uint8Array(mdl.HEAP8.subarray(bufferOffset1, bufferOffset1 + bufferSize1));
  bufferOffset2 = _get_buffer2(ctx);
  bufferSize2 = _get_buffer_size2(ctx);
  console.log("size2 " + bufferSize2);
  output2 = new Uint8Array(mdl.HEAP8.subarray(bufferOffset2, bufferOffset2 + bufferSize2));

  bufferOffset3 = _get_buffer3(ctx);
  bufferSize3 = _get_buffer_size3(ctx);

  console.log(bufferSize3);
  output3 = new Uint8Array(mdl.HEAP8.subarray(bufferOffset3, bufferOffset3 + bufferSize3));
  output3 = String.fromCharCode.apply(null, output3);

  return [output1, output2, output3];
}

function getErrorMsg(mdl, ctx) {
  let _get_buffer = mdl.cwrap('get_buffer3', "number", ["number"])
  let _get_buffer_size = mdl.cwrap('get_buffer_size3', "number", ["number"])
  bufferOffset = _get_buffer(ctx);
  bufferSize = _get_buffer_size(ctx);
  output = new Uint8Array(mdl.HEAP8.subarray(bufferOffset, bufferOffset + bufferSize));
  const str = String.fromCharCode.apply(null, output);

  return str;
}

function transferToHeap(mdl, ui8a) {
  heapSpace = mdl._malloc(ui8a.length *
    ui8a.BYTES_PER_ELEMENT); // 1
  mdl.HEAP8.set(ui8a, heapSpace); // 2 
  return heapSpace;
}

function transferToHeapInt32(mdl, arr) {
  heapSpace = mdl._malloc(arr.length *
    arr.BYTES_PER_ELEMENT); // 1
  mdl.HEAP32.set(arr, heapSpace / arr.BYTES_PER_ELEMENT); // 2 
  return heapSpace;
}

const readFileAsArrayBuffer = (inputFile) => {
  const temporaryFileReader = new FileReader();

  return new Promise((resolve, reject) => {
    temporaryFileReader.onerror = () => {
      temporaryFileReader.abort();
      reject(new DOMException("Problem parsing input file."));
    };

    temporaryFileReader.onload = () => {
      resolve(temporaryFileReader.result);
    };
    temporaryFileReader.readAsArrayBuffer(inputFile);
  });
};

const cpp_js_wrapper = (mdl, export_name, uint8_arrs, extra_args, extra_types, free = false) => {
  var ctx = mdl.ccall('create_exporter', 'number');
  var args = [ctx];
  var arg_types = ["number"];
  const n = uint8_arrs.length;
  for (var i = 0; i < n; i++) {
    const arr = uint8_arrs[i];
    const arr_heap = transferToHeap(mdl, arr);
    args.push(arr_heap, arr.length);
    arg_types.push("array", "number");
  }
  const n2 = extra_args.length;
  for (var i = 0; i < n2; i++) {
    args.push(extra_args[i]);
    arg_types.push(extra_types[i]);
  }
  const convert = mdl.cwrap(export_name, arg_types);
  const success = convert.apply(null, args);
  if (success) {
    ret = getConvertedModelsAndErrorMsg(mdl, ctx);
  } else {
    ret = getErrorMsg(mdl, ctx);
  }
  mdl.ccall('free_exporter', null, ['number'], ctx);
  return [success, ret];
}

const files_to_uint8_arrs = async (files) => {
  const n = files.length;
  if (n == 2) {
    var swap = false;
    const filename0 = files[0].name;
    const filename1 = files[1].name;
    const size0 = files[0].size;
    const size1 = files[1].size;
    const format = vm.inputFromat;
    if (filename0.substring(filename0.length - 11) == '.caffemodel' && format == 'caffe') {
      swap = true;
    } else if (filename0.substring(filename0.length - 4) == '.bin' && format == 'ncnn') {
      swap = true;
    } else if (filename0.substring(filename0.length - 10) == '.pdiparams' && format == 'paddle') {
      swap = true;
    } else if ((filename0.substring(filename0.length - 6) == '.param' || filename0.substring(filename0.length - 7) == '.params') && format == 'mxnet') {
      swap = true;
    } else if (filename1.substring(filename1.length - 4) == '.cfg' && format == 'darknet') {
      swap = true;
    } else if (filename1.substring(filename1.length - 11) == '.caffemodel' && format == 'caffe') {
      swap = false;
    } else if (filename1.substring(filename1.length - 4) == '.bin' && format == 'ncnn') {
      swap = false;
    } else if ((filename1.substring(filename1.length - 6) == '.param' || filename1.substring(filename1.length - 7) == '.params') && format == 'mxnet') {
      swap = false;
    } else if (filename0.substring(filename0.length - 4) == '.cfg' && format == 'darknet') {
      swap = false;
    } else if (size0 > 3 * 1024 * 1024 && size1 < 512 * 1024) {
      swap = true;
    }
    if (swap) {
      tmp = files[0];
      files[0] = files[1];
      files[1] = tmp;
    }
  }

  var uint8_arrs = []
  for (var i = 0; i < n; i++) {
    const arr = await readFileAsArrayBuffer(files[i]);
    const arr_ui8a = new Uint8Array(arr);
    uint8_arrs.push(arr_ui8a);
  }
  return uint8_arrs;
}

const onnxsim_js = async (uint8_arrs, simplify, optimize, infer_shape) => {
  try {
    exit_status = 0;
    module = await create_onnxsim(
      {
        noInitialRun: true,
        print: (text) => {console.log(text); msg += ("<br/>" + text);},
        printErr: (text) => {
          console.log(text); 
          msg += ("<br/>" + text);
        },
        onExit: (status) => {exit_status = status;}
      });
    module['FS'].writeFile('/file1', uint8_arrs[0]);
    OUTPUT_FILE = '/sim.onnx'
    args = ['-i', '/file1', '-o', OUTPUT_FILE];
    if (!simplify) {
      args.push("--no-sim")
    }
    if (!optimize) {
      args.push("--no-opt")
    }
    if (!infer_shape) {
      args.push("--no-shape-inference")
    }
    msg = "";
    module.callMain(args)
    success = (exit_status == 0);
    if (success) {
      ret = [];
      ret.push(module['FS'].readFile(OUTPUT_FILE));
      ret.push(msg);
    } else {
      ret = msg;
    }
  } catch (e) {
    console.log(e);
    success = false;
    ret = e;
  }

  return [success, ret];
}

const x2ncnn_js = async (create_module_fn, uint8_arrs, extra_args, opt, fp16) => {
  try {
    // mlir2ncnn seems not trigger onExit
    exit_status = 0;
    module = await create_module_fn(
      {
        noInitialRun: true,
        print: (text) => {console.log(text); msg += ("<br/>" + text);},
        printErr: (text) => {
          console.log(text); 
          // TODO: move this check to mlir2ncnn itself
          if (!text.includes("this is a no-op")) {
            msg += ("<br/>" + text);
          }
        },
        onExit: (status) => {exit_status = status;}
      });
    module['FS'].writeFile('/file1', uint8_arrs[0]);
    args = ['/file1'];
    if (uint8_arrs.length > 1) {
      module['FS'].writeFile('/file2', uint8_arrs[1]);
      args.push('/file2')
    }
    NCNN_PARAM = '/ncnn.param';
    NCNN_BIN = '/ncnn.bin';
    args.push(NCNN_PARAM, NCNN_BIN);
    args = args.concat(extra_args);
    msg = "";
    module.callMain(args)
    success = (exit_status == 0);
    if (success) {
      ret = [];
      ret.push(module['FS'].readFile(NCNN_PARAM));
      ret.push(module['FS'].readFile(NCNN_BIN));
      if (create_module_fn == create_darknet2ncnn || create_module_fn == create_ncnnoptimize) {
        ret.push("");     // FIXME: support general log
      } else {
        ret.push(msg);
      }
    } else {
      ret = msg;
    }
  } catch (e) {
    console.log(e);
    success = false;
    ret = e;
  }

  if (!success || !(ret[2] === "")) {
    return [success, ret];
  }

  if (opt) {
    [success, ret] = await ncnnoptimize_js([ret[0], ret[1]], fp16);
  }

  return [success, ret];
}

const ncnnoptimize_js = async (uint8_arrs, fp16) => {
  const fp16_arg = fp16 ? "1" : "0";
  return x2ncnn_js(create_ncnnoptimize, uint8_arrs, [fp16_arg], false, false);
}

const onnx2ncnn_js = async (uint8_arrs, onnxsim, ncnnopt, fp16) => {
  if (onnxsim) {
    const tmp = await onnxsim_js(uint8_arrs, true, true, true);
    [success, ret] = tmp;
    if (!success) {
      return tmp;
    }
    uint8_arrs = [ret[0]];
  }

  return x2ncnn_js(create_onnx2ncnn, uint8_arrs, [], ncnnopt, fp16);
}

const mlir2ncnn_js = async (uint8_arrs, opt, fp16) => {
  return x2ncnn_js(create_mlir2ncnn, uint8_arrs, [], opt, fp16);
}

const darknet2ncnn_js = async (uint8_arrs, merge, opt, fp16) => {
  merge_arg = merge ? 1 : 0;
  return x2ncnn_js(create_darknet2ncnn, uint8_arrs, [merge_arg], opt, fp16);
}

const caffe2ncnn_js = async (uint8_arrs, opt, fp16) => {
  return x2ncnn_js(create_caffe2ncnn, uint8_arrs, [], opt, fp16);
}

const mxnet2ncnn_js = async (uint8_arrs, opt, fp16) => {
  return x2ncnn_js(create_mxnet2ncnn, uint8_arrs, [], opt, fp16);
}

const x2mnn_js = async (src_format, uint8_arrs, extra_args) => {
  try {
    exit_status = 0;
    module = await create_x2mnn(
      {
        noInitialRun: true,
        print: (text) => {console.log(text); msg += ("<br/>" + text);},
        printErr: (text) => {
          console.log(text); 
          msg += ("<br/>" + text);
        },
        onExit: (status) => {exit_status = status;}
      });
    args = ['--bizCode', 'mnn', '-f', src_format];
    module['FS'].writeFile('/file_one', uint8_arrs[0]);
    if (uint8_arrs.length == 1) {
      args.push('--modelFile')
      args.push('/file_one')
    } else if (uint8_arrs.length == 2) {
      args.push('--prototxt')
      args.push('/file_one')
      module['FS'].writeFile('/file_two', uint8_arrs[1]);
      args.push('--modelFile')
      args.push('/file_two')
    } else {
      // TODO: raise exception
    }
    OUTPUT_FILE = '/tmp/model.mnn';
    args.push('--MNNModel');
    args.push(OUTPUT_FILE);
    args = args.concat(extra_args);
    msg = "";
    module.callMain(args)
    success = (exit_status == 0);
    if (success) {
      ret = [];
      ret.push(module['FS'].readFile(OUTPUT_FILE));
      ret.push(msg);
    } else {
      ret = msg;
    }
  } catch (e) {
    console.log(e);
    success = false;
    ret = e;
  }

  return [success, ret];
}

const caffe2mnn_js = async (uint8_arrs) => {
  return x2mnn_js("CAFFE", uint8_arrs, []);
}

const onnx2mnn_js = async (uint8_arrs, sim) => {
  mdl = Module;

  if (sim) {
    const tmp = await onnxsim_js(uint8_arrs, true, true, true);
    [success, ret] = tmp;
    if (!success) {
      return tmp;
    }
    uint8_arrs = [ret[0]];
  }

  return x2mnn_js('ONNX', uint8_arrs, []);
}

const tf2mnn_js = async (uint8_arrs) => {
  return x2mnn_js('TF', uint8_arrs, []);
}

const tflite2mnn_js = async (uint8_arrs) => {
  return x2mnn_js('TFLITE', uint8_arrs, []);
}

const x2tengine_js = async (src_format, uint8_arrs, extra_args) => {
  try {
    exit_status = 0;
    module = await create_x2tengine(
      {
        noInitialRun: true,
        print: (text) => {console.log(text); msg += ("<br/>" + text);},
        printErr: (text) => {
          console.log(text); 
          msg += ("<br/>" + text);
        },
        onExit: (status) => {exit_status = status;}
      });
    args = ['-f', src_format];
    module['FS'].writeFile('/file1', uint8_arrs[0]);
    if (uint8_arrs.length == 1) {
      args.push('-m')
      args.push('/file1')
    } else if (uint8_arrs.length == 2) {
      args.push('-p')
      args.push('/file1')
      module['FS'].writeFile('/file2', uint8_arrs[1]);
      args.push('-m')
      args.push('/file2')
    } else {
      // TODO: raise exception
    }
    TMFILE = '/tmp/tengine.tmfile';
    args.push('-o');
    args.push(TMFILE);
    args = args.concat(extra_args);
    msg = "";
    module.callMain(args)
    success = (exit_status == 0);
    if (success) {
      ret = [];
      ret.push(module['FS'].readFile(TMFILE));
      ret.push(msg);
    } else {
      ret = msg;
    }
  } catch (e) {
    console.log(e);
    success = false;
    ret = e;
  }

  return [success, ret];
}

const onnx2tengine_js = async (uint8_arrs, sim) => {
  mdl = Module;

  if (sim) {
    const tmp = await onnxsim_js(uint8_arrs, true, true, true);
    [success, ret] = tmp;
    if (!success) {
      return tmp;
    }
    uint8_arrs = [ret[0]];
  }

  return x2tengine_js("onnx", uint8_arrs, []);
}

const caffe2tengine_js = async (uint8_arrs) => {
  return x2tengine_js("caffe", uint8_arrs, []);
}

const tf2tengine_js = async (uint8_arrs) => {
  return x2tengine_js("tensorflow", uint8_arrs, []);
}

const mxnet2tengine_js = async (uint8_arrs) => {
  return x2tengine_js("mxnet", uint8_arrs, []);
}

const darknet2tengine_js = async (uint8_arrs) => {
  return x2tengine_js("darknet", uint8_arrs, []);
}

const tflite2tengine_js = async (uint8_arrs) => {
  return x2tengine_js("tflite", uint8_arrs, []);
}

const ncnn2tengine_js = async (uint8_arrs) => {
  return x2tengine_js("ncnn", uint8_arrs, []);
}

const onnx2tnn_js = async (uint8_arrs, onnxsim) => {
  mdl = Module;

  if (onnxsim) {
    const tmp = await onnxsim_js(uint8_arrs, true, true, true);
    [success, ret] = tmp;
    if (!success) {
      return tmp;
    }
    uint8_arrs = [ret[0]];
  }

  tmp = cpp_js_wrapper(mdl, 'onnx2tnn_export', uint8_arrs, [], []);
  [success, ret] = tmp;
  if (!success || !(ret[2] === "")) {
    return tmp;
  }

  return [success, ret];
}

const check_onnx_static_input_shape_js = (uint8_arrs) => {
  mdl = Module;
  const export_name = 'check_static_input_size_export';
  return cpp_js_wrapper(mdl, export_name, uint8_arrs, [], []);
}

const paddle_js = async (uint8_arrs) => {
  try {
    exit_status = 0;
    module = await create_paddle_opt(
      {
        noInitialRun: true,
        print: (text) => {console.log(text); msg += ("<br/>" + text);},
        printErr: (text) => {
          console.log(text); 
          msg += ("<br/>" + text);
        },
        onExit: (status) => {exit_status = status;}
      });
    module['FS'].writeFile('/file1', uint8_arrs[0]);
    module['FS'].writeFile('/file2', uint8_arrs[1]);
    OUTPUT_FILE = '/xxx'
    args = ['--model_file', '/file1', '--param_file', '/file2', '--optimize_out', OUTPUT_FILE];
    msg = "";
    module.callMain(args)
    success = (exit_status == 0);
    if (success) {
      ret = [];
      ret.push(module['FS'].readFile(OUTPUT_FILE + '.nb'));
      ret.push(msg);
    } else {
      ret = msg;
    }
  } catch (e) {
    console.log(e);
    success = false;
    ret = e;
  }

  return [success, ret];
}

//
// create object url
//
function createObjectURL(array, type) {
  var useTypedArray = (typeof Uint8Array !== 'undefined');
  var isSafari = (
    navigator.userAgent.indexOf('Safari') !== -1 &&
    navigator.vendor.indexOf('Apple') !== -1
  );
  var data = '';
  var bb;
  var blob;
  var tmp;
  var i;
  var il;

  if (useTypedArray) {
    array = new Uint8Array(array);
  }

  // avoid blob url in safari
  if (!isSafari) {

    // Blob constructor
    try {
      blob = new Blob([array], {type: type});
    } catch (e) {
    }

    // BlobBuilder
    if (
      (tmp = window.WebkitBlobBuilder) !== void 0 ||
      (tmp = window.MozBlobBuilder) !== void 0 ||
      (tmp = window.MSBlobBuilder) !== void 0
    ) {
      bb = new tmp();
      bb.append(array.buffer);
      blob = bb.getBlob(type);
    }

    // createObjectURL
    if (blob && (
      ((tmp = window.URL) && tmp.createObjectURL) ||
      ((tmp = window.webkitURL) && tmp.createObjectURL)
    )) {
      return tmp.createObjectURL(blob);
    }
  }

  // DataURL
  for (i = 0, il = array.length; i < il;) {
    data += String.fromCharCode.apply(
      null,
      useTypedArray ?
        array.subarray(i, i += 0x7fff) :
        array.slice(i, i += 0x7fff)
    );
  }

  return 'data:' + type + ';base64,' + window.btoa(data);
}

createOnnxOpt(/* optional default settings */).then(function (Module) {
  // this is reached when everything is ready, and you can call methods on Module
  oom = Module;
});
