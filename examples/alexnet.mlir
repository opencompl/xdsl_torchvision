"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !torch.vtensor<[1,3,224,224],f32>):
    %0 = "torch.constant.int"() {value = 0 : i64} : () -> !torch.int
    %1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
    %2 = "torch.constant.float"() {value = 1.000000e+00 : f64} : () -> !torch.float
    %3 = "torch.constant.int"() {value = -1 : i64} : () -> !torch.int
    %4 = "torch.constant.bool"() {value = true} : () -> !torch.bool
    %5 = "torch.constant.bool"() {value = false} : () -> !torch.bool
    %6 = "torch.constant.none"() : () -> !torch.none
    %7 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<1000x4096xf32>} : () -> !torch.vtensor<[1000,4096],f32>
    %8 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<1000xf32>} : () -> !torch.vtensor<[1000],f32>
    %9 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<4096x4096xf32>} : () -> !torch.vtensor<[4096,4096],f32>
    %10 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<4096xf32>} : () -> !torch.vtensor<[4096],f32>
    %11 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<4096x9216xf32>} : () -> !torch.vtensor<[4096,9216],f32>
    %12 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<4096xf32>} : () -> !torch.vtensor<[4096],f32>
    %13 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<256x256x3x3xf32>} : () -> !torch.vtensor<[256,256,3,3],f32>
    %14 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<256xf32>} : () -> !torch.vtensor<[256],f32>
    %15 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<256x384x3x3xf32>} : () -> !torch.vtensor<[256,384,3,3],f32>
    %16 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<256xf32>} : () -> !torch.vtensor<[256],f32>
    %17 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<384x192x3x3xf32>} : () -> !torch.vtensor<[384,192,3,3],f32>
    %18 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<384xf32>} : () -> !torch.vtensor<[384],f32>
    %19 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<192x64x5x5xf32>} : () -> !torch.vtensor<[192,64,5,5],f32>
    %20 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<192xf32>} : () -> !torch.vtensor<[192],f32>
    %21 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<64x3x11x11xf32>} : () -> !torch.vtensor<[64,3,11,11],f32>
    %22 = "torch.vtensor.literal"() {value = dense_resource<__elided__> : tensor<64xf32>} : () -> !torch.vtensor<[64],f32>
    %23 = "torch.constant.int"() {value = 4 : i64} : () -> !torch.int
    %24 = "torch.constant.int"() {value = 2 : i64} : () -> !torch.int
    %25 = "torch.constant.int"() {value = 3 : i64} : () -> !torch.int
    %26 = "torch.prim.ListConstruct"(%23, %23) : (!torch.int, !torch.int) -> !torch.list<int>
    %27 = "torch.prim.ListConstruct"(%24, %24) : (!torch.int, !torch.int) -> !torch.list<int>
    %28 = "torch.prim.ListConstruct"(%1, %1) : (!torch.int, !torch.int) -> !torch.list<int>
    %29 = "torch.prim.ListConstruct"(%0, %0) : (!torch.int, !torch.int) -> !torch.list<int>
    %30 = "torch.aten.convolution"(%arg0, %21, %22, %26, %27, %28, %5, %29, %1) : (!torch.vtensor<[1,3,224,224],f32>, !torch.vtensor<[64,3,11,11],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,64,55,55],f32>
    %31 = "torch.aten.relu"(%30) : (!torch.vtensor<[1,64,55,55],f32>) -> !torch.vtensor<[1,64,55,55],f32>
    %32 = "torch.prim.ListConstruct"(%25, %25) : (!torch.int, !torch.int) -> !torch.list<int>
    %33 = "torch.aten.max_pool2d"(%31, %32, %27, %29, %28, %5) : (!torch.vtensor<[1,64,55,55],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool) -> !torch.vtensor<[1,64,27,27],f32>
    %34 = "torch.aten.convolution"(%33, %19, %20, %28, %27, %28, %5, %29, %1) : (!torch.vtensor<[1,64,27,27],f32>, !torch.vtensor<[192,64,5,5],f32>, !torch.vtensor<[192],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,192,27,27],f32>
    %35 = "torch.aten.relu"(%34) : (!torch.vtensor<[1,192,27,27],f32>) -> !torch.vtensor<[1,192,27,27],f32>
    %36 = "torch.aten.max_pool2d"(%35, %32, %27, %29, %28, %5) : (!torch.vtensor<[1,192,27,27],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool) -> !torch.vtensor<[1,192,13,13],f32>
    %37 = "torch.aten.convolution"(%36, %17, %18, %28, %28, %28, %5, %29, %1) : (!torch.vtensor<[1,192,13,13],f32>, !torch.vtensor<[384,192,3,3],f32>, !torch.vtensor<[384],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,384,13,13],f32>
    %38 = "torch.aten.relu"(%37) : (!torch.vtensor<[1,384,13,13],f32>) -> !torch.vtensor<[1,384,13,13],f32>
    %39 = "torch.aten.convolution"(%38, %15, %16, %28, %28, %28, %5, %29, %1) : (!torch.vtensor<[1,384,13,13],f32>, !torch.vtensor<[256,384,3,3],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,13,13],f32>
    %40 = "torch.aten.relu"(%39) : (!torch.vtensor<[1,256,13,13],f32>) -> !torch.vtensor<[1,256,13,13],f32>
    %41 = "torch.aten.convolution"(%40, %13, %14, %28, %28, %28, %5, %29, %1) : (!torch.vtensor<[1,256,13,13],f32>, !torch.vtensor<[256,256,3,3],f32>, !torch.vtensor<[256],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int) -> !torch.vtensor<[1,256,13,13],f32>
    %42 = "torch.aten.relu"(%41) : (!torch.vtensor<[1,256,13,13],f32>) -> !torch.vtensor<[1,256,13,13],f32>
    %43 = "torch.aten.max_pool2d"(%42, %32, %27, %29, %28, %5) : (!torch.vtensor<[1,256,13,13],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool) -> !torch.vtensor<[1,256,6,6],f32>
    "torch.runtime.assert"(%4) {message = "unimplemented: only support cases where input and output size are equal for non-unit output size"} : (!torch.bool) -> ()
    "torch.runtime.assert"(%4) {message = "unimplemented: only support cases where input and output size are equal for non-unit output size"} : (!torch.bool) -> ()
    %44 = "torch.prim.ListConstruct"(%1, %1) : (!torch.int, !torch.int) -> !torch.list<int>
    %45 = "torch.prim.ListConstruct"(%1, %1) : (!torch.int, !torch.int) -> !torch.list<int>
    %46 = "torch.prim.ListConstruct"(%0, %0) : (!torch.int, !torch.int) -> !torch.list<int>
    %47 = "torch.aten.avg_pool2d"(%43, %44, %45, %46, %5, %4, %6) : (!torch.vtensor<[1,256,6,6],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.bool, !torch.none) -> !torch.vtensor<[1,256,6,6],f32>
    %48 = "torch.prim.ListConstruct"(%1, %3) : (!torch.int, !torch.int) -> !torch.list<int>
    %49 = "torch.aten.view"(%47, %48) : (!torch.vtensor<[1,256,6,6],f32>, !torch.list<int>) -> !torch.vtensor<[1,9216],f32>
    %50 = "torch.aten.transpose.int"(%11, %0, %1) : (!torch.vtensor<[4096,9216],f32>, !torch.int, !torch.int) -> !torch.vtensor<[9216,4096],f32>
    %51 = "torch.aten.mm"(%49, %50) : (!torch.vtensor<[1,9216],f32>, !torch.vtensor<[9216,4096],f32>) -> !torch.vtensor<[1,4096],f32>
    %52 = "torch.aten.add.Tensor"(%51, %12, %2) : (!torch.vtensor<[1,4096],f32>, !torch.vtensor<[4096],f32>, !torch.float) -> !torch.vtensor<[1,4096],f32>
    %53 = "torch.aten.relu"(%52) : (!torch.vtensor<[1,4096],f32>) -> !torch.vtensor<[1,4096],f32>
    %54 = "torch.aten.transpose.int"(%9, %0, %1) : (!torch.vtensor<[4096,4096],f32>, !torch.int, !torch.int) -> !torch.vtensor<[4096,4096],f32>
    %55 = "torch.aten.mm"(%53, %54) : (!torch.vtensor<[1,4096],f32>, !torch.vtensor<[4096,4096],f32>) -> !torch.vtensor<[1,4096],f32>
    %56 = "torch.aten.add.Tensor"(%55, %10, %2) : (!torch.vtensor<[1,4096],f32>, !torch.vtensor<[4096],f32>, !torch.float) -> !torch.vtensor<[1,4096],f32>
    %57 = "torch.aten.relu"(%56) : (!torch.vtensor<[1,4096],f32>) -> !torch.vtensor<[1,4096],f32>
    %58 = "torch.aten.transpose.int"(%7, %0, %1) : (!torch.vtensor<[1000,4096],f32>, !torch.int, !torch.int) -> !torch.vtensor<[4096,1000],f32>
    %59 = "torch.aten.mm"(%57, %58) : (!torch.vtensor<[1,4096],f32>, !torch.vtensor<[4096,1000],f32>) -> !torch.vtensor<[1,1000],f32>
    %60 = "torch.aten.add.Tensor"(%59, %8, %2) : (!torch.vtensor<[1,1000],f32>, !torch.vtensor<[1000],f32>, !torch.float) -> !torch.vtensor<[1,1000],f32>
    "func.return"(%60) : (!torch.vtensor<[1,1000],f32>) -> ()
  }) {function_type = (!torch.vtensor<[1,3,224,224],f32>) -> !torch.vtensor<[1,1000],f32>, sym_name = "forward"} : () -> ()
}) {torch.debug_module_name = "AlexNet"} : () -> ()
