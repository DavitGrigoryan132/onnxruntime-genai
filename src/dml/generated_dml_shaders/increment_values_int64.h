#if 0
;
; Note: shader requires additional functionality:
;       64-Bit integer
;
;
; Input signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
;
; Output signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
; shader hash: ab8d7ee474c0d59fbe4c5c5c817ce5bd
;
; Pipeline Runtime Information: 
;
; Compute Shader
; NumThreads=(256,1,1)
;
;
; Buffer Definitions:
;
; cbuffer 
; {
;
;   [12 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [8 x i8] (type annotation not present)
;
; }
;
;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ------
;                                   cbuffer      NA          NA     CB0            cb0     1
;                                       UAV  struct         r/w      U0             u0     1
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

%dx.types.Handle = type { i8* }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.ResRet.i32 = type { i32, i32, i32, i32, i32 }
%"class.RWStructuredBuffer<long long>" = type { i64 }
%Constants = type { i32, i32, i32 }

define void @CSMain() {
  %1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 2, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %3 = call i32 @dx.op.threadId.i32(i32 93, i32 0)  ; ThreadId(component)
  %4 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %2, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %5 = extractvalue %dx.types.CBufRet.i32 %4, 1
  %6 = add i32 %5, %3
  %7 = extractvalue %dx.types.CBufRet.i32 %4, 0
  %8 = icmp ult i32 %6, %7
  br i1 %8, label %9, label %21

; <label>:9                                       ; preds = %0
  %10 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %1, i32 %6, i32 0, i8 3, i32 8)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %11 = extractvalue %dx.types.ResRet.i32 %10, 0
  %12 = extractvalue %dx.types.ResRet.i32 %10, 1
  %13 = zext i32 %11 to i64
  %14 = zext i32 %12 to i64
  %15 = shl i64 %14, 32
  %16 = or i64 %13, %15
  %17 = add nsw i64 %16, 1
  %18 = trunc i64 %17 to i32
  %19 = lshr i64 %17, 32
  %20 = trunc i64 %19 to i32
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %1, i32 %6, i32 0, i32 %18, i32 %20, i32 undef, i32 undef, i8 3, i32 8)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  br label %21

; <label>:21                                      ; preds = %9, %0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.threadId.i32(i32, i32) #0

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32, %dx.types.Handle, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1) #1

; Function Attrs: nounwind readonly
declare %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32, %dx.types.Handle, i32, i32, i8, i32) #1

; Function Attrs: nounwind
declare void @dx.op.rawBufferStore.i32(i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8, i32) #2

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind }

!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.resources = !{!4}
!dx.entryPoints = !{!10}

!0 = !{!"dxcoob 1.7.2308.7 (69e54e290)"}
!1 = !{i32 1, i32 2}
!2 = !{i32 1, i32 7}
!3 = !{!"cs", i32 6, i32 2}
!4 = !{null, !5, !8, null}
!5 = !{!6}
!6 = !{i32 0, %"class.RWStructuredBuffer<long long>"* undef, !"", i32 0, i32 0, i32 1, i32 12, i1 false, i1 false, i1 false, !7}
!7 = !{i32 1, i32 8}
!8 = !{!9}
!9 = !{i32 0, %Constants* undef, !"", i32 0, i32 0, i32 1, i32 12, null}
!10 = !{void ()* @CSMain, !"CSMain", null, !4, !11}
!11 = !{i32 0, i64 1048592, i32 4, !12}
!12 = !{i32 256, i32 1, i32 1}

#endif

const unsigned char g_CSMain[] = {
  0x44, 0x58, 0x42, 0x43, 0xd0, 0x1c, 0xb5, 0x23, 0x6a, 0xda, 0x2c, 0xb9,
  0x45, 0x01, 0x1c, 0x52, 0x9f, 0x33, 0x40, 0x0b, 0x01, 0x00, 0x00, 0x00,
  0x1c, 0x07, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0xe8, 0x00, 0x00, 0x00, 0x04, 0x01, 0x00, 0x00, 0x53, 0x46, 0x49, 0x30,
  0x08, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x49, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x4f, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x50, 0x53, 0x56, 0x30,
  0x78, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x48, 0x41, 0x53, 0x48, 0x14, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xab, 0x8d, 0x7e, 0xe4, 0x74, 0xc0, 0xd5, 0x9f,
  0xbe, 0x4c, 0x5c, 0x5c, 0x81, 0x7c, 0xe5, 0xbd, 0x44, 0x58, 0x49, 0x4c,
  0x10, 0x06, 0x00, 0x00, 0x62, 0x00, 0x05, 0x00, 0x84, 0x01, 0x00, 0x00,
  0x44, 0x58, 0x49, 0x4c, 0x02, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0xf8, 0x05, 0x00, 0x00, 0x42, 0x43, 0xc0, 0xde, 0x21, 0x0c, 0x00, 0x00,
  0x7b, 0x01, 0x00, 0x00, 0x0b, 0x82, 0x20, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x07, 0x81, 0x23, 0x91, 0x41, 0xc8, 0x04, 0x49,
  0x06, 0x10, 0x32, 0x39, 0x92, 0x01, 0x84, 0x0c, 0x25, 0x05, 0x08, 0x19,
  0x1e, 0x04, 0x8b, 0x62, 0x80, 0x14, 0x45, 0x02, 0x42, 0x92, 0x0b, 0x42,
  0xa4, 0x10, 0x32, 0x14, 0x38, 0x08, 0x18, 0x4b, 0x0a, 0x32, 0x52, 0x88,
  0x48, 0x90, 0x14, 0x20, 0x43, 0x46, 0x88, 0xa5, 0x00, 0x19, 0x32, 0x42,
  0xe4, 0x48, 0x0e, 0x90, 0x91, 0x22, 0xc4, 0x50, 0x41, 0x51, 0x81, 0x8c,
  0xe1, 0x83, 0xe5, 0x8a, 0x04, 0x29, 0x46, 0x06, 0x51, 0x18, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x1b, 0x8c, 0xe0, 0xff, 0xff, 0xff, 0xff, 0x07,
  0x40, 0x02, 0xa8, 0x0d, 0x86, 0xf0, 0xff, 0xff, 0xff, 0xff, 0x03, 0x20,
  0x01, 0xd5, 0x06, 0x62, 0xf8, 0xff, 0xff, 0xff, 0xff, 0x01, 0x90, 0x00,
  0x49, 0x18, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x13, 0x82, 0x60, 0x42,
  0x20, 0x4c, 0x08, 0x06, 0x00, 0x00, 0x00, 0x00, 0x89, 0x20, 0x00, 0x00,
  0x35, 0x00, 0x00, 0x00, 0x32, 0x22, 0x48, 0x09, 0x20, 0x64, 0x85, 0x04,
  0x93, 0x22, 0xa4, 0x84, 0x04, 0x93, 0x22, 0xe3, 0x84, 0xa1, 0x90, 0x14,
  0x12, 0x4c, 0x8a, 0x8c, 0x0b, 0x84, 0xa4, 0x4c, 0x10, 0x6c, 0x23, 0x00,
  0x25, 0x00, 0x14, 0xe6, 0x08, 0xc0, 0xa0, 0x0c, 0x63, 0x0c, 0x22, 0x47,
  0x0d, 0x97, 0x3f, 0x61, 0x0f, 0x21, 0xf9, 0xdc, 0x46, 0x15, 0x2b, 0x31,
  0xf9, 0xc8, 0x6d, 0x23, 0x62, 0x8c, 0x31, 0xe6, 0x08, 0x10, 0x3a, 0xf7,
  0x0c, 0x97, 0x3f, 0x61, 0x0f, 0x21, 0xf9, 0x21, 0xd0, 0x0c, 0x0b, 0x81,
  0x02, 0x54, 0x08, 0x33, 0xd2, 0x20, 0x35, 0x47, 0x10, 0x14, 0x23, 0x8d,
  0x33, 0x06, 0xa3, 0x76, 0xd3, 0x70, 0xf9, 0x13, 0xf6, 0x10, 0x92, 0xbf,
  0x12, 0xd2, 0x4a, 0x4c, 0x3e, 0x72, 0xdb, 0xa8, 0x18, 0x63, 0x8c, 0x51,
  0x8e, 0x37, 0xd2, 0x18, 0x67, 0x10, 0x2c, 0x0b, 0x18, 0x69, 0x8c, 0x31,
  0xc6, 0x38, 0x83, 0xe4, 0x40, 0xc0, 0x1c, 0x01, 0x28, 0xcc, 0x34, 0x06,
  0xe3, 0xc0, 0x0e, 0xe1, 0x30, 0x0f, 0xf3, 0xe0, 0x06, 0xb2, 0x70, 0x0b,
  0xb3, 0x40, 0x0f, 0xf2, 0x50, 0x0f, 0xe3, 0x40, 0x0f, 0xf5, 0x20, 0x0f,
  0xe5, 0x40, 0x0e, 0xa2, 0x50, 0x0f, 0xe6, 0x60, 0x0e, 0xe5, 0x20, 0x0f,
  0x7c, 0xc0, 0x0e, 0xef, 0xe0, 0x0e, 0xe7, 0x00, 0x06, 0xec, 0xf0, 0x0e,
  0xee, 0x70, 0x0e, 0x7e, 0x80, 0x82, 0x4a, 0xf6, 0x12, 0xce, 0x69, 0xa4,
  0x09, 0x68, 0x26, 0x09, 0x0d, 0x63, 0x0c, 0xc2, 0x53, 0x00, 0x00, 0x00,
  0x13, 0x14, 0x72, 0xc0, 0x87, 0x74, 0x60, 0x87, 0x36, 0x68, 0x87, 0x79,
  0x68, 0x03, 0x72, 0xc0, 0x87, 0x0d, 0xaf, 0x50, 0x0e, 0x6d, 0xd0, 0x0e,
  0x7a, 0x50, 0x0e, 0x6d, 0x00, 0x0f, 0x7a, 0x30, 0x07, 0x72, 0xa0, 0x07,
  0x73, 0x20, 0x07, 0x6d, 0x90, 0x0e, 0x71, 0xa0, 0x07, 0x73, 0x20, 0x07,
  0x6d, 0x90, 0x0e, 0x78, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d, 0x90, 0x0e,
  0x71, 0x60, 0x07, 0x7a, 0x30, 0x07, 0x72, 0xd0, 0x06, 0xe9, 0x30, 0x07,
  0x72, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d, 0x90, 0x0e, 0x76, 0x40, 0x07,
  0x7a, 0x60, 0x07, 0x74, 0xd0, 0x06, 0xe6, 0x10, 0x07, 0x76, 0xa0, 0x07,
  0x73, 0x20, 0x07, 0x6d, 0x60, 0x0e, 0x73, 0x20, 0x07, 0x7a, 0x30, 0x07,
  0x72, 0xd0, 0x06, 0xe6, 0x60, 0x07, 0x74, 0xa0, 0x07, 0x76, 0x40, 0x07,
  0x6d, 0xe0, 0x0e, 0x78, 0xa0, 0x07, 0x71, 0x60, 0x07, 0x7a, 0x30, 0x07,
  0x72, 0xa0, 0x07, 0x76, 0x40, 0x07, 0x43, 0x9e, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x86, 0x3c, 0x04, 0x10, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x79, 0x14, 0x20,
  0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0xf2, 0x34,
  0x40, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0xe4,
  0x81, 0x80, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60,
  0xc8, 0x23, 0x01, 0x01, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x40, 0x16, 0x08, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x32, 0x1e, 0x98, 0x14,
  0x19, 0x11, 0x4c, 0x90, 0x8c, 0x09, 0x26, 0x47, 0xc6, 0x04, 0x43, 0x1a,
  0x25, 0x50, 0x04, 0xe5, 0x50, 0x0c, 0x23, 0x00, 0x85, 0x51, 0x10, 0x85,
  0x50, 0x80, 0x04, 0xc4, 0x46, 0x00, 0xa8, 0x16, 0x28, 0x20, 0x60, 0x00,
  0xdd, 0x19, 0x00, 0xca, 0x33, 0x00, 0x00, 0x00, 0x79, 0x18, 0x00, 0x00,
  0x3e, 0x00, 0x00, 0x00, 0x1a, 0x03, 0x4c, 0x90, 0x46, 0x02, 0x13, 0xc4,
  0x8e, 0x0c, 0x6f, 0xec, 0xed, 0x4d, 0x0c, 0x24, 0xc6, 0xe5, 0xc6, 0x45,
  0x66, 0x06, 0x06, 0xc7, 0xe5, 0x06, 0x04, 0xc5, 0x26, 0xa7, 0xac, 0x86,
  0xa6, 0x4c, 0x26, 0x07, 0x26, 0x65, 0x43, 0x10, 0x4c, 0x10, 0x06, 0x63,
  0x82, 0x30, 0x1c, 0x1b, 0x84, 0x81, 0x98, 0x20, 0x0c, 0xc8, 0x06, 0x61,
  0x30, 0x28, 0x8c, 0xcd, 0x4d, 0x10, 0x86, 0x64, 0xc3, 0x80, 0x24, 0xc4,
  0x04, 0x61, 0x50, 0x26, 0x08, 0x57, 0x44, 0x60, 0x82, 0x30, 0x2c, 0x13,
  0x04, 0xe6, 0x99, 0x20, 0x0c, 0xcc, 0x06, 0x61, 0x80, 0x36, 0x2c, 0x0b,
  0xd3, 0x2c, 0xcb, 0xe0, 0x3c, 0xcf, 0x13, 0x6d, 0x08, 0xa4, 0x09, 0x42,
  0x26, 0x6d, 0x40, 0x16, 0xaa, 0x59, 0x96, 0xc1, 0x01, 0x36, 0x04, 0xd5,
  0x06, 0x02, 0x98, 0x2c, 0x60, 0x82, 0x20, 0x00, 0x34, 0x86, 0xa6, 0x9a,
  0xc2, 0xd2, 0xdc, 0x26, 0x08, 0x15, 0x34, 0x41, 0x18, 0x9a, 0x09, 0xc2,
  0xe0, 0x6c, 0x18, 0xb8, 0x61, 0xd8, 0x40, 0x2c, 0xda, 0xd6, 0x6d, 0x28,
  0xb0, 0x0c, 0xb8, 0xbc, 0x2a, 0x6c, 0x6c, 0x76, 0x6d, 0x2e, 0x69, 0x64,
  0x65, 0x6e, 0x74, 0x53, 0x82, 0xa0, 0x0a, 0x19, 0x9e, 0x8b, 0x5d, 0x99,
  0xdc, 0x5c, 0xda, 0x9b, 0xdb, 0x94, 0x80, 0x68, 0x42, 0x86, 0xe7, 0x62,
  0x17, 0xc6, 0x66, 0x57, 0x26, 0x37, 0x25, 0x30, 0xea, 0x90, 0xe1, 0xb9,
  0xcc, 0xa1, 0x85, 0x91, 0x95, 0xc9, 0x35, 0xbd, 0x91, 0x95, 0xb1, 0x4d,
  0x09, 0x92, 0x32, 0x64, 0x78, 0x2e, 0x72, 0x65, 0x73, 0x6f, 0x75, 0x72,
  0x63, 0x65, 0x73, 0x53, 0x02, 0xab, 0x0e, 0x19, 0x9e, 0x4b, 0x99, 0x1b,
  0x9d, 0x5c, 0x1e, 0xd4, 0x5b, 0x9a, 0x1b, 0xdd, 0xdc, 0x94, 0xc0, 0x03,
  0x79, 0x18, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00, 0x33, 0x08, 0x80, 0x1c,
  0xc4, 0xe1, 0x1c, 0x66, 0x14, 0x01, 0x3d, 0x88, 0x43, 0x38, 0x84, 0xc3,
  0x8c, 0x42, 0x80, 0x07, 0x79, 0x78, 0x07, 0x73, 0x98, 0x71, 0x0c, 0xe6,
  0x00, 0x0f, 0xed, 0x10, 0x0e, 0xf4, 0x80, 0x0e, 0x33, 0x0c, 0x42, 0x1e,
  0xc2, 0xc1, 0x1d, 0xce, 0xa1, 0x1c, 0x66, 0x30, 0x05, 0x3d, 0x88, 0x43,
  0x38, 0x84, 0x83, 0x1b, 0xcc, 0x03, 0x3d, 0xc8, 0x43, 0x3d, 0x8c, 0x03,
  0x3d, 0xcc, 0x78, 0x8c, 0x74, 0x70, 0x07, 0x7b, 0x08, 0x07, 0x79, 0x48,
  0x87, 0x70, 0x70, 0x07, 0x7a, 0x70, 0x03, 0x76, 0x78, 0x87, 0x70, 0x20,
  0x87, 0x19, 0xcc, 0x11, 0x0e, 0xec, 0x90, 0x0e, 0xe1, 0x30, 0x0f, 0x6e,
  0x30, 0x0f, 0xe3, 0xf0, 0x0e, 0xf0, 0x50, 0x0e, 0x33, 0x10, 0xc4, 0x1d,
  0xde, 0x21, 0x1c, 0xd8, 0x21, 0x1d, 0xc2, 0x61, 0x1e, 0x66, 0x30, 0x89,
  0x3b, 0xbc, 0x83, 0x3b, 0xd0, 0x43, 0x39, 0xb4, 0x03, 0x3c, 0xbc, 0x83,
  0x3c, 0x84, 0x03, 0x3b, 0xcc, 0xf0, 0x14, 0x76, 0x60, 0x07, 0x7b, 0x68,
  0x07, 0x37, 0x68, 0x87, 0x72, 0x68, 0x07, 0x37, 0x80, 0x87, 0x70, 0x90,
  0x87, 0x70, 0x60, 0x07, 0x76, 0x28, 0x07, 0x76, 0xf8, 0x05, 0x76, 0x78,
  0x87, 0x77, 0x80, 0x87, 0x5f, 0x08, 0x87, 0x71, 0x18, 0x87, 0x72, 0x98,
  0x87, 0x79, 0x98, 0x81, 0x2c, 0xee, 0xf0, 0x0e, 0xee, 0xe0, 0x0e, 0xf5,
  0xc0, 0x0e, 0xec, 0x30, 0x03, 0x62, 0xc8, 0xa1, 0x1c, 0xe4, 0xa1, 0x1c,
  0xcc, 0xa1, 0x1c, 0xe4, 0xa1, 0x1c, 0xdc, 0x61, 0x1c, 0xca, 0x21, 0x1c,
  0xc4, 0x81, 0x1d, 0xca, 0x61, 0x06, 0xd6, 0x90, 0x43, 0x39, 0xc8, 0x43,
  0x39, 0x98, 0x43, 0x39, 0xc8, 0x43, 0x39, 0xb8, 0xc3, 0x38, 0x94, 0x43,
  0x38, 0x88, 0x03, 0x3b, 0x94, 0xc3, 0x2f, 0xbc, 0x83, 0x3c, 0xfc, 0x82,
  0x3b, 0xd4, 0x03, 0x3b, 0xb0, 0xc3, 0x0c, 0xc4, 0x21, 0x07, 0x7c, 0x70,
  0x03, 0x7a, 0x28, 0x87, 0x76, 0x80, 0x87, 0x19, 0xd1, 0x43, 0x0e, 0xf8,
  0xe0, 0x06, 0xe4, 0x20, 0x0e, 0xe7, 0xe0, 0x06, 0xf6, 0x10, 0x0e, 0xf2,
  0xc0, 0x0e, 0xe1, 0x90, 0x0f, 0xef, 0x50, 0x0f, 0xf4, 0x30, 0x83, 0x81,
  0xc8, 0x01, 0x1f, 0xdc, 0x40, 0x1c, 0xe4, 0xa1, 0x1c, 0xc2, 0x61, 0x1d,
  0xdc, 0x40, 0x1c, 0xe4, 0x01, 0x00, 0x00, 0x00, 0x71, 0x20, 0x00, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x06, 0x60, 0x70, 0xac, 0x09, 0x20, 0x8d, 0x09,
  0x6c, 0xc3, 0xe5, 0x3b, 0x8f, 0x2f, 0x04, 0x54, 0x51, 0x10, 0x51, 0xe9,
  0x00, 0x43, 0x49, 0x18, 0x80, 0x80, 0xf9, 0xc8, 0x6d, 0xdb, 0x80, 0x34,
  0x5c, 0xbe, 0xf3, 0xf8, 0x42, 0x44, 0x00, 0x13, 0x11, 0x02, 0xcd, 0xb0,
  0x10, 0x46, 0x70, 0x0d, 0x97, 0xef, 0x3c, 0x7e, 0x04, 0x58, 0x1b, 0x55,
  0x14, 0x44, 0x54, 0x3a, 0xc0, 0xe0, 0x23, 0xb7, 0x6d, 0x05, 0xd8, 0x70,
  0xf9, 0xce, 0xe3, 0x47, 0x80, 0xb5, 0x51, 0x45, 0x41, 0x44, 0xec, 0xe4,
  0x44, 0x84, 0x8f, 0xdc, 0xb6, 0x05, 0x48, 0xc3, 0xe5, 0x3b, 0x8f, 0x3f,
  0x1d, 0x11, 0x01, 0x0c, 0xe2, 0xe0, 0x23, 0xb7, 0x0d, 0x00, 0x00, 0x00,
  0x61, 0x20, 0x00, 0x00, 0x29, 0x00, 0x00, 0x00, 0x13, 0x04, 0x43, 0x2c,
  0x10, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x34, 0x4a, 0x6e, 0x06,
  0xa0, 0x74, 0x03, 0xca, 0xae, 0x2c, 0x05, 0x0a, 0x53, 0x80, 0x4e, 0x19,
  0x94, 0x40, 0x11, 0x50, 0x2d, 0xa0, 0x12, 0x00, 0x23, 0x06, 0x09, 0x00,
  0x82, 0x60, 0xd0, 0x6c, 0x0b, 0x41, 0x51, 0xcf, 0x88, 0x41, 0x02, 0x80,
  0x20, 0x18, 0x34, 0x1c, 0x43, 0x54, 0x15, 0x34, 0x62, 0x60, 0x00, 0x20,
  0x08, 0x06, 0xc4, 0xb7, 0x58, 0x23, 0x06, 0x07, 0x00, 0x82, 0x60, 0xa0,
  0x7c, 0x8b, 0x70, 0x8d, 0x26, 0x04, 0x41, 0x05, 0x03, 0x8c, 0x26, 0x0c,
  0xc0, 0x70, 0x83, 0x10, 0x90, 0xc1, 0x2c, 0x43, 0x20, 0x04, 0x23, 0x06,
  0x0a, 0x00, 0x82, 0x60, 0x00, 0x89, 0xc1, 0x83, 0x0c, 0x5c, 0xa3, 0x8d,
  0x26, 0x04, 0xc0, 0x68, 0x82, 0x10, 0x9c, 0x50, 0xe3, 0x84, 0x1a, 0x15,
  0x3c, 0x57, 0x43, 0xb0, 0x16, 0x40, 0x20, 0xb8, 0x60, 0x40, 0x09, 0x13,
  0x5e, 0x30, 0x60, 0xc4, 0xc0, 0x01, 0x40, 0x10, 0x0c, 0x24, 0x36, 0xc8,
  0x26, 0xe7, 0x0c, 0x86, 0xa0, 0xeb, 0xb0, 0x32, 0x98, 0x25, 0x10, 0x10,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
