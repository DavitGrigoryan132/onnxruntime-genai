#if 0
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
; shader hash: 576a95fc9e51cfae8ad3d2f3b9c07550
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
;   [4 x i8] (type annotation not present)
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
%"class.RWStructuredBuffer<int>" = type { i32 }
%Constants = type { i32, i32, i32 }

define void @CSMainContinuous() {
  %1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 2, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %3 = call i32 @dx.op.threadId.i32(i32 93, i32 0)  ; ThreadId(component)
  %4 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %2, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %5 = extractvalue %dx.types.CBufRet.i32 %4, 1
  %6 = add i32 %5, %3
  %7 = extractvalue %dx.types.CBufRet.i32 %4, 0
  %8 = icmp ult i32 %6, %7
  br i1 %8, label %9, label %13

; <label>:9                                       ; preds = %0
  %10 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %2, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %11 = extractvalue %dx.types.CBufRet.i32 %10, 2
  %12 = add i32 %11, %6
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %1, i32 %6, i32 0, i32 %12, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  br label %13

; <label>:13                                      ; preds = %9, %0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.threadId.i32(i32, i32) #0

; Function Attrs: nounwind
declare void @dx.op.rawBufferStore.i32(i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32, %dx.types.Handle, i32) #2

; Function Attrs: nounwind readonly
declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1) #2

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }

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
!6 = !{i32 0, %"class.RWStructuredBuffer<int>"* undef, !"", i32 0, i32 0, i32 1, i32 12, i1 false, i1 false, i1 false, !7}
!7 = !{i32 1, i32 4}
!8 = !{!9}
!9 = !{i32 0, %Constants* undef, !"", i32 0, i32 0, i32 1, i32 12, null}
!10 = !{void ()* @CSMainContinuous, !"CSMainContinuous", null, !4, !11}
!11 = !{i32 0, i64 16, i32 4, !12}
!12 = !{i32 256, i32 1, i32 1}

#endif

const unsigned char g_CSMainContinuous[] = {
  0x44, 0x58, 0x42, 0x43, 0x50, 0xed, 0xff, 0xc8, 0x8b, 0xa2, 0xa2, 0xca,
  0xb2, 0x4b, 0x2e, 0x0b, 0xf8, 0x97, 0xbe, 0x7c, 0x01, 0x00, 0x00, 0x00,
  0xc0, 0x06, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0xe8, 0x00, 0x00, 0x00, 0x04, 0x01, 0x00, 0x00, 0x53, 0x46, 0x49, 0x30,
  0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
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
  0x00, 0x00, 0x00, 0x00, 0x57, 0x6a, 0x95, 0xfc, 0x9e, 0x51, 0xcf, 0xae,
  0x8a, 0xd3, 0xd2, 0xf3, 0xb9, 0xc0, 0x75, 0x50, 0x44, 0x58, 0x49, 0x4c,
  0xb4, 0x05, 0x00, 0x00, 0x62, 0x00, 0x05, 0x00, 0x6d, 0x01, 0x00, 0x00,
  0x44, 0x58, 0x49, 0x4c, 0x02, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x9c, 0x05, 0x00, 0x00, 0x42, 0x43, 0xc0, 0xde, 0x21, 0x0c, 0x00, 0x00,
  0x64, 0x01, 0x00, 0x00, 0x0b, 0x82, 0x20, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x07, 0x81, 0x23, 0x91, 0x41, 0xc8, 0x04, 0x49,
  0x06, 0x10, 0x32, 0x39, 0x92, 0x01, 0x84, 0x0c, 0x25, 0x05, 0x08, 0x19,
  0x1e, 0x04, 0x8b, 0x62, 0x80, 0x14, 0x45, 0x02, 0x42, 0x92, 0x0b, 0x42,
  0xa4, 0x10, 0x32, 0x14, 0x38, 0x08, 0x18, 0x4b, 0x0a, 0x32, 0x52, 0x88,
  0x48, 0x90, 0x14, 0x20, 0x43, 0x46, 0x88, 0xa5, 0x00, 0x19, 0x32, 0x42,
  0xe4, 0x48, 0x0e, 0x90, 0x91, 0x22, 0xc4, 0x50, 0x41, 0x51, 0x81, 0x8c,
  0xe1, 0x83, 0xe5, 0x8a, 0x04, 0x29, 0x46, 0x06, 0x51, 0x18, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x1b, 0x8c, 0xe0, 0xff, 0xff, 0xff, 0xff, 0x07,
  0x40, 0x02, 0xa8, 0x0d, 0x84, 0xf0, 0xff, 0xff, 0xff, 0xff, 0x03, 0x20,
  0x6d, 0x30, 0x86, 0xff, 0xff, 0xff, 0xff, 0x1f, 0x00, 0x09, 0xa8, 0x00,
  0x49, 0x18, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x13, 0x82, 0x60, 0x42,
  0x20, 0x4c, 0x08, 0x06, 0x00, 0x00, 0x00, 0x00, 0x89, 0x20, 0x00, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0x32, 0x22, 0x48, 0x09, 0x20, 0x64, 0x85, 0x04,
  0x93, 0x22, 0xa4, 0x84, 0x04, 0x93, 0x22, 0xe3, 0x84, 0xa1, 0x90, 0x14,
  0x12, 0x4c, 0x8a, 0x8c, 0x0b, 0x84, 0xa4, 0x4c, 0x10, 0x60, 0x23, 0x00,
  0x25, 0x00, 0x14, 0xe6, 0x08, 0xc0, 0xa0, 0x0c, 0x63, 0x0c, 0x22, 0x73,
  0x04, 0x08, 0x99, 0x7b, 0x86, 0xcb, 0x9f, 0xb0, 0x87, 0x90, 0xfc, 0x10,
  0x68, 0x86, 0x85, 0x40, 0xc1, 0x29, 0x0b, 0x18, 0x68, 0x8c, 0x31, 0xc6,
  0x30, 0x83, 0xd2, 0x51, 0xc3, 0xe5, 0x4f, 0xd8, 0x43, 0x48, 0x3e, 0xb7,
  0x51, 0xc5, 0x4a, 0x4c, 0x3e, 0x72, 0xdb, 0x88, 0x18, 0x63, 0x8c, 0x42,
  0xac, 0x81, 0x06, 0xb1, 0x39, 0x82, 0xa0, 0x18, 0x68, 0x98, 0x31, 0x1c,
  0xbd, 0x81, 0x80, 0x99, 0xba, 0x71, 0x60, 0x87, 0x70, 0x98, 0x87, 0x79,
  0x70, 0x03, 0x59, 0xb8, 0x85, 0x59, 0xa0, 0x07, 0x79, 0xa8, 0x87, 0x71,
  0xa0, 0x87, 0x7a, 0x90, 0x87, 0x72, 0x20, 0x07, 0x51, 0xa8, 0x07, 0x73,
  0x30, 0x87, 0x72, 0x90, 0x07, 0x3e, 0x48, 0x07, 0x77, 0xa0, 0x07, 0x3f,
  0x40, 0xc1, 0x20, 0x79, 0x09, 0xe7, 0x34, 0xd2, 0x04, 0x34, 0x93, 0x84,
  0x86, 0x31, 0x06, 0xd1, 0x39, 0x02, 0x50, 0x98, 0x02, 0x00, 0x00, 0x00,
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
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x79, 0x12, 0x20,
  0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0xf2, 0x30,
  0x40, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0xe4,
  0x79, 0x80, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20,
  0x0b, 0x04, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x32, 0x1e, 0x98, 0x14,
  0x19, 0x11, 0x4c, 0x90, 0x8c, 0x09, 0x26, 0x47, 0xc6, 0x04, 0x43, 0x1a,
  0x25, 0x50, 0x04, 0xe5, 0x50, 0x0c, 0x23, 0x00, 0x85, 0x51, 0x08, 0x05,
  0x48, 0x40, 0x6e, 0x04, 0x80, 0x6c, 0x81, 0x50, 0x9d, 0x01, 0xa0, 0x39,
  0x03, 0x00, 0x00, 0x00, 0x79, 0x18, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
  0x1a, 0x03, 0x4c, 0x90, 0x46, 0x02, 0x13, 0xc4, 0x8e, 0x0c, 0x6f, 0xec,
  0xed, 0x4d, 0x0c, 0x24, 0xc6, 0xe5, 0xc6, 0x45, 0x66, 0x06, 0x06, 0xc7,
  0xe5, 0x06, 0x04, 0xc5, 0x26, 0xa7, 0xac, 0x86, 0xa6, 0x4c, 0x26, 0x07,
  0x26, 0x65, 0x43, 0x10, 0x4c, 0x10, 0x86, 0x62, 0x82, 0x30, 0x18, 0x1b,
  0x84, 0x81, 0x98, 0x20, 0x0c, 0xc7, 0x06, 0x61, 0x30, 0x28, 0x8c, 0xcd,
  0x4d, 0x10, 0x06, 0x64, 0xc3, 0x80, 0x24, 0xc4, 0x04, 0x61, 0x48, 0x26,
  0x08, 0x13, 0x44, 0x60, 0x82, 0x30, 0x28, 0x13, 0x04, 0xa7, 0x99, 0x20,
  0x0c, 0xcb, 0x06, 0x61, 0x80, 0x36, 0x2c, 0x0b, 0xd3, 0x2c, 0xcb, 0xe0,
  0x3c, 0xcf, 0x13, 0x6d, 0x08, 0xa4, 0x09, 0x42, 0xf5, 0x6c, 0x40, 0x16,
  0xaa, 0x59, 0x96, 0xc1, 0x01, 0x36, 0x04, 0xd5, 0x06, 0x02, 0x98, 0x2c,
  0x60, 0x82, 0x20, 0x00, 0x84, 0x86, 0xa6, 0x9a, 0xc2, 0xd2, 0xdc, 0x86,
  0xde, 0xdc, 0xe8, 0xd2, 0xdc, 0xea, 0xde, 0xea, 0xe6, 0x26, 0x08, 0x96,
  0x33, 0x41, 0x18, 0x98, 0x0d, 0xc3, 0x36, 0x0c, 0x1b, 0x88, 0x45, 0x83,
  0xb8, 0x0d, 0x05, 0x96, 0x01, 0x57, 0x57, 0x85, 0x8d, 0xcd, 0xae, 0xcd,
  0x25, 0x8d, 0xac, 0xcc, 0x8d, 0x6e, 0x4a, 0x10, 0x54, 0x21, 0xc3, 0x73,
  0xb1, 0x2b, 0x93, 0x9b, 0x4b, 0x7b, 0x73, 0x9b, 0x12, 0x10, 0x4d, 0xc8,
  0xf0, 0x5c, 0xec, 0xc2, 0xd8, 0xec, 0xca, 0xe4, 0xa6, 0x04, 0x46, 0x1d,
  0x32, 0x3c, 0x97, 0x39, 0xb4, 0x30, 0xb2, 0x32, 0xb9, 0xa6, 0x37, 0xb2,
  0x32, 0xb6, 0x29, 0x41, 0x52, 0x86, 0x0c, 0xcf, 0x45, 0xae, 0x6c, 0xee,
  0xad, 0x4e, 0x6e, 0xac, 0x6c, 0x6e, 0x4a, 0x60, 0xd5, 0x21, 0xc3, 0x73,
  0x29, 0x73, 0xa3, 0x93, 0xcb, 0x83, 0x7a, 0x4b, 0x73, 0xa3, 0x9b, 0x9b,
  0x12, 0x74, 0x00, 0x00, 0x79, 0x18, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00,
  0x33, 0x08, 0x80, 0x1c, 0xc4, 0xe1, 0x1c, 0x66, 0x14, 0x01, 0x3d, 0x88,
  0x43, 0x38, 0x84, 0xc3, 0x8c, 0x42, 0x80, 0x07, 0x79, 0x78, 0x07, 0x73,
  0x98, 0x71, 0x0c, 0xe6, 0x00, 0x0f, 0xed, 0x10, 0x0e, 0xf4, 0x80, 0x0e,
  0x33, 0x0c, 0x42, 0x1e, 0xc2, 0xc1, 0x1d, 0xce, 0xa1, 0x1c, 0x66, 0x30,
  0x05, 0x3d, 0x88, 0x43, 0x38, 0x84, 0x83, 0x1b, 0xcc, 0x03, 0x3d, 0xc8,
  0x43, 0x3d, 0x8c, 0x03, 0x3d, 0xcc, 0x78, 0x8c, 0x74, 0x70, 0x07, 0x7b,
  0x08, 0x07, 0x79, 0x48, 0x87, 0x70, 0x70, 0x07, 0x7a, 0x70, 0x03, 0x76,
  0x78, 0x87, 0x70, 0x20, 0x87, 0x19, 0xcc, 0x11, 0x0e, 0xec, 0x90, 0x0e,
  0xe1, 0x30, 0x0f, 0x6e, 0x30, 0x0f, 0xe3, 0xf0, 0x0e, 0xf0, 0x50, 0x0e,
  0x33, 0x10, 0xc4, 0x1d, 0xde, 0x21, 0x1c, 0xd8, 0x21, 0x1d, 0xc2, 0x61,
  0x1e, 0x66, 0x30, 0x89, 0x3b, 0xbc, 0x83, 0x3b, 0xd0, 0x43, 0x39, 0xb4,
  0x03, 0x3c, 0xbc, 0x83, 0x3c, 0x84, 0x03, 0x3b, 0xcc, 0xf0, 0x14, 0x76,
  0x60, 0x07, 0x7b, 0x68, 0x07, 0x37, 0x68, 0x87, 0x72, 0x68, 0x07, 0x37,
  0x80, 0x87, 0x70, 0x90, 0x87, 0x70, 0x60, 0x07, 0x76, 0x28, 0x07, 0x76,
  0xf8, 0x05, 0x76, 0x78, 0x87, 0x77, 0x80, 0x87, 0x5f, 0x08, 0x87, 0x71,
  0x18, 0x87, 0x72, 0x98, 0x87, 0x79, 0x98, 0x81, 0x2c, 0xee, 0xf0, 0x0e,
  0xee, 0xe0, 0x0e, 0xf5, 0xc0, 0x0e, 0xec, 0x30, 0x03, 0x62, 0xc8, 0xa1,
  0x1c, 0xe4, 0xa1, 0x1c, 0xcc, 0xa1, 0x1c, 0xe4, 0xa1, 0x1c, 0xdc, 0x61,
  0x1c, 0xca, 0x21, 0x1c, 0xc4, 0x81, 0x1d, 0xca, 0x61, 0x06, 0xd6, 0x90,
  0x43, 0x39, 0xc8, 0x43, 0x39, 0x98, 0x43, 0x39, 0xc8, 0x43, 0x39, 0xb8,
  0xc3, 0x38, 0x94, 0x43, 0x38, 0x88, 0x03, 0x3b, 0x94, 0xc3, 0x2f, 0xbc,
  0x83, 0x3c, 0xfc, 0x82, 0x3b, 0xd4, 0x03, 0x3b, 0xb0, 0xc3, 0x0c, 0xc4,
  0x21, 0x07, 0x7c, 0x70, 0x03, 0x7a, 0x28, 0x87, 0x76, 0x80, 0x87, 0x19,
  0xd1, 0x43, 0x0e, 0xf8, 0xe0, 0x06, 0xe4, 0x20, 0x0e, 0xe7, 0xe0, 0x06,
  0xf6, 0x10, 0x0e, 0xf2, 0xc0, 0x0e, 0xe1, 0x90, 0x0f, 0xef, 0x50, 0x0f,
  0xf4, 0x30, 0x83, 0x81, 0xc8, 0x01, 0x1f, 0xdc, 0x40, 0x1c, 0xe4, 0xa1,
  0x1c, 0xc2, 0x61, 0x1d, 0xdc, 0x40, 0x1c, 0xe4, 0x01, 0x00, 0x00, 0x00,
  0x71, 0x20, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00, 0x06, 0x00, 0x71, 0xac,
  0x09, 0x20, 0x0d, 0xe7, 0x34, 0x13, 0xd2, 0x50, 0x0e, 0x25, 0xd9, 0xc0,
  0x36, 0x5c, 0xbe, 0xf3, 0xf8, 0x42, 0x40, 0x15, 0x05, 0x11, 0x95, 0x0e,
  0x30, 0x94, 0x84, 0x01, 0x08, 0x98, 0x8f, 0xdc, 0xb6, 0x11, 0x48, 0xc3,
  0xe5, 0x3b, 0x8f, 0x2f, 0x44, 0x04, 0x30, 0x11, 0x21, 0xd0, 0x0c, 0x0b,
  0x61, 0x02, 0xd8, 0x70, 0xf9, 0xce, 0xe3, 0x47, 0x80, 0xb5, 0x51, 0x45,
  0x41, 0x44, 0xec, 0xe4, 0x44, 0x84, 0x8f, 0xdc, 0xb6, 0x05, 0x48, 0xc3,
  0xe5, 0x3b, 0x8f, 0x3f, 0x1d, 0x11, 0x01, 0x0c, 0xe2, 0xe0, 0x23, 0xb7,
  0x0d, 0x00, 0x00, 0x00, 0x61, 0x20, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x13, 0x04, 0x43, 0x2c, 0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x34, 0x66, 0x00, 0x4a, 0xae, 0xec, 0x4a, 0x37, 0xa0, 0x30, 0x05, 0xc8,
  0x94, 0x40, 0x11, 0x00, 0x23, 0x06, 0x09, 0x00, 0x82, 0x60, 0xf0, 0x50,
  0x86, 0xf0, 0x3c, 0xcb, 0x88, 0x41, 0x02, 0x80, 0x20, 0x18, 0x3c, 0xd5,
  0x21, 0x40, 0x10, 0x33, 0x62, 0x60, 0x00, 0x20, 0x08, 0x06, 0x44, 0x66,
  0x44, 0x23, 0x06, 0x07, 0x00, 0x82, 0x60, 0xc0, 0x60, 0x88, 0x20, 0x8d,
  0x26, 0x04, 0x41, 0x05, 0x03, 0x8c, 0x26, 0x0c, 0xc0, 0x70, 0x83, 0x10,
  0x90, 0xc1, 0x2c, 0x43, 0x20, 0x04, 0x23, 0x06, 0x07, 0x00, 0x82, 0x60,
  0xc0, 0x74, 0xcd, 0x71, 0x8d, 0x26, 0x04, 0x42, 0x05, 0x05, 0x8c, 0x18,
  0x38, 0x00, 0x08, 0x82, 0x41, 0x12, 0x06, 0xce, 0x62, 0x68, 0x81, 0x24,
  0x49, 0x0d, 0x36, 0x4b, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
