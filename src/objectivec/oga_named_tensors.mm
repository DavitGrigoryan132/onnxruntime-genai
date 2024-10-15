// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGANamedTensors {
  std::unique_ptr<OgaNamedTensors> _tensor;
}

- (instancetype)initWithNativePointer:(std::unique_ptr<OgaNamedTensors>)ptr;
{
  _tensor = std::move(ptr);
  return self;
}

- (OgaNamedTensors&)CXXAPIOgaNamedTensors {
  return *(_tensor.get());
}

@end