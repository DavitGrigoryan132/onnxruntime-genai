// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_genai_objc.h"
#import "error_utils.h"
#import "oga_internal.h"

@implementation OGATokenizer {
    std::unique_ptr<OgaTokenizer> _tokenizer;
}


- (nullable)initWithModel:(OGAModel *)model
                    error:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }

    try {
        _tokenizer = OgaTokenizer::Create([model CXXAPIOgaModel]);
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGASequences *)encode:(NSString *)str
                            error:(NSError **)error {
    OGASequences *sequences = [[OGASequences alloc] initWithError:error];
    if (*error) {
        return nil;
    }
    try {
        _tokenizer->Encode([str UTF8String], [sequences CXXAPIOgaSequences]);
        return sequences;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSString *)decode:(OGASpan *) data
                        error:(NSError **)error {
    try {
        OgaString result = _tokenizer->Decode(data.pointer, data.size);
        return [NSString stringWithUTF8String:result];
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (const OgaTokenizer&)CXXAPIOgaTokenizer {
    return *(_tokenizer.get());
}

@end

@implementation OGATokenizerStream {
    std::unique_ptr<OgaTokenizerStream> _stream;
}

- (nullable)initWithTokenizer:(OGATokenizer *)tokenizer
                        error:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }

    try {
        _stream = OgaTokenizerStream::Create([tokenizer CXXAPIOgaTokenizer]);
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSString *)decode:(int32_t)token
                        error:(NSError **)error {
    try {
        return [NSString stringWithUTF8String:_stream->Decode(token)];
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

@end
