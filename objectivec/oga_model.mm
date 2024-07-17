#import "ort_genai_objc.h"
#import "error_utils.h"
#import "oga_internal.h"

@implementation OGAModel {
    std::unique_ptr<OgaModel> _model;
}


- (nullable)initWithPath:(NSString *)path
                   error:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }

    try {
        _model = OgaModel::Create(path.UTF8String);
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGASequences *)generate:(OGAGeneratorParams *)params
                              error:(NSError **)error {
    try {
        std::unique_ptr<OgaSequences> output_sequences =  _model->Generate([params CXXAPIOgaGeneratorParams]);
        return [[OGASequences alloc] initWithNativeSeqquences:std::move(output_sequences)];
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (const OgaModel&)CXXAPIOgaModel {
    return *(_model.get());
}

@end