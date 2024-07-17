#import "ort_genai_objc.h"
#import "error_utils.h"
#import "oga_internal.h"

@implementation OGAImages {
    std::unique_ptr<OgaImages> _images;
}


- (nullable)initWithPath:(NSString *)path
                   error:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }

    try {
        _images = OgaImages::Load(path.UTF8String);
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (OgaImages *)CXXAPIOgaImages {
    return _images.get();
}

@end