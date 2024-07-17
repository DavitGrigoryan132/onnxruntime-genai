// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

NS_ASSUME_NONNULL_BEGIN


@class OGASpan;
@class OGATensor;
@class OGASequences;
@class OGAGeneratorParams;
@class OGATokenizerStream;

typedef NS_ENUM(NSInteger, OGAElementType) {
  OGAElementTypeUndefined,
  OGAElementTypeFloat32,  // maps to c type float
  OGAElementTypeUint8,    // maps to c type uint8_t
  OGAElementTypeInt8,     // maps to c type int8_t
  OGAElementTypeUint16,   // maps to c type uint16_t
  OGAElementTypeInt16,    // maps to c type int16_t
  OGAElementTypeInt32,    // maps to c type int32_t
  OGAElementTypeInt64,    // maps to c type int64_t
  OGAElementTypeString,   // string type (not currently supported by Oga)
  OGAElementTypeBool,     // maps to c type bool
  OGAElementTypeFloat16,  // IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
  OGAElementTypeFloat64,  // maps to c type double
  OGAElementTypeUint32,   // maps to c type uint32_t
  OGAElementTypeUint64,   // maps to c type uint64_t
};


@interface OGAModel : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithPath:(NSString *)path
                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;


- (nullable OGASequences *)generate:(OGAGeneratorParams *)params
                              error:(NSError **)error;

@end


@interface OGATokenizer : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithModel:(OGAModel *)model
                    error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (nullable OGASequences *)encode:(NSString *)str
                            error:(NSError **)error;

- (nullable NSString *)decode:(OGASpan *)data
                        error:(NSError **)error;

@end

@interface OGATokenizerStream: NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithTokenizer:(OGATokenizer *)tokenizer
                        error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (nullable NSString *)decode:(int32_t)token
                        error:(NSError **)error;
@end;


@interface OGASpan : NSObject

- (instancetype)init NS_UNAVAILABLE;

- (int32_t)last;

@end

@interface OGASequences : NSObject

- (instancetype)init NS_UNAVAILABLE;

- (size_t)count;
- (nullable OGASpan *)sequenceAtIndex:(size_t)index;

@end

@interface OGAGeneratorParams : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithModel:(OGAModel *)model
                    error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (BOOL)setInputs:(OGANamedTensors *)namedTensors
            error:(NSError **)error;

- (BOOL)setInputSequences:(OGASequences *)sequences
                    error:(NSError **)error;

- (BOOL)setModelInput:(NSString *)name
               tensor:(OGATensor* tensor)
                error:(NSError **)error;

- (BOOL)setSearchOption:(NSString *)key
            doubleValue:(double)value
                  error:(NSError **)error;

- (BOOL)setSearchOption:(NSString *)key
              boolValue:(BOOL)value
                  error:(NSError **)error;
@end

@interface OGAGenerator : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithModel:(OGAModel *)model
                   params:(OGAGeneratorParams *)params
                    error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (BOOL)isDone;
- (void)computeLogits;
- (void)generateNextToken;

- (nullable OGASpan *)sequenceAtIndex:(size_t) index;

@end

@interface OGAImages : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithPath:(NSString *)path
                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
