/**
 * @file fractal.h
 * @brief Public interface for the PIFS Fractal Image Compression engine.
 *
 * Exposes the core structures and functions needed to compress and decompress
 * images using an adaptive Quadtree, Fisher Classification, and YCbCr processing.
 */
#ifndef FRACTAL_H
#define FRACTAL_H

#include <stdint.h>

/**
 * @brief Configuration parameters for tuning the compressor.
 */
typedef struct {
    int min_r_size;         // Minimum quadtree block size (e.g. 4)
    int max_r_size;         // Maximum quadtree block size (e.g. 32)
    float error_threshold;  // Target MSE tolerance before splitting (e.g. 20.0)
    int search_window;      // Search radius for matching domain blocks (e.g. 127)
} CompressParams;

/**
 * @brief A single PIFS affine transformation for an adaptively sized range block.
 * Represents the geometric mapping and color shift needed to construct a block.
 */
typedef struct {
    uint16_t rx, ry;     // Range block spatial position (Top-Left X, Y)
    uint16_t r_size;     // Edge size of the square range block
    uint16_t x, y;       // Source Domain block position
    uint8_t isometry;    // 3-bit flag representing 1 of 8 rotations/reflections
    float contrast_y, brightness_y;      // Luma affine shift (s, o)
    float contrast_cb, brightness_cb;    // Blue-chroma affine shift
    float contrast_cr, brightness_cr;    // Red-chroma affine shift
} FractalTransform;

/**
 * @brief The fully compressed representation of an image.
 */
typedef struct {
    int width;           // Original image width in pixels
    int height;          // Original image height in pixels
    int min_r_size;      // Copied from CompressParams for decompression tracking
    int max_r_size;      
    int num_transforms;  // Total number of discrete quadtree blocks
    FractalTransform* transforms; // Heap array of all blocks
} FractalImage;

/**
 * @brief Compresses a raw RGB image array into a Fractal representation.
 * @param pixels 1D array of 8-bit RGB pixels (w * h * 3 size).
 * @param width Width of the image.
 * @param height Height of the image.
 * @param params Tuning configuration.
 * @return Heap-allocated FractalImage object. Caller must free it.
 */
FractalImage* compress_image(const uint8_t* pixels, int width, int height, CompressParams params);

/**
 * @brief Callback invoked at the end of each decompression iteration.
 * Useful for rendering GIFs or showing progress in real-time.
 */
typedef void (*FrameCallback)(int iter, const uint8_t* img, int w, int h, void* user_data);

/**
 * @brief Decompresses a Fractal image back into an RGB pixel array iteratively.
 * @param fi The parsed fractal image to decompress.
 * @param max_iterations Maximum fixed-point contraction iterations.
 * @param cb Optional callback function.
 * @param user_data Custom pointer passed to the callback.
 * @return 1D array of 8-bit RGB pixels (w * h * 3 size).
 */
uint8_t* decompress_image_cb(const FractalImage* fi, int max_iterations, FrameCallback cb, void* user_data);

/**
 * @brief Wrapper for decompress_image_cb without using a callback.
 */
uint8_t* decompress_image(const FractalImage* fi, int max_iterations);

/**
 * @brief Safely frees memory allocated by the compression engine.
 */
void free_fractal_image(FractalImage* fi);

/**
 * @brief Serializes a FractalImage into the highly optimized FRC3 bit-packed binary format.
 */
int save_fractal_image(const FractalImage* fi, const char* filename);

/**
 * @brief Parses an FRC3 binary file back into a FractalImage.
 */
FractalImage* load_fractal_image(const char* filename);

#endif // FRACTAL_H
