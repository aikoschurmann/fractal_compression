/**
 * @file main.c
 * @brief CLI entry point for compressing, decompressing, rendering GIFs, and benchmarking.
 * Wraps stb_image.h to handle standard format generic parsing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fractal.h"

// Third-party single-header libraries for standard image formats
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void print_help(const char* prog_name) {
    printf("Usage:\n");
    printf("  %s compress -i <in_img> -o <out_frc> [options]\n", prog_name);
    printf("  %s decompress -i <in_frc> -o <out_img> [options]\n", prog_name);
    printf("  %s gif -i <in_frc> -o <out_gif> [options]\n", prog_name);
    printf("  %s bench -i <in_img> [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -d <int>      Downsample factor (default 1)\n");
    printf("  -m <int>      Minimum quadtree block size (default 4)\n");
    printf("  -M <int>      Maximum quadtree block size (default 32)\n");
    printf("  -t <float>    MSE allowed before quadtree splits (default 20.0)\n");
    printf("  -w <int>      Search window distance (default 128)\n");
    printf("  -n <int>      Max iterations for decompress/gif (default 20)\n");
    printf("  --fast        Preset: Fast compression\n");
    printf("  --best        Preset: High quality compression\n");
}

/**
 * @brief Downsamples an RGB image prior to fractal encoding.
 * Massive images (e.g. 8k photos) take too long to search heuristically.
 * Pre-downsampling shrinks the global dataset, heavily reducing search times.
 */
uint8_t* downsample_image(const uint8_t* src, int w, int h, int factor, int* out_w, int* out_h) {
    if (factor <= 1) {
        *out_w = w;
        *out_h = h;
        uint8_t* copy = malloc(w * h * 3);
        memcpy(copy, src, w * h * 3);
        return copy;
    }
    
    *out_w = w / factor;
    *out_h = h / factor;
    uint8_t* dst = malloc((*out_w) * (*out_h) * 3);
    
    for (int y = 0; y < *out_h; y++) {
        for (int x = 0; x < *out_w; x++) {
            long sum_r = 0, sum_g = 0, sum_b = 0;
            // Accumulate RGB average over the Box Filter
            for (int dy = 0; dy < factor; dy++) {
                for (int dx = 0; dx < factor; dx++) {
                    int px = x * factor + dx;
                    int py = y * factor + dy;
                    sum_r += src[(py * w + px) * 3 + 0];
                    sum_g += src[(py * w + px) * 3 + 1];
                    sum_b += src[(py * w + px) * 3 + 2];
                }
            }
            dst[(y * (*out_w) + x) * 3 + 0] = (uint8_t)(sum_r / (factor * factor));
            dst[(y * (*out_w) + x) * 3 + 1] = (uint8_t)(sum_g / (factor * factor));
            dst[(y * (*out_w) + x) * 3 + 2] = (uint8_t)(sum_b / (factor * factor));
        }
    }
    return dst;
}

/**
 * @brief Storage for the GIF rendering callback.
 */
struct CbData {
    const char* prefix;
    int last_iter;
};

/**
 * @brief Receives the in-progress Banach Fixed Point image directly from the engine memory.
 * Spits it out to disk as an independent .png frame.
 */
void frame_cb(int iter, const uint8_t* img, int w, int h, void* user_data) {
    struct CbData* data = (struct CbData*)user_data;
    char filename[512];
    snprintf(filename, sizeof(filename), "%s_%03d.png", data->prefix, iter);
    stbi_write_png(filename, w, h, 3, img, w * 3);
    printf("Saved frame: %s\n", filename);
    data->last_iter = iter;
}

/**
 * @brief Utility function to get the size of a file in bytes.
 */
long get_file_size(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fclose(f);
    return size;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }

    const char* command = argv[1];

    // =========================================================================
    // COMMAND: COMPRESS
    // =========================================================================
    if (strcmp(command, "compress") == 0) {
        const char* in_file = NULL;
        const char* out_file = NULL;
        int downsample_factor = 1;
        CompressParams params = {4, 32, 20.0f, 128};
        
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "-i") == 0 && i+1 < argc) in_file = argv[++i];
            else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_file = argv[++i];
            else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) downsample_factor = atoi(argv[++i]);
            else if (strcmp(argv[i], "-m") == 0 && i+1 < argc) params.min_r_size = atoi(argv[++i]);
            else if (strcmp(argv[i], "-M") == 0 && i+1 < argc) params.max_r_size = atoi(argv[++i]);
            else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) params.error_threshold = atof(argv[++i]);
            else if (strcmp(argv[i], "-w") == 0 && i+1 < argc) params.search_window = atoi(argv[++i]);
            else if (strcmp(argv[i], "--fast") == 0) { params.error_threshold = 40.0f; params.search_window = 64; params.max_r_size = 16; }
            else if (strcmp(argv[i], "--best") == 0) { params.error_threshold = 10.0f; params.search_window = 255; params.min_r_size = 2; }
        }
        
        if (!in_file || !out_file) { print_help(argv[0]); return 1; }

        int w, h, channels;
        uint8_t* pixels = stbi_load(in_file, &w, &h, &channels, 3);
        if (!pixels) {
            fprintf(stderr, "Error: Failed to load image '%s'\n", in_file);
            return 1;
        }

        int new_w, new_h;
        uint8_t* ds_pixels = downsample_image(pixels, w, h, downsample_factor, &new_w, &new_h);
        
        // Keep original pixels around for PSNR calculation if no downsampling
        if (downsample_factor > 1) {
            stbi_image_free(pixels);
            pixels = NULL;
        }

        printf("Compressing '%s' (%dx%d downsampled to %dx%d) RGB...\n", in_file, w, h, new_w, new_h);
        printf("Params: min_size=%d, max_size=%d, threshold=%.1f, search_window=%d\n",
               params.min_r_size, params.max_r_size, params.error_threshold, params.search_window);
        
        FractalImage* fi = compress_image(ds_pixels, new_w, new_h, params);

        if (fi && save_fractal_image(fi, out_file)) {
            printf("\nSuccessfully saved compressed data to '%s'\n", out_file);
            
            // PSNR Calculation
            printf("Verifying quality (calculating PSNR)...\n");
            uint8_t* decompressed = decompress_image(fi, 20);
            double mse = 0;
            const uint8_t* ref_pixels = (downsample_factor > 1) ? ds_pixels : pixels;
            for (int i = 0; i < new_w * new_h * 3; i++) {
                double diff = ref_pixels[i] - decompressed[i];
                mse += diff * diff;
            }
            mse /= (new_w * new_h * 3);
            double psnr = 10.0 * log10((255.0 * 255.0) / (mse + 1e-9));
            printf("Quality: MSE = %.2f, PSNR = %.2f dB\n", mse, psnr);
            free(decompressed);
        } else {
            fprintf(stderr, "\nError: Failed to compress or save to '%s'\n", out_file);
        }

        free(ds_pixels);
        if (pixels) stbi_image_free(pixels);
        free_fractal_image(fi);
    }
    
    // =========================================================================
    // COMMAND: DECOMPRESS
    // =========================================================================
    else if (strcmp(command, "decompress") == 0) {
        const char* in_file = NULL;
        const char* out_file = NULL;
        int max_iters = 20;

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "-i") == 0 && i+1 < argc) in_file = argv[++i];
            else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_file = argv[++i];
            else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) max_iters = atoi(argv[++i]);
        }
        if (!in_file || !out_file) { print_help(argv[0]); return 1; }

        FractalImage* fi = load_fractal_image(in_file);
        if (!fi) {
            fprintf(stderr, "Error: Failed to load fractal data from '%s'\n", in_file);
            return 1;
        }

        printf("Decompressing '%s' to '%s' (max %d iterations)...\n", in_file, out_file, max_iters);
        uint8_t* img = decompress_image(fi, max_iters);
        stbi_write_png(out_file, fi->width, fi->height, 3, img, fi->width * 3);
        
        free(img);
        free_fractal_image(fi);
        printf("Decompression complete.\n");
    }
    
    // =========================================================================
    // COMMAND: GIF
    // =========================================================================
    else if (strcmp(command, "gif") == 0) {
        const char* in_file = NULL;
        const char* out_file = NULL;
        int max_iters = 20;

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "-i") == 0 && i+1 < argc) in_file = argv[++i];
            else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) out_file = argv[++i];
            else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) max_iters = atoi(argv[++i]);
        }
        if (!in_file || !out_file) { print_help(argv[0]); return 1; }

        FractalImage* fi = load_fractal_image(in_file);
        if (!fi) {
            fprintf(stderr, "Error: Failed to load fractal data from '%s'\n", in_file);
            return 1;
        }

        printf("Generating frames for '%s' (max %d iterations)...\n", in_file, max_iters);
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "frame_tmp_%d", rand());
        struct CbData cb_data = {prefix, 0};

        uint8_t* final_img = decompress_image_cb(fi, max_iters, frame_cb, &cb_data);
        free(final_img);
        free_fractal_image(fi);
        
        printf("\nAll frames generated.\n");
        printf("Stitching frames into GIF: %s\n", out_file);
        
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "ffmpeg -y -v warning -framerate 5 -i %s_%%03d.png -c:v libwebp -lossless 0 -q:v 85 -loop 0 -vf \"tpad=stop_mode=clone:stop_duration=2\" \"%s\"", prefix, out_file);        int ret = system(cmd);
        
        if (ret == 0) {
            printf("GIF generation finished successfully.\n");
        } else {
            fprintf(stderr, "Error: ffmpeg failed. Make sure ffmpeg is installed.\n");
        }
        
        for (int i = 0; i <= cb_data.last_iter; i++) {
            char filename[512];
            snprintf(filename, sizeof(filename), "%s_%03d.png", prefix, i);
            remove(filename);
        }
    }
    
    // =========================================================================
    // COMMAND: BENCH (RATE-DISTORTION ANALYZER)
    // =========================================================================
    else if (strcmp(command, "bench") == 0) {
        const char* in_file = NULL;
        int downsample_factor = 1;
        
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "-i") == 0 && i+1 < argc) in_file = argv[++i];
            else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) downsample_factor = atoi(argv[++i]);
        }
        if (!in_file) { print_help(argv[0]); return 1; }

        int w, h, channels;
        uint8_t* pixels = stbi_load(in_file, &w, &h, &channels, 3);
        if (!pixels) {
            fprintf(stderr, "Error: Failed to load image '%s'\n", in_file);
            return 1;
        }

        int new_w, new_h;
        uint8_t* ds_pixels = downsample_image(pixels, w, h, downsample_factor, &new_w, &new_h);
        const uint8_t* ref_pixels = ds_pixels;

        // Configuration Grid (Adjust arrays to add/remove tests)
        int min_sizes[] = {8};
        int max_sizes[] = {16, 32, 64, 128};
        float thresholds[] = {1.0f, 2.0f, 4.0f, 5.0f,10.0f, 20.0f, 40.0f};
        int windows[] = {32, 64, 128};

        int num_min = sizeof(min_sizes)/sizeof(min_sizes[0]);
        int num_max = sizeof(max_sizes)/sizeof(max_sizes[0]);
        int num_t = sizeof(thresholds)/sizeof(thresholds[0]);
        int num_w = sizeof(windows)/sizeof(windows[0]);

        // Print CSV Header
        printf("min_r,max_r,thresh,window,time_s,size_bytes,bpp,mse,psnr,score\n");
        fflush(stdout);
        for(int im=0; im<num_min; im++) {
            for(int iM=0; iM<num_max; iM++) {
                if (min_sizes[im] > max_sizes[iM]) continue; // Skip invalid
                
                for(int it=0; it<num_t; it++) {
                    for(int iw=0; iw<num_w; iw++) {
                        
                        CompressParams p = { min_sizes[im], max_sizes[iM], thresholds[it], windows[iw] };
                        
                        clock_t start = clock();
                        FractalImage* fi = compress_image(ds_pixels, new_w, new_h, p);
                        clock_t end = clock();
                        double time_s = ((double)(end - start)) / CLOCKS_PER_SEC;

                        const char* tmp_file = "bench_tmp.frc";
                        if (fi && save_fractal_image(fi, tmp_file)) {
                            long size_bytes = get_file_size(tmp_file);
                            
                            // Calculate Bits Per Pixel (bpp)
                            double bpp = (double)(size_bytes * 8) / (double)(new_w * new_h);

                            // Calculate PSNR
                            uint8_t* decomp = decompress_image(fi, 20);
                            double mse = 0;
                            for (int i = 0; i < new_w * new_h * 3; i++) {
                                double diff = ref_pixels[i] - decomp[i];
                                mse += diff * diff;
                            }
                            mse /= (new_w * new_h * 3);
                            double psnr = 10.0 * log10((255.0 * 255.0) / (mse + 1e-9));
                            
                            // THE COMBINED METRIC: Compression Efficiency Score (CES)
                            // Lower BPP gives a logarithmic boost to the score.
                            double score = psnr - (10.0 * log10(bpp + 1e-9));

                            // Output CSV row
                            printf("%d,%d,%.1f,%d,%.2f,%ld,%.4f,%.2f,%.2f,%.2f\n", 
                                   p.min_r_size, p.max_r_size, p.error_threshold, p.search_window, 
                                   time_s, size_bytes, bpp, mse, psnr, score);
                            fflush(stdout);
                            free(decomp);
                        }
                        if (fi) free_fractal_image(fi);
                        remove(tmp_file); // Clean up temp file
                    }
                }
            }
        }
        
        free(ds_pixels);
        stbi_image_free(pixels);
    }
    
    // =========================================================================
    // UNKNOWN COMMAND ERROR HANDLING
    // =========================================================================
    else {
        fprintf(stderr, "Error: Unknown command '%s'\n", command);
        print_help(argv[0]);
        return 1;
    }

    return 0;
}