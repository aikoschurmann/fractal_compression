/**
 * @file fractal.c
 * @brief Core PIFS implementation including Quadtree search, Fisher classification,
 * POSIX multithreading, and a custom Huffman Entropy FRC4 structural bit-packer.
 */
#include "fractal.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>

/** ============================================================================
 * MATH & GEOMETRY
 * ============================================================================ */

/**
 * @brief Mathematically applies 1 of 8 geometric isometries (rotations/reflections) to a 2D coordinate.
 */
void apply_isometry(int iso, int x, int y, int S, int* out_x, int* out_y) {
    int rx = x;
    int ry = y;
    
    // Apply reflection if isometry index is 4 or greater
    if (iso >= 4) { 
        rx = S - 1 - x; 
        iso -= 4; 
    }

    // Apply 90-degree rotations based on the remaining index
    if (iso == 1) { 
        *out_x = S - 1 - ry; 
        *out_y = rx; 
    } else if (iso == 2) { 
        *out_x = S - 1 - rx; 
        *out_y = S - 1 - ry; 
    } else if (iso == 3) { 
        *out_x = ry; 
        *out_y = S - 1 - rx; 
    } else { 
        *out_x = rx; 
        *out_y = ry; 
    }
}

/**
 * @brief Extracts a domain block from the pre-downsampled domain pool, automatically applying the desired isometry.
 */
void get_domain_block(const float* downsampled, int orig_w, int orig_h, int dx, int dy, int iso, int r_size, float* out_block) {
    int hw = orig_w / 2;
    int hh = orig_h / 2;
    int hdx = dx / 2;
    int hdy = dy / 2;

    for (int y = 0; y < r_size; y++) {
        for (int x = 0; x < r_size; x++) {
            int ix, iy;
            apply_isometry(iso, x, y, r_size, &ix, &iy);
            
            int d_img_x = hdx + ix;
            int d_img_y = hdy + iy;
            
            // Clamp to image boundaries
            if (d_img_x >= hw) {
                d_img_x = hw - 1;
            }
            if (d_img_y >= hh) {
                d_img_y = hh - 1;
            }
            
            out_block[y * r_size + x] = downsampled[d_img_y * hw + d_img_x];
        }
    }
}

/**
 * @brief Analytically solves for the optimal contrast (s) and brightness (o) using Least Squares.
 */
void compute_s_o(const float* R, const float* D, int n, float* s, float* o) {
    float sum_R = 0, sum_D = 0, sum_R_D = 0, sum_D_D = 0;
    
    for (int i = 0; i < n; i++) {
        sum_R += R[i];
        sum_D += D[i];
        sum_R_D += R[i] * D[i];
        sum_D_D += D[i] * D[i];
    }
    
    float denom = n * sum_D_D - sum_D * sum_D;
    
    // Check for flat domain blocks to avoid division by zero
    if (denom == 0.0f) { 
        *s = 0; 
        *o = sum_R / n; 
    } else {
        *s = (n * sum_R_D - sum_R * sum_D) / denom;
        *o = (sum_R - (*s) * sum_D) / n;
    }
    
    // Strict bounding for convergence (Contractive Mapping Theorem)
    if (*s > 1.0f) {
        *s = 1.0f;
    }
    if (*s < -1.0f) {
        *s = -1.0f;
    }
    
    // Recalculate brightness with the bounded contrast
    *o = (sum_R - (*s) * sum_D) / n;
}

/**
 * @brief Computes the absolute sum of squared error (SSE) between the Target block and the Affine Domain block.
 */
float compute_error(const float* R, const float* D, int n, float s, float o) {
    float err = 0;
    for (int i = 0; i < n; i++) {
        float diff = R[i] - (s * D[i] + o);
        err += diff * diff;
    }
    return err;
}

/**
 * @brief Fisher's Domain Classification. Splits a block into 4 quadrants and creates a 6-bit hash
 * based on the relative brightness ordering of the quadrants. Used to fast-reject non-matching blocks.
 */
int get_class_hash(const float* blk, int r_size, int invert) {
    int half = r_size / 2;
    float q[4] = {0};
    
    // Sum pixel intensities for each of the 4 quadrants
    for (int y = 0; y < half; y++) {
        for (int x = 0; x < half; x++) {
            q[0] += blk[y * r_size + x];
        }
        for (int x = half; x < r_size; x++) {
            q[1] += blk[y * r_size + x];
        }
    }
    for (int y = half; y < r_size; y++) {
        for (int x = 0; x < half; x++) {
            q[2] += blk[y * r_size + x];
        }
        for (int x = half; x < r_size; x++) {
            q[3] += blk[y * r_size + x];
        }
    }
    
    int hash = 0;
    
    // Bitmask based on the boolean > or <= results of the 6 quadrant pairs
    if (!invert) {
        if (q[0] > q[1]) hash |= 1;
        if (q[0] > q[2]) hash |= 2;
        if (q[0] > q[3]) hash |= 4;
        if (q[1] > q[2]) hash |= 8;
        if (q[1] > q[3]) hash |= 16;
        if (q[2] > q[3]) hash |= 32;
    } else {
        if (q[0] <= q[1]) hash |= 1;
        if (q[0] <= q[2]) hash |= 2;
        if (q[0] <= q[3]) hash |= 4;
        if (q[1] <= q[2]) hash |= 8;
        if (q[1] <= q[3]) hash |= 16;
        if (q[2] <= q[3]) hash |= 32;
    }
    
    return hash;
}

/**
 * @brief Container passing pointers to the full-size and pre-downsampled YCbCr planes.
 */
typedef struct {
    float* Y; 
    float* Cb; 
    float* Cr;
    float* dY; 
    float* dCb; 
    float* dCr;
} ImagePlanes;

/**
 * @brief The recursive heart of the compressor. Explores the local domain window on the Luma channel,
 * and recursively splits into 4 sub-quadrants if the error threshold cannot be met.
 */
void compress_quadtree(ImagePlanes planes, int width, int height, 
                       int rx, int ry, int r_size, CompressParams params,
                       FractalTransform** transforms, int* num_t, int* cap_t) {
    if (rx >= width || ry >= height) {
        return;
    }
    
    int d_size = 2 * r_size;
    int d_step = (r_size > 8) ? 8 : r_size; // Step size optimization for large blocks

    int max_alloc = params.max_r_size;
    if (max_alloc < 32) {
        max_alloc = 32;
    }
    int n_pixels = r_size * r_size;
    
    // Extract the target Range block for all 3 color planes
    float R_Y[max_alloc * max_alloc], R_Cb[max_alloc * max_alloc], R_Cr[max_alloc * max_alloc];
    for (int j = 0; j < r_size; j++) {
        for (int i = 0; i < r_size; i++) {
            int px = rx + i;
            int py = ry + j;
            
            if (px < width && py < height) {
                R_Y[j * r_size + i] = planes.Y[py * width + px];
                R_Cb[j * r_size + i] = planes.Cb[py * width + px];
                R_Cr[j * r_size + i] = planes.Cr[py * width + px];
            } else {
                R_Y[j * r_size + i] = 128.0f; // Padding
                R_Cb[j * r_size + i] = 128.0f;
                R_Cr[j * r_size + i] = 128.0f;
            }
        }
    }

    double sum_R = 0.0, sum_R_R = 0.0;
    for (int i = 0; i < n_pixels; i++) {
        sum_R += R_Y[i];
        sum_R_R += (double)R_Y[i] * R_Y[i];
    }
    
    int hash_R = get_class_hash(R_Y, r_size, 0);
    int hash_R_inv = get_class_hash(R_Y, r_size, 1);
    
    float best_err = 1e9f;
    FractalTransform best_t = {(uint16_t)rx, (uint16_t)ry, (uint16_t)r_size, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    // Define the bounding box for the local search window, constrained to [-127, 127] for 8-bit packing
    int min_dx = rx - params.search_window; 
    if (min_dx < rx - 127) {
        min_dx = rx - 127;
    }
    if (min_dx < 0) {
        min_dx = 0;
    }

    int max_dx = rx + params.search_window; 
    if (max_dx > rx + 127) {
        max_dx = rx + 127;
    }
    if (max_dx > width - d_size) {
        max_dx = width - d_size;
    }

    int min_dy = ry - params.search_window; 
    if (min_dy < ry - 127) {
        min_dy = ry - 127;
    }
    if (min_dy < 0) {
        min_dy = 0;
    }

    int max_dy = ry + params.search_window; 
    if (max_dy > ry + 127) {
        max_dy = ry + 127;
    }
    if (max_dy > height - d_size) {
        max_dy = height - d_size;
    }
    
    // Search the window exclusively using the Y (Luma) channel
    for (int dy = min_dy; dy <= max_dy; dy += d_step) {
        for (int dx = min_dx; dx <= max_dx; dx += d_step) {
            float base_D[max_alloc * max_alloc];
            get_domain_block(planes.dY, width, height, dx, dy, 0, r_size, base_D);
            
            double sum_D = 0.0, sum_D_D = 0.0;
            for (int i = 0; i < n_pixels; i++) {
                sum_D += base_D[i];
                sum_D_D += (double)base_D[i] * base_D[i];
            }
            
            double denom = (double)n_pixels * sum_D_D - sum_D * sum_D;
            int is_flat = (denom <= 1e-6); 

            float q[4] = {0};
            if (r_size >= 4) {
                int half = r_size / 2;
                for (int y = 0; y < half; y++) {
                    for (int x = 0; x < half; x++) {
                        q[0] += base_D[y * r_size + x];
                    }
                    for (int x = half; x < r_size; x++) {
                        q[1] += base_D[y * r_size + x];
                    }
                }
                for (int y = half; y < r_size; y++) {
                    for (int x = 0; x < half; x++) {
                        q[2] += base_D[y * r_size + x];
                    }
                    for (int x = half; x < r_size; x++) {
                        q[3] += base_D[y * r_size + x];
                    }
                }
            }

            for (int iso = 0; iso < 8; iso++) {
                if (r_size >= 4) {
                    float pq[4];
                    for (int i = 0; i < 4; i++) {
                        int qx = i % 2; 
                        int qy = i / 2;
                        int out_x, out_y;
                        apply_isometry(iso, qx, qy, 2, &out_x, &out_y);
                        pq[i] = q[out_y * 2 + out_x];
                    }
                    
                    int hash_D = 0;
                    if (pq[0] > pq[1]) hash_D |= 1; 
                    if (pq[0] > pq[2]) hash_D |= 2;
                    if (pq[0] > pq[3]) hash_D |= 4; 
                    if (pq[1] > pq[2]) hash_D |= 8;
                    if (pq[1] > pq[3]) hash_D |= 16; 
                    if (pq[2] > pq[3]) hash_D |= 32;
                    
                    // Reject early if class hash doesn't match
                    if (hash_D != hash_R && hash_D != hash_R_inv) {
                        continue;
                    }
                }
                
                float D_Y[max_alloc * max_alloc];
                if (iso == 0) {
                    memcpy(D_Y, base_D, r_size * r_size * sizeof(float));
                } else {
                    get_domain_block(planes.dY, width, height, dx, dy, iso, r_size, D_Y);
                }
                
                double sum_R_D = 0.0;
                for (int i = 0; i < n_pixels; i++) {
                    sum_R_D += (double)R_Y[i] * D_Y[i];
                }

                double s, o;
                if (is_flat) {
                    s = 0.0; 
                    o = sum_R / n_pixels;
                } else {
                    s = ((double)n_pixels * sum_R_D - sum_R * sum_D) / denom;
                    if (s > 1.0) {
                        s = 1.0; 
                    }
                    if (s < -1.0) {
                        s = -1.0;
                    }
                    o = (sum_R - s * sum_D) / n_pixels;
                }

                double err = sum_R_R + s * s * sum_D_D + (double)n_pixels * o * o 
                            - 2.0 * s * sum_R_D - 2.0 * o * sum_R 
                            + 2.0 * s * o * sum_D;
                
                if (err < 0.0) {
                    err = 0.0; 
                }
                
                // Track best match
                if ((float)err < best_err) {
                    best_err = (float)err;
                    best_t.x = dx; 
                    best_t.y = dy; 
                    best_t.isometry = iso;
                    best_t.contrast_y = (float)s; 
                    best_t.brightness_y = (float)o;
                }
                
                if (is_flat) {
                    break;
                }
            }
        }
    }
    
    // Decide whether to lock this block or split the Quadtree
    float mse_per_pixel = best_err / (r_size * r_size);
    if (mse_per_pixel > params.error_threshold && r_size > params.min_r_size) {
        // Error too high, split and recurse into 4 sub-blocks
        int half = r_size / 2;
        compress_quadtree(planes, width, height, rx, ry, half, params, transforms, num_t, cap_t);
        compress_quadtree(planes, width, height, rx + half, ry, half, params, transforms, num_t, cap_t);
        compress_quadtree(planes, width, height, rx, ry + half, half, params, transforms, num_t, cap_t);
        compress_quadtree(planes, width, height, rx + half, ry + half, half, params, transforms, num_t, cap_t);
    } else {
        // Accept block. Calculate Cb and Cr least squares using geometric coordinates found for Y.
        float D_Cb[max_alloc * max_alloc], D_Cr[max_alloc * max_alloc];
        get_domain_block(planes.dCb, width, height, best_t.x, best_t.y, best_t.isometry, r_size, D_Cb);
        get_domain_block(planes.dCr, width, height, best_t.x, best_t.y, best_t.isometry, r_size, D_Cr);
        compute_s_o(R_Cb, D_Cb, r_size * r_size, &best_t.contrast_cb, &best_t.brightness_cb);
        compute_s_o(R_Cr, D_Cr, r_size * r_size, &best_t.contrast_cr, &best_t.brightness_cr);
        
        // Push the finalized block to the dynamically sizing heap array
        if (*num_t >= *cap_t) {
            *cap_t *= 2;
            *transforms = realloc(*transforms, (*cap_t) * sizeof(FractalTransform));
        }
        (*transforms)[*num_t] = best_t;
        (*num_t)++;
    }
}

/**
 * @brief Defines a single Quadtree root processing job.
 */
typedef struct {
    ImagePlanes planes; 
    int width;
    int height; 
    int rx; 
    int ry;
    CompressParams params; 
    FractalTransform* transforms; 
    int num_transforms;
} ThreadTask;

/**
 * @brief Thread-safe queue containing all top-level Quadtree nodes.
 */
typedef struct {
    ThreadTask* tasks;
    int total_tasks;
    int current_task;
    pthread_mutex_t lock;
} ThreadPool;

/**
 * @brief Worker thread loop. Pulls tasks (Root Quadtree Nodes) off the mutex-protected queue.
 */
void* worker_thread(void* arg) {
    ThreadPool* pool = (ThreadPool*)arg;
    while (1) {
        pthread_mutex_lock(&pool->lock);
        int task_idx = pool->current_task++;
        printf(stderr, "Compressing... %d%%\r", (task_idx + 1) * 100 / pool->total_tasks); 
        fflush(stderr);
        pthread_mutex_unlock(&pool->lock);
        
        // Terminate thread if work is completely done
        if (task_idx >= pool->total_tasks) {
            break; 
        }
        
        ThreadTask* task = &pool->tasks[task_idx];
        int cap = 16;
        task->transforms = malloc(cap * sizeof(FractalTransform));
        task->num_transforms = 0;
        
        // Run the heavy recursive workload
        compress_quadtree(task->planes, task->width, task->height, 
                          task->rx, task->ry, task->params.max_r_size, task->params, 
                          &task->transforms, &task->num_transforms, &cap);
    }
    return NULL;
}

FractalImage* compress_image(const uint8_t* pixels, int width, int height, CompressParams params) {
    ImagePlanes planes;
    planes.Y = malloc(width * height * sizeof(float)); 
    planes.Cb = malloc(width * height * sizeof(float)); 
    planes.Cr = malloc(width * height * sizeof(float));
    
    // Convert RGB to YCbCr
    for (int i = 0; i < width * height; i++) {
        uint8_t r = pixels[i * 3 + 0]; 
        uint8_t g = pixels[i * 3 + 1]; 
        uint8_t b = pixels[i * 3 + 2];
        
        planes.Y[i]  = 0.299f * r + 0.587f * g + 0.114f * b;
        planes.Cb[i] = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b;
        planes.Cr[i] = 128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b;
    }
    
    // Pre-Downsample the planes by 50% for instant domain matching inside inner loops
    int hw = width / 2;
    int hh = height / 2;
    planes.dY = malloc(hw * hh * sizeof(float)); 
    planes.dCb = malloc(hw * hh * sizeof(float)); 
    planes.dCr = malloc(hw * hh * sizeof(float));
    
    for (int y = 0; y < hh; y++) {
        for (int x = 0; x < hw; x++) {
            int px = x * 2;
            int py = y * 2;
            int idx1 = py * width + px; 
            int idx2 = (px + 1 < width) ? py * width + (px + 1) : idx1;
            int idx3 = (py + 1 < height) ? (py + 1) * width + px : idx1; 
            int idx4 = (px + 1 < width && py + 1 < height) ? (py + 1) * width + (px + 1) : idx1;

            planes.dY[y * hw + x] = (planes.Y[idx1] + planes.Y[idx2] + planes.Y[idx3] + planes.Y[idx4]) / 4.0f;
            planes.dCb[y * hw + x] = (planes.Cb[idx1] + planes.Cb[idx2] + planes.Cb[idx3] + planes.Cb[idx4]) / 4.0f;
            planes.dCr[y * hw + x] = (planes.Cr[idx1] + planes.Cr[idx2] + planes.Cr[idx3] + planes.Cr[idx4]) / 4.0f;
        }
    }
    
    // Prepare the Thread Pool grid based on max block size
    int blocks_x = (width + params.max_r_size - 1) / params.max_r_size;
    int blocks_y = (height + params.max_r_size - 1) / params.max_r_size;
    int total_tasks = blocks_x * blocks_y;
    
    ThreadTask* tasks = malloc(total_tasks * sizeof(ThreadTask));
    for (int i = 0; i < total_tasks; i++) {
        tasks[i].planes = planes; 
        tasks[i].width = width; 
        tasks[i].height = height;
        tasks[i].rx = (i % blocks_x) * params.max_r_size; 
        tasks[i].ry = (i / blocks_x) * params.max_r_size;
        tasks[i].params = params; 
        tasks[i].num_transforms = 0; 
        tasks[i].transforms = NULL;
    }
    
    ThreadPool pool; 
    pool.tasks = tasks; 
    pool.total_tasks = total_tasks; 
    pool.current_task = 0;
    pthread_mutex_init(&pool.lock, NULL);
    
    // Launch threads equivalent to system logical cores
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN); 
    if (num_threads <= 0) {
        num_threads = 8;
    }
    
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));

    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, &pool);
    }

    // Await finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL); 
    }
    
    // Gather the parallelized disjoint lists of structs into one linear master array
    int total_transforms = 0;
    for (int i = 0; i < total_tasks; i++) {
        total_transforms += tasks[i].num_transforms;
    }

    FractalTransform* all_t = malloc(total_transforms * sizeof(FractalTransform));

    int offset = 0;
    for (int i = 0; i < total_tasks; i++) {
        if (tasks[i].num_transforms > 0) {
            memcpy(all_t + offset, tasks[i].transforms, tasks[i].num_transforms * sizeof(FractalTransform));
            offset += tasks[i].num_transforms;
            free(tasks[i].transforms);
        }
    }
    
    // Cleanup temporary processing arrays and threading structures
    free(tasks); 
    free(threads); 
    pthread_mutex_destroy(&pool.lock);
    free(planes.Y); 
    free(planes.Cb); 
    free(planes.Cr);
    free(planes.dY); 
    free(planes.dCb); 
    free(planes.dCr);
    
    // Return finalized Fractal payload
    FractalImage* fi = malloc(sizeof(FractalImage));
    fi->width = width; 
    fi->height = height;
    fi->min_r_size = params.min_r_size; 
    fi->max_r_size = params.max_r_size;
    fi->num_transforms = total_transforms; 
    fi->transforms = all_t;
    
    return fi;
}

/** ============================================================================
 * DECOMPRESSION
 * ============================================================================ */

typedef struct {
    const FractalImage* fi; 
    const float *dY, *dCb, *dCr; 
    float *nY, *nCb, *nCr;
    int w;
    int h;
    int start_idx;
    int end_idx;
} DecompTask;

void* decomp_worker(void* arg) {
    DecompTask* task = (DecompTask*)arg;
    int max_r_alloc = 32;

    for (int i = 0; i < task->fi->num_transforms; i++) {
        if (task->fi->transforms[i].r_size > max_r_alloc) {
            max_r_alloc = task->fi->transforms[i].r_size;
        }
    }
    
    for (int i = task->start_idx; i < task->end_idx; i++) {
        FractalTransform t = task->fi->transforms[i];
        float D_Y[max_r_alloc * max_r_alloc];
        float D_Cb[max_r_alloc * max_r_alloc];
        float D_Cr[max_r_alloc * max_r_alloc];
        
        get_domain_block(task->dY, task->w, task->h, t.x, t.y, t.isometry, t.r_size, D_Y);
        get_domain_block(task->dCb, task->w, task->h, t.x, t.y, t.isometry, t.r_size, D_Cb);
        get_domain_block(task->dCr, task->w, task->h, t.x, t.y, t.isometry, t.r_size, D_Cr);
        
        for (int r_j = 0; r_j < t.r_size; r_j++) {
            for (int r_i = 0; r_i < t.r_size; r_i++) {
                int px = t.rx + r_i;
                int py = t.ry + r_j;
                
                if (px < task->w && py < task->h) {
                    task->nY[py * task->w + px] = t.contrast_y * D_Y[r_j * t.r_size + r_i] + t.brightness_y;
                    task->nCb[py * task->w + px] = t.contrast_cb * D_Cb[r_j * t.r_size + r_i] + t.brightness_cb;
                    task->nCr[py * task->w + px] = t.contrast_cr * D_Cr[r_j * t.r_size + r_i] + t.brightness_cr;
                }
            }
        }
    }
    return NULL;
}

uint8_t* decompress_image_cb(const FractalImage* fi, int max_iterations, FrameCallback cb, void* user_data) {
    int w = fi->width;
    int h = fi->height;
    
    // Working memory arrays. The contraction mapping needs an old state (Y) and writes to a new state (nY).
    float* Y = malloc(w * h * sizeof(float)); 
    float* Cb = malloc(w * h * sizeof(float)); 
    float* Cr = malloc(w * h * sizeof(float));
    float* nY = malloc(w * h * sizeof(float)); 
    float* nCb = malloc(w * h * sizeof(float)); 
    float* nCr = malloc(w * h * sizeof(float));
    
    // Iteration 0: The Banach Fixed Point theorem works from ANY starting image. We use uniform gray.
    for (int i = 0; i < w * h; i++) { 
        Y[i] = 128.0f; 
        Cb[i] = 128.0f; 
        Cr[i] = 128.0f; 
    }
    
    uint8_t* img = malloc(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) {
        img[i] = 128;
    }
    
    if (cb) {
        cb(0, img, w, h, user_data); 
    }
    
    int hw = w / 2;
    int hh = h / 2;
    float* dY = malloc(hw * hh * sizeof(float)); 
    float* dCb = malloc(hw * hh * sizeof(float)); 
    float* dCr = malloc(hw * hh * sizeof(float));

    for (int iter = 0; iter < max_iterations; iter++) {
        for (int y = 0; y < hh; y++) {
            for (int x = 0; x < hw; x++) {
                int px = x * 2;
                int py = y * 2;
                int idx1 = py * w + px; 
                int idx2 = (px + 1 < w) ? py * w + (px + 1) : idx1;
                int idx3 = (py + 1 < h) ? (py + 1) * w + px : idx1; 
                int idx4 = (px + 1 < w && py + 1 < h) ? (py + 1) * w + (px + 1) : idx1;

                dY[y * hw + x] = (Y[idx1] + Y[idx2] + Y[idx3] + Y[idx4]) / 4.0f;
                dCb[y * hw + x] = (Cb[idx1] + Cb[idx2] + Cb[idx3] + Cb[idx4]) / 4.0f;
                dCr[y * hw + x] = (Cr[idx1] + Cr[idx2] + Cr[idx3] + Cr[idx4]) / 4.0f;
            }
        }
        
        int num_threads = sysconf(_SC_NPROCESSORS_ONLN); 
        if (num_threads <= 0) {
            num_threads = 8;
        }
        
        pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
        DecompTask* dtasks = malloc(num_threads * sizeof(DecompTask));
        
        int chunk = (fi->num_transforms + num_threads - 1) / num_threads;
        for (int i = 0; i < num_threads; i++) {
            dtasks[i].fi = fi; 
            dtasks[i].dY = dY; 
            dtasks[i].dCb = dCb; 
            dtasks[i].dCr = dCr;
            dtasks[i].nY = nY; 
            dtasks[i].nCb = nCb; 
            dtasks[i].nCr = nCr;
            dtasks[i].w = w; 
            dtasks[i].h = h;
            dtasks[i].start_idx = i * chunk;
            dtasks[i].end_idx = (i == num_threads - 1) ? fi->num_transforms : (i + 1) * chunk;

            if (dtasks[i].start_idx > fi->num_transforms) {
                dtasks[i].start_idx = fi->num_transforms;
            }
            if (dtasks[i].end_idx > fi->num_transforms) {
                dtasks[i].end_idx = fi->num_transforms;
            }
            pthread_create(&threads[i], NULL, decomp_worker, &dtasks[i]);
        }
        
        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
        
        free(threads); 
        free(dtasks);
        
        float diff_sum = 0;
        for (int i = 0; i < w * h; i++) {
            diff_sum += fabs(Y[i] - nY[i]);
            
            // Inverse map YCbCr -> RGB
            int rgb_r = (int)(nY[i] + 1.402f * (nCr[i] - 128.0f));
            int rgb_g = (int)(nY[i] - 0.344136f * (nCb[i] - 128.0f) - 0.714136f * (nCr[i] - 128.0f));
            int rgb_b = (int)(nY[i] + 1.772f * (nCb[i] - 128.0f));
            
            // Clamp RGB values
            if (rgb_r < 0) rgb_r = 0; 
            if (rgb_r > 255) rgb_r = 255;
            
            if (rgb_g < 0) rgb_g = 0; 
            if (rgb_g > 255) rgb_g = 255;
            
            if (rgb_b < 0) rgb_b = 0; 
            if (rgb_b > 255) rgb_b = 255;
            
            img[i * 3 + 0] = rgb_r; 
            img[i * 3 + 1] = rgb_g; 
            img[i * 3 + 2] = rgb_b;
        }
        
        // Swap pointers for next iteration
        float* tmp; 
        tmp = Y; Y = nY; nY = tmp; 
        tmp = Cb; Cb = nCb; nCb = tmp; 
        tmp = Cr; Cr = nCr; nCr = tmp;
        
        if (cb) {
            cb(iter + 1, img, w, h, user_data);
        }
        
        if (diff_sum / (w * h) <= 0.5f) {
            break; 
        }
    }
    
    // --- POST-PROCESSING: Deblocking Filter ---
    // Identifies quadtree block boundaries to apply a localized lightweight Gaussian blur.
    uint8_t* boundary = calloc(w * h, 1);
    for (int i = 0; i < fi->num_transforms; i++) {
        FractalTransform t = fi->transforms[i];
        
        // Mark top boundary
        if (t.ry > 0) {
            for (int dx = 0; dx < t.r_size; dx++) {
                int px = t.rx + dx;
                if (px < w) {
                    boundary[t.ry * w + px] = 1;
                    boundary[(t.ry - 1) * w + px] = 1;
                }
            }
        }
        
        // Mark left boundary
        if (t.rx > 0) {
            for (int dy = 0; dy < t.r_size; dy++) {
                int py = t.ry + dy;
                if (py < h) {
                    boundary[py * w + t.rx] = 1;
                    boundary[py * w + t.rx - 1] = 1;
                }
            }
        }
    }

    uint8_t* blurred_img = malloc(w * h * 3);
    memcpy(blurred_img, img, w * h * 3);

    // 3x3 Gaussian weights
    int g_kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (boundary[y * w + x]) {
                int r_sum = 0, g_sum = 0, b_sum = 0, weight_sum = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            int wgt = g_kernel[dy + 1][dx + 1];
                            int idx = (ny * w + nx) * 3;
                            r_sum += img[idx + 0] * wgt;
                            g_sum += img[idx + 1] * wgt;
                            b_sum += img[idx + 2] * wgt;
                            weight_sum += wgt;
                        }
                    }
                }
                int idx = (y * w + x) * 3;
                blurred_img[idx + 0] = r_sum / weight_sum;
                blurred_img[idx + 1] = g_sum / weight_sum;
                blurred_img[idx + 2] = b_sum / weight_sum;
            }
        }
    }
    
    memcpy(img, blurred_img, w * h * 3);
    
    // Cleanup
    free(blurred_img);
    free(boundary);
    free(Y); 
    free(Cb); 
    free(Cr); 
    free(nY); 
    free(nCb); 
    free(nCr); 
    free(dY); 
    free(dCb); 
    free(dCr);
    
    return img;
}

uint8_t* decompress_image(const FractalImage* fi, int max_iterations) { 
    return decompress_image_cb(fi, max_iterations, NULL, NULL); 
}

void free_fractal_image(FractalImage* fi) { 
    if (fi) { 
        free(fi->transforms); 
        free(fi); 
    } 
}

/** ============================================================================
 * MEMORY STREAM (PRE-ENTROPY BUFFER)
 * ============================================================================ */

typedef struct { 
    uint8_t* data; 
    size_t size;
    size_t cap;
    size_t pos; 
} MemStream;

void ms_init(MemStream* ms, size_t initial_cap) {
    ms->cap = initial_cap; 
    ms->size = 0; 
    ms->pos = 0; 
    ms->data = malloc(ms->cap);
}

void ms_write(MemStream* ms, const void* ptr, size_t len) {
    if (ms->size + len > ms->cap) { 
        ms->cap = (ms->size + len) * 2; 
        ms->data = realloc(ms->data, ms->cap); 
    }
    memcpy(ms->data + ms->size, ptr, len); 
    ms->size += len;
}

void ms_read(MemStream* ms, void* ptr, size_t len) {
    if (ms->pos + len <= ms->size) { 
        memcpy(ptr, ms->data + ms->pos, len); 
        ms->pos += len; 
    }
}

void ms_free(MemStream* ms) { 
    free(ms->data); 
}

/** ============================================================================
 * HUFFMAN ENTROPY ENCODER / DECODER
 * ============================================================================ */

typedef struct HuffNode { 
    int freq; 
    uint8_t byte; 
    struct HuffNode *left, *right; 
} HuffNode;

HuffNode* new_huff_node(int freq, uint8_t byte, HuffNode* l, HuffNode* r) {
    HuffNode* n = malloc(sizeof(HuffNode));
    n->freq = freq;
    n->byte = byte; 
    n->left = l; 
    n->right = r; 
    return n;
}

void free_huff_tree(HuffNode* root) {
    if (!root) {
        return; 
    }
    free_huff_tree(root->left); 
    free_huff_tree(root->right); 
    free(root);
}

// Build codebook recursively
void build_codes(HuffNode* root, uint32_t code, int depth, uint32_t* codes, int* lengths) {
    if (!root) {
        return;
    }
    if (!root->left && !root->right) { 
        codes[root->byte] = code; 
        lengths[root->byte] = depth; 
        return; 
    }

    build_codes(root->left, (code << 1), depth + 1, codes, lengths);
    build_codes(root->right, (code << 1) | 1, depth + 1, codes, lengths);
}

// Applies self-contained Huffman Entropy encoding to squash the Memory Stream to disk
void compress_huffman(MemStream* in, FILE* out) {
    int freqs[256] = {0};
    for (size_t i = 0; i < in->size; i++) {
        freqs[in->data[i]]++;
    }
    
    // Write the raw size and frequency table (header overhead)
    fwrite(&in->size, sizeof(size_t), 1, out);
    fwrite(freqs, sizeof(int), 256, out);
    
    // Min-Heap approach array for Tree Building
    HuffNode* nodes[512]; 
    int num_nodes = 0;
    
    for (int i = 0; i < 256; i++) {
        if (freqs[i] > 0) {
            nodes[num_nodes++] = new_huff_node(freqs[i], i, NULL, NULL);
        }
    }
    
    if (num_nodes == 0) {
        return;
    }
    
    while (num_nodes > 1) {
        // Find two minimum nodes
        int min1 = -1, min2 = -1;
        for (int i = 0; i < num_nodes; i++) {
            if (min1 == -1 || nodes[i]->freq < nodes[min1]->freq) { 
                min2 = min1; 
                min1 = i; 
            } else if (min2 == -1 || nodes[i]->freq < nodes[min2]->freq) { 
                min2 = i; 
            }
        }
        
        HuffNode* l = nodes[min1]; 
        HuffNode* r = nodes[min2];
        HuffNode* parent = new_huff_node(l->freq + r->freq, 0, l, r);
        
        // Swap and reduce array
        if (min1 > min2) { 
            int t = min1; 
            min1 = min2; 
            min2 = t; 
        }

        nodes[min1] = parent;
        nodes[min2] = nodes[num_nodes - 1];
        num_nodes--;
    }
    
    HuffNode* root = nodes[0];
    uint32_t codes[256] = {0}; 
    int lengths[256] = {0};
    build_codes(root, 0, 0, codes, lengths);
    
    // Bit-packing Engine
    uint8_t bit_buf = 0; 
    int bit_count = 0;
    
    for (size_t i = 0; i < in->size; i++) {
        uint8_t b = in->data[i];
        uint32_t code = codes[b]; 
        int len = lengths[b];
        
        for (int j = len - 1; j >= 0; j--) {
            uint8_t bit = (code >> j) & 1;
            bit_buf = (bit_buf << 1) | bit; 
            bit_count++;
            
            if (bit_count == 8) { 
                fwrite(&bit_buf, 1, 1, out); 
                bit_buf = 0; 
                bit_count = 0; 
            }
        }
    }

    if (bit_count > 0) { 
        bit_buf <<= (8 - bit_count); 
        fwrite(&bit_buf, 1, 1, out); 
    }

    free_huff_tree(root);
}

// Reads the Huffman Entropy dictionary and unpacks the bits into a flat Memory Stream
void decompress_huffman(FILE* in, MemStream* out) {
    size_t target_size;
    if (fread(&target_size, sizeof(size_t), 1, in) != 1) {
        return;
    }
    
    int freqs[256];
    if (fread(freqs, sizeof(int), 256, in) != 256) {
        return;
    }
    
    ms_init(out, target_size);
    HuffNode* nodes[512]; 
    int num_nodes = 0;
    
    for (int i = 0; i < 256; i++) {
        if (freqs[i] > 0) {
            nodes[num_nodes++] = new_huff_node(freqs[i], i, NULL, NULL);
        }
    }
    
    if (num_nodes == 0) {
        return;
    }
    
    while (num_nodes > 1) {
        int min1 = -1, min2 = -1;
        for (int i = 0; i < num_nodes; i++) {
            if (min1 == -1 || nodes[i]->freq < nodes[min1]->freq) { 
                min2 = min1; 
                min1 = i; 
            } else if (min2 == -1 || nodes[i]->freq < nodes[min2]->freq) { 
                min2 = i; 
            }
        }
        
        HuffNode* l = nodes[min1]; 
        HuffNode* r = nodes[min2];
        HuffNode* parent = new_huff_node(l->freq + r->freq, 0, l, r);
        
        if (min1 > min2) { 
            int t = min1; 
            min1 = min2; 
            min2 = t; 
        }
        nodes[min1] = parent; 
        nodes[min2] = nodes[num_nodes - 1]; 
        num_nodes--;
    }
    
    HuffNode* root = nodes[0]; 
    HuffNode* curr = root;
    uint8_t bit_buf = 0; 
    int bit_count = 0;
    
    while (out->size < target_size) {
        if (bit_count == 0) { 
            if (fread(&bit_buf, 1, 1, in) != 1) {
                break; 
            }
            bit_count = 8; 
        }
        
        uint8_t bit = (bit_buf >> 7) & 1;
        bit_buf <<= 1; 
        bit_count--;
        
        curr = bit ? curr->right : curr->left;
        if (!curr->left && !curr->right) {
            ms_write(out, &curr->byte, 1);
            curr = root;
        }
    }
    
    free_huff_tree(root);
    out->pos = 0; // Reset read cursor for the Quadtree parser
}

/** ============================================================================
 * FILE I/O (QUADTREE -> MEMSTREAM -> HUFFMAN)
 * ============================================================================ */

void write_quadtree(MemStream* ms, FractalTransform** t_ptr, int* t_count, int rx, int ry, int r_size, int w, int h) {
    if (rx >= w || ry >= h) {
        return;
    }
    
    if (*t_count > 0 && (*t_ptr)->rx == rx && (*t_ptr)->ry == ry && (*t_ptr)->r_size == r_size) {
        uint8_t header = ((*t_ptr)->isometry & 0x07); 
        ms_write(ms, &header, 1);
        
        int8_t dx = (int8_t)((*t_ptr)->x - rx);
        int8_t dy = (int8_t)((*t_ptr)->y - ry);
        ms_write(ms, &dx, 1);
        ms_write(ms, &dy, 1);
        
        // Convert strict float contrast [-1.0, 1.0] to single int8_t [-127, 127]
        int8_t sy = (int8_t)((*t_ptr)->contrast_y * 127.0f); 
        int16_t oy = (int16_t)((*t_ptr)->brightness_y);
        int8_t scb = (int8_t)((*t_ptr)->contrast_cb * 127.0f); 
        int16_t ocb = (int16_t)((*t_ptr)->brightness_cb);
        int8_t scr = (int8_t)((*t_ptr)->contrast_cr * 127.0f); 
        int16_t ocr = (int16_t)((*t_ptr)->brightness_cr);
        
        ms_write(ms, &sy, 1); 
        ms_write(ms, &oy, 2);
        ms_write(ms, &scb, 1); 
        ms_write(ms, &ocb, 2);
        ms_write(ms, &scr, 1); 
        ms_write(ms, &ocr, 2);
        
        (*t_ptr)++; 
        (*t_count)--;
    } else {
        uint8_t header = 0x80; 
        ms_write(ms, &header, 1);
        
        int half = r_size / 2;
        write_quadtree(ms, t_ptr, t_count, rx, ry, half, w, h);
        write_quadtree(ms, t_ptr, t_count, rx + half, ry, half, w, h);
        write_quadtree(ms, t_ptr, t_count, rx, ry + half, half, w, h);
        write_quadtree(ms, t_ptr, t_count, rx + half, ry + half, half, w, h);
    }
}

int save_fractal_image(const FractalImage* fi, const char* filename) {
    FILE* f = fopen(filename, "wb"); 
    if (!f) {
        return 0;
    }
    
    fwrite("FRC4", 1, 4, f); // Revision 4 (Entropy Enabled)
    fwrite(&fi->width, sizeof(int), 1, f); 
    fwrite(&fi->height, sizeof(int), 1, f);
    fwrite(&fi->min_r_size, sizeof(int), 1, f); 
    fwrite(&fi->max_r_size, sizeof(int), 1, f);
    
    // Dump raw quadtree payload to an intermediate RAM buffer
    MemStream ms; 
    ms_init(&ms, 1024 * 1024);
    
    FractalTransform* t_ptr = fi->transforms; 
    int t_count = fi->num_transforms;

    for (int ry = 0; ry < fi->height; ry += fi->max_r_size) {
        for (int rx = 0; rx < fi->width; rx += fi->max_r_size) {
            write_quadtree(&ms, &t_ptr, &t_count, rx, ry, fi->max_r_size, fi->width, fi->height);
        }
    }
    
    // Squash the RAM buffer to disk using Entropy Reduction
    compress_huffman(&ms, f);
    
    ms_free(&ms); 
    fclose(f); 
    return 1;
}

void read_quadtree(MemStream* ms, FractalTransform** arr, int* cap, int* count, int rx, int ry, int r_size, int w, int h) {
    if (rx >= w || ry >= h) {
        return;
    }
    
    uint8_t header; 
    ms_read(ms, &header, 1);
    
    if ((header & 0x80) == 0) {
        int8_t dx, dy, sy, scb, scr; 
        int16_t oy, ocb, ocr;
        
        ms_read(ms, &dx, 1); 
        ms_read(ms, &dy, 1);
        ms_read(ms, &sy, 1); 
        ms_read(ms, &oy, 2);
        ms_read(ms, &scb, 1); 
        ms_read(ms, &ocb, 2);
        ms_read(ms, &scr, 1); 
        ms_read(ms, &ocr, 2);
        
        if (*count >= *cap) { 
            *cap *= 2; 
            *arr = realloc(*arr, *cap * sizeof(FractalTransform)); 
        }
        
        (*arr)[*count].rx = rx; 
        (*arr)[*count].ry = ry; 
        (*arr)[*count].r_size = r_size;
        (*arr)[*count].x = rx + dx; 
        (*arr)[*count].y = ry + dy; 
        (*arr)[*count].isometry = header & 0x07;
        
        (*arr)[*count].contrast_y = (float)sy / 127.0f; 
        (*arr)[*count].brightness_y = (float)oy;
        (*arr)[*count].contrast_cb = (float)scb / 127.0f; 
        (*arr)[*count].brightness_cb = (float)ocb;
        (*arr)[*count].contrast_cr = (float)scr / 127.0f; 
        (*arr)[*count].brightness_cr = (float)ocr;
        
        (*count)++;
    } else {
        int half = r_size / 2;
        read_quadtree(ms, arr, cap, count, rx, ry, half, w, h);
        read_quadtree(ms, arr, cap, count, rx + half, ry, half, w, h);
        read_quadtree(ms, arr, cap, count, rx, ry + half, half, w, h);
        read_quadtree(ms, arr, cap, count, rx + half, ry + half, half, w, h);
    }
}

FractalImage* load_fractal_image(const char* filename) {
    FILE* f = fopen(filename, "rb"); 
    if (!f) {
        return NULL;
    }
    
    char magic[4]; 
    if (fread(magic, 1, 4, f) != 4 || strncmp(magic, "FRC4", 4) != 0) { 
        fclose(f); 
        return NULL; 
    }
    
    FractalImage* fi = malloc(sizeof(FractalImage));
    fread(&fi->width, sizeof(int), 1, f); 
    fread(&fi->height, sizeof(int), 1, f);
    fread(&fi->min_r_size, sizeof(int), 1, f); 
    fread(&fi->max_r_size, sizeof(int), 1, f);
    
    // Entropy Decode from disk into RAM buffer
    MemStream ms; 
    decompress_huffman(f, &ms);
    
    int cap = 1024; 
    int count = 0; 
    
    FractalTransform* arr = malloc(cap * sizeof(FractalTransform));
    
    for (int ry = 0; ry < fi->height; ry += fi->max_r_size) {
        for (int rx = 0; rx < fi->width; rx += fi->max_r_size) {
            read_quadtree(&ms, &arr, &cap, &count, rx, ry, fi->max_r_size, fi->width, fi->height);
        }
    }
    
    ms_free(&ms); 
    fi->num_transforms = count; 
    fi->transforms = arr; 
    fclose(f); 
    
    return fi;
}