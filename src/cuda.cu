#include "cuda.cuh"
#include "helper.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>

#include <cstring>
#include <cmath>
#include <stdio.h>

#define CUDA_MAX_RECURSIVE_DEPTH 16

///
/// Algorithm storage
///
// Number of particles in d_particles
unsigned int cuda_particles_count;
__constant__ unsigned int D_CONST_PARTICLES_COUNT;
// Host pointer to a list of particles
const Particle* h_particles;
// Device pointer to a list of particles
Particle* d_particles;
// Host pointer to a histogram of the number of particles contributing to each pixel
unsigned int* h_pixel_contribs;
// Device pointer to a histogram of the number of particles contributing to each pixel
unsigned int* d_pixel_contribs;
// Host pointer to an index of unique offsets for each pixels contributing colours
unsigned int* h_pixel_index;
// Device pointer to an index of unique offsets for each pixels contributing colours
unsigned int* d_pixel_index;

// Device pointer to storage for each pixels contributing colours
unsigned char* d_pixel_contrib_colours;

// Device pointer to storage for each pixels contributing colours' depth
float* d_pixel_contrib_depth;
// The number of contributors d_pixel_contrib_colours and d_pixel_contrib_depth have been allocated for
unsigned int cuda_pixel_contrib_count;
// Host storage of the output image dimensions
int cuda_output_image_width;
int cuda_output_image_height;
// Device storage of the output image dimensions
__constant__ int D_OUTPUT_IMAGE_WIDTH;
__constant__ int D_OUTPUT_IMAGE_HEIGHT;
unsigned char* h_output_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
cudaDeviceProp prop;
unsigned int maxThreads, maxThreadsPerBlock;

__device__ void selection_sort(float *data, unsigned char *d_pixel_contrib_colours, int left, int right) {
    for (int i = left; i <= right; ++i) {
        float min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j) {
            float val_j = data[j];
            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }
        // Swap the values.
        if (i != min_idx) {
            unsigned char colour_t[4];
            data[min_idx] = data[i];
            data[i] = min_val;
            memcpy(colour_t, d_pixel_contrib_colours + (4 * i), 4 * sizeof(unsigned char));
            memcpy(d_pixel_contrib_colours + (4 * i), d_pixel_contrib_colours + (4 * min_idx),
                   4 * sizeof(unsigned char));
            memcpy(d_pixel_contrib_colours + (4 * min_idx), colour_t, 4 * sizeof(unsigned char));
        }
    }
}

__device__ void
cuda_quick_sort(float *d_pixel_contrib_depth, unsigned char *d_pixel_contrib_colours, int first, int last, int depth) {
    if (depth > CUDA_MAX_RECURSIVE_DEPTH) {
        selection_sort(d_pixel_contrib_depth, d_pixel_contrib_colours, first, last);
    }
    int i, j, pivot;
    float depth_t;
    unsigned char color_t[4];
    if (first < last) {
        pivot = first;
        i = first;
        j = last;
        while (i < j) {
            while (d_pixel_contrib_depth[i] <= d_pixel_contrib_depth[pivot] && i < last)
                i++;
            while (d_pixel_contrib_depth[j] > d_pixel_contrib_depth[pivot])
                j--;
            if (i < j) {
                // Swap key
                depth_t = d_pixel_contrib_depth[i];
                d_pixel_contrib_depth[i] = d_pixel_contrib_depth[j];
                d_pixel_contrib_depth[j] = depth_t;
                // Swap color
                memcpy(color_t, d_pixel_contrib_colours + (4 * i), 4 * sizeof(unsigned char));
                memcpy(d_pixel_contrib_colours + (4 * i), d_pixel_contrib_colours + (4 * j),
                       4 * sizeof(unsigned char));
                memcpy(d_pixel_contrib_colours + (4 * j), color_t, 4 * sizeof(unsigned char));
            }
        }
        // Swap key
        depth_t = d_pixel_contrib_depth[pivot];
        d_pixel_contrib_depth[pivot] = d_pixel_contrib_depth[j];
        d_pixel_contrib_depth[j] = depth_t;
        // Swap color
        memcpy(color_t, d_pixel_contrib_colours + (4 * pivot), 4 * sizeof(unsigned char));
        memcpy(d_pixel_contrib_colours + (4 * pivot), d_pixel_contrib_colours + (4 * j),
               4 * sizeof(unsigned char));
        memcpy(d_pixel_contrib_colours + (4 * j), color_t, 4 * sizeof(unsigned char));
        // Recurse
        cuda_quick_sort(d_pixel_contrib_depth, d_pixel_contrib_colours, first, j - 1, depth + 1);
        cuda_quick_sort(d_pixel_contrib_depth, d_pixel_contrib_colours, j + 1, last, depth + 1);
    }
}

void cuda_begin(const Particle* init_particles, const unsigned int init_particles_count,
                const unsigned int out_image_width, const unsigned int out_image_height) {
    // These are basic CUDA memory allocations that match the CPU implementation
    // Depending on your optimisation, you may wish to rewrite these (and update cuda_end())
    h_particles = init_particles;

    // Allocate a copy of the initial particles, to be used during computation
    cuda_particles_count = init_particles_count;
    CUDA_CALL(cudaMemcpyToSymbol(D_CONST_PARTICLES_COUNT, &init_particles_count, sizeof(unsigned int)))
    CUDA_CALL(cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles, init_particles_count * sizeof(Particle), cudaMemcpyHostToDevice))

    // Allocate a histogram to track how many particles contribute to each pixel
    CUDA_CALL(cudaMalloc(&d_pixel_contribs, out_image_width * out_image_height * sizeof(unsigned int)));
    h_pixel_contribs = (unsigned int *) malloc(out_image_width * out_image_height * sizeof(unsigned int));
    // Allocate an index to track where data for each pixel's contributing colour starts/ends
    CUDA_CALL(cudaMalloc(&d_pixel_index, (out_image_width * out_image_height + 1) * sizeof(unsigned int)));
    h_pixel_index = (unsigned int *) malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));
    // Init a buffer to store colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_colours = 0;
    // Init a buffer to store depth of colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_depth = 0;
    // This tracks the number of contributes the two above buffers are allocated for, init 0
    cuda_pixel_contrib_count = 0;

    // Allocate output image
    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)))
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)))

    const int CHANNELS = 3;  // RGB
    CUDA_CALL(cudaMalloc(&d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char)));

    cudaSetDevice(0);
    cudaGetDeviceProperties(&prop, 0);

    maxThreads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    maxThreadsPerBlock = prop.maxThreadsPerBlock;
}

__global__ void stage1_calc_pixel_contribs(const Particle* __restrict__ particles, unsigned int* d_pixel_contribs){
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= D_CONST_PARTICLES_COUNT) {
        return;
    }
    float location_0 = __ldg(&particles[id].location[0]);
    float location_1 = __ldg(&particles[id].location[1]);
    float radius = __ldg(&particles[id].radius);

    int x_min = (int) roundf(location_0 - radius);
    int y_min = (int) roundf(location_1 - radius);
    int x_max = (int) roundf(location_0 + radius);
    int y_max = (int) roundf(location_1 + radius);

    x_min = x_min < 0 ? 0 : x_min;
    y_min = y_min < 0 ? 0 : y_min;
    x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_WIDTH - 1 : x_max;
    y_max = y_max >= D_OUTPUT_IMAGE_HEIGHT ? D_OUTPUT_IMAGE_HEIGHT - 1 : y_max;

    for (int x = x_min; x <= x_max; ++x) {
        for (int y = y_min; y <= y_max; ++y) {
            const float x_ab = (float) x + 0.5f - location_0;
            const float y_ab = (float) y + 0.5f - location_1;
            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
            if (pixel_distance <= radius) {
                const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;
                atomicAdd(&d_pixel_contribs[pixel_offset], 1);
            }
        }
    }
}


void cuda_stage1() {
    cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));
    stage1_calc_pixel_contribs<<<cuda_particles_count / maxThreadsPerBlock + 1, maxThreadsPerBlock>>>(d_particles, d_pixel_contribs);
#ifdef VALIDATION
    CUDA_CALL(cudaMemcpy(h_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost))
    validate_pixel_contribs(h_particles, cuda_particles_count, h_pixel_contribs, cuda_output_image_width,
                            cuda_output_image_height);
#endif
}

//__global__ void test(int x_min, int y_min, float location_0, float location_1, float location_2, unsigned char color_0, unsigned char color_1,
//                     unsigned char color_2, unsigned char color_3, float radius,
//                unsigned int *d_pixel_index, unsigned int *d_pixel_contribs,
//                float *d_pixel_contrib_depth, unsigned char *d_pixel_contrib_colours){
//    int x = threadIdx.x + x_min;
//    int y = threadIdx.y + y_min;
//    unsigned char color[] = {color_0, color_1, color_2, color_3};
//
//    const float x_ab = (float) x + 0.5f - location_0;
//    const float y_ab = (float) y + 0.5f - location_1;
//    const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
//    if (pixel_distance <= radius) {
//        const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;
//
//        unsigned int storage_offset = d_pixel_index[pixel_offset];
//        storage_offset += atomicAdd(&d_pixel_contribs[pixel_offset], 1);
//
//        memcpy(d_pixel_contrib_colours + (4 * storage_offset), color, 4 * sizeof(unsigned char));
//        *(d_pixel_contrib_depth + storage_offset) = location_2;
//    }
//}


__global__ void
stage2_calc_pixel_colour_depth(const Particle* __restrict__ particles, unsigned int *d_pixel_contribs, unsigned int *d_pixel_index,
                               unsigned char *d_pixel_contrib_colours, float *d_pixel_contrib_depth) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= D_CONST_PARTICLES_COUNT) {
        return;
    }
    float radius = __ldg(&particles[id].radius);
    float location_0 = __ldg(&particles[id].location[0]);
    float location_1 = __ldg(&particles[id].location[1]);
    float location_2 = __ldg(&particles[id].location[2]);
    unsigned char color[4];
    color[0] = __ldg(&particles[id].color[0]);
    color[1] = __ldg(&particles[id].color[1]);
    color[2] = __ldg(&particles[id].color[2]);
    color[3] = __ldg(&particles[id].color[3]);

    int x_min = (int) roundf(location_0 - radius);
    int y_min = (int) roundf(location_1 - radius);
    int x_max = (int) roundf(location_0 + radius);
    int y_max = (int) roundf(location_1 + radius);

    x_min = x_min < 0 ? 0 : x_min;
    y_min = y_min < 0 ? 0 : y_min;
    x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_WIDTH - 1 : x_max;
    y_max = y_max >= D_OUTPUT_IMAGE_HEIGHT ? D_OUTPUT_IMAGE_HEIGHT - 1 : y_max;

    for (int x = x_min; x <= x_max; ++x) {
        for (int y = y_min; y <= y_max; ++y) {
            const float x_ab = (float) x + 0.5f - location_0;
            const float y_ab = (float) y + 0.5f - location_1;
            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
            if (pixel_distance <= radius) {
                const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;

                unsigned int storage_offset = d_pixel_index[pixel_offset];
                storage_offset += atomicAdd(&d_pixel_contribs[pixel_offset], 1);

                memcpy(d_pixel_contrib_colours + (4 * storage_offset), color, 4 * sizeof(unsigned char));
                *(d_pixel_contrib_depth + storage_offset) = location_2;
            }
        }
    }
//    dim3 blockSize(x_max - x_min + 1, y_max - y_min + 1);
//    test<<<1, blockSize>>>(x_min, y_min, location_0, location_1, location_2, color[0], color[1], color[2], color[3], radius, d_pixel_index, d_pixel_contribs, d_pixel_contrib_depth, d_pixel_contrib_colours);
}


__global__ void stage2_sort_pairs(float *d_pixel_contrib_depth, unsigned char *d_pixel_contrib_colours,
                                  const unsigned int *__restrict__ d_pixel_index) {
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT) {
        return;
    }
    int first = __ldg(&d_pixel_index[id]);
    int last = __ldg(&d_pixel_index[id + 1]) - 1;
    if (first < last) {
        cuda_quick_sort(d_pixel_contrib_depth, d_pixel_contrib_colours, first, last, 0);
    }
}

void cuda_stage2() {
    thrust::device_ptr<unsigned int> d_index_vec = thrust::device_pointer_cast(d_pixel_index);
    thrust::device_ptr<unsigned int> d_contribs_vec = thrust::device_pointer_cast(d_pixel_contribs);
    thrust::exclusive_scan(d_contribs_vec, d_contribs_vec + cuda_output_image_width * cuda_output_image_height + 1,
                           d_index_vec);
    unsigned int TOTAL_CONTRIBS;
    CUDA_CALL(cudaMemcpy(&TOTAL_CONTRIBS, d_pixel_index + cuda_output_image_width * cuda_output_image_height, sizeof(unsigned int), cudaMemcpyDeviceToHost))
    if (TOTAL_CONTRIBS > cuda_particles_count) {
        if (d_pixel_contrib_colours) CUDA_CALL(cudaFree(d_pixel_contrib_colours))
        if (d_pixel_contrib_depth) CUDA_CALL(cudaFree(d_pixel_contrib_depth))

        CUDA_CALL(cudaMalloc(&d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char)))
        CUDA_CALL(cudaMalloc(&d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float)))
        cuda_pixel_contrib_count = TOTAL_CONTRIBS;
    }

    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int)))
    stage2_calc_pixel_colour_depth<<<cuda_particles_count / maxThreadsPerBlock + 1, maxThreadsPerBlock>>>(d_particles,
                                                                                                          d_pixel_contribs, d_pixel_index, d_pixel_contrib_colours, d_pixel_contrib_depth);
    stage2_sort_pairs<<<cuda_output_image_width * cuda_output_image_height / 896 + 1, 896>>>(d_pixel_contrib_depth,
                                                                                             d_pixel_contrib_colours, d_pixel_index);

#ifdef VALIDATION
    // Host pointer to storage for each pixels contributing colours
    unsigned char* h_pixel_contrib_colours;
    h_pixel_contrib_colours = (unsigned char *) malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
    // Host pointer to storage for each pixels contributing colours' depth
    float* h_pixel_contrib_depth;
    h_pixel_contrib_depth = (float *) malloc(TOTAL_CONTRIBS * sizeof(float));

    CUDA_CALL(cudaMemcpy(h_pixel_contribs, d_pixel_contribs, (cuda_output_image_width * cuda_output_image_height) * sizeof(unsigned int), cudaMemcpyDeviceToHost))
    CUDA_CALL(cudaMemcpy(h_pixel_index, d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost))
    CUDA_CALL(cudaMemcpy(h_pixel_contrib_colours, d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost))
    CUDA_CALL(cudaMemcpy(h_pixel_contrib_depth, d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float), cudaMemcpyDeviceToHost))
    validate_pixel_index(h_pixel_contribs, h_pixel_index, cuda_output_image_width, cuda_output_image_height);
    validate_sorted_pairs(h_particles, cuda_particles_count, h_pixel_index, cuda_output_image_width,
                          cuda_output_image_height, h_pixel_contrib_colours, h_pixel_contrib_depth);
    free(h_pixel_contrib_depth);
    free(h_pixel_contrib_colours);
#endif
}

__global__ void stage3_blend(unsigned char* d_output_image_data, unsigned char *d_pixel_contrib_colours, const unsigned int* __restrict__ d_pixel_index){
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT) {
        return;
    }
    unsigned int first = __ldg(&d_pixel_index[id]);
    unsigned int last = __ldg(&d_pixel_index[id + 1]);
    for (unsigned int i = first; i < last; i++) {
        const float opacity = (float) d_pixel_contrib_colours[i * 4 + 3] / (float) 255;
        d_output_image_data[id * 3 + 0] = (unsigned char) ((float) d_pixel_contrib_colours[i * 4 + 0] * opacity + (float) d_output_image_data[id * 3 + 0] * (1 - opacity));
        d_output_image_data[id * 3 + 1] = (unsigned char) ((float) d_pixel_contrib_colours[i * 4 + 1] * opacity + (float) d_output_image_data[id * 3 + 1] * (1 - opacity));
        d_output_image_data[id * 3 + 2] = (unsigned char) ((float) d_pixel_contrib_colours[i * 4 + 2] * opacity + (float) d_output_image_data[id * 3 + 2] * (1 - opacity));
    }
}

void cuda_stage3() {
    CUDA_CALL(cudaMemset(d_output_image_data, 255, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char)))
    stage3_blend<<<cuda_output_image_width * cuda_output_image_height / 1024 + 1, 1024>>>(d_output_image_data, d_pixel_contrib_colours, d_pixel_index);

#ifdef VALIDATION
    unsigned char* h_pixel_contrib_colours;
    h_pixel_contrib_colours = (unsigned char *) malloc(cuda_pixel_contrib_count * 4 * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(h_pixel_contrib_colours, d_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost))
    CImage image;
    image.height = cuda_output_image_height;
    image.width = cuda_output_image_width;
    image.channels = 3;
    image.data = (unsigned char *) malloc(
            cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(image.data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost))
    validate_blend(h_pixel_index, h_pixel_contrib_colours, &image);
    free(h_pixel_contrib_colours);
#endif
}
void cuda_end(CImage *output_image) {
    const int CHANNELS = 3;
    output_image->width = cuda_output_image_width;
    output_image->height = cuda_output_image_height;
    output_image->channels = CHANNELS;
    CUDA_CALL(cudaMemcpyAsync(output_image->data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    CUDA_CALL(cudaFree(d_pixel_contrib_depth));
    CUDA_CALL(cudaFree(d_pixel_contrib_colours));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_pixel_index));
    CUDA_CALL(cudaFree(d_pixel_contribs));
    CUDA_CALL(cudaFree(d_particles));
    free(h_pixel_index);
    free(h_pixel_contribs);
    // Return ptrs to nullptr
    d_pixel_contrib_depth = 0;
    d_pixel_contrib_colours = 0;
    d_output_image_data = 0;
    d_pixel_index = 0;
    d_pixel_contribs = 0;
    d_particles = 0;
}