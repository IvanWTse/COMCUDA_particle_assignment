#include "openmp.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

extern void cpu_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last);

unsigned int openmp_particles_count;
Particle *openmp_particles;
unsigned int *openmp_pixel_contribs;
unsigned int *openmp_pixel_index;
unsigned char *openmp_pixel_contrib_colours;
float *openmp_pixel_contrib_depth;
unsigned int openmp_pixel_contrib_count;
CImage openmp_output_image;

void openmp_begin(const Particle* init_particles, const unsigned int init_particles_count,
                  const unsigned int out_image_width, const unsigned int out_image_height) {

    openmp_particles_count = init_particles_count;
    openmp_particles = malloc(init_particles_count * sizeof(Particle));
    memcpy(openmp_particles, init_particles, init_particles_count * sizeof(Particle));

    openmp_pixel_contribs = (unsigned int *)malloc(out_image_width * out_image_height * sizeof(unsigned int));
    openmp_pixel_index = (unsigned int*)malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));
    openmp_pixel_contrib_colours = 0;
    openmp_pixel_contrib_depth = 0;
    openmp_pixel_contrib_count = 0;

    openmp_output_image.width = (int)out_image_width;
    openmp_output_image.height = (int)out_image_height;
    openmp_output_image.channels = 3;  // RGB
    openmp_output_image.data = (unsigned char *)malloc(openmp_output_image.width * openmp_output_image.height * openmp_output_image.channels * sizeof(unsigned char));
}
void openmp_stage1() {
    memset(openmp_pixel_contribs, 0, openmp_output_image.width * openmp_output_image.height * sizeof(unsigned int));
#pragma omp parallel for default(none) shared(openmp_particles, openmp_particles_count, openmp_output_image, openmp_pixel_contribs)
    for (unsigned int i = 0; i < openmp_particles_count; ++i) {
        // Local cache of the particle
        Particle openmp_particle_i = openmp_particles[i];
        int x_min = (int)roundf(openmp_particle_i.location[0] - openmp_particle_i.radius);
        int y_min = (int)roundf(openmp_particle_i.location[1] - openmp_particle_i.radius);
        int x_max = (int)roundf(openmp_particle_i.location[0] + openmp_particle_i.radius);
        int y_max = (int)roundf(openmp_particle_i.location[1] + openmp_particle_i.radius);

        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= openmp_output_image.width ? openmp_output_image.width - 1 : x_max;
        y_max = y_max >= openmp_output_image.height ? openmp_output_image.height - 1 : y_max;

        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float) x + 0.5f - openmp_particle_i.location[0];
                const float y_ab = (float) y + 0.5f - openmp_particle_i.location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= openmp_particle_i.radius) {
                    const unsigned int pixel_offset = y * openmp_output_image.width + x;
#pragma omp atomic
                    ++openmp_pixel_contribs[pixel_offset];
                }
            }
        }
    }

#ifdef VALIDATION
    validate_pixel_contribs(openmp_particles, openmp_particles_count, openmp_pixel_contribs, openmp_output_image.width, openmp_output_image.height);
#endif
}
void openmp_stage2() {
    openmp_pixel_index[0] = 0;
    for (int i = 0; i < openmp_output_image.width * openmp_output_image.height; ++i) {
        openmp_pixel_index[i + 1] = openmp_pixel_index[i] + openmp_pixel_contribs[i];
    }
    const unsigned int TOTAL_CONTRIBS = openmp_pixel_index[openmp_output_image.width * openmp_output_image.height];
    if (TOTAL_CONTRIBS > openmp_pixel_contrib_count) {
        if (openmp_pixel_contrib_colours) free(openmp_pixel_contrib_colours);
        if (openmp_pixel_contrib_depth) free(openmp_pixel_contrib_depth);
        openmp_pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        openmp_pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        openmp_pixel_contrib_count = TOTAL_CONTRIBS;
    }

    memset(openmp_pixel_contribs, 0, openmp_output_image.width * openmp_output_image.height * sizeof(unsigned int));

#pragma omp parallel for default(none) shared(openmp_particles, openmp_particles_count, openmp_output_image, openmp_pixel_contrib_colours, openmp_pixel_index, openmp_pixel_contribs, openmp_pixel_contrib_depth)
    for (unsigned int i = 0; i < openmp_particles_count; ++i) {
        Particle openmp_particle_i = openmp_particles[i];

        int x_min = (int)roundf(openmp_particle_i.location[0] - openmp_particle_i.radius);
        int y_min = (int)roundf(openmp_particle_i.location[1] - openmp_particle_i.radius);
        int x_max = (int)roundf(openmp_particle_i.location[0] + openmp_particle_i.radius);
        int y_max = (int)roundf(openmp_particle_i.location[1] + openmp_particle_i.radius);

        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= openmp_output_image.width ? openmp_output_image.width - 1 : x_max;
        y_max = y_max >= openmp_output_image.height ? openmp_output_image.height - 1 : y_max;

        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - openmp_particle_i.location[0];
                const float y_ab = (float)y + 0.5f - openmp_particle_i.location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= openmp_particle_i.radius) {
                    const unsigned int pixel_offset = y * openmp_output_image.width + x;

                    unsigned int storage_offset = openmp_pixel_index[pixel_offset];
                    unsigned int contris;
#pragma omp atomic capture
                    contris = openmp_pixel_contribs[pixel_offset]++;
                    storage_offset += contris;

                    memcpy(openmp_pixel_contrib_colours + (4 * storage_offset), openmp_particle_i.color, 4 * sizeof(unsigned char));
                    memcpy(openmp_pixel_contrib_depth + storage_offset, &openmp_particle_i.location[2], sizeof(float));
                }
            }
        }
    }

#pragma omp parallel for default(none) shared(openmp_pixel_contrib_depth, openmp_pixel_contrib_colours, openmp_pixel_index, openmp_output_image)
    for (int i = 0; i < openmp_output_image.width * openmp_output_image.height; ++i) {
        if (openmp_pixel_index[i] < openmp_pixel_index[i + 1] - 1) {
            cpu_sort_pairs(
                    openmp_pixel_contrib_depth,
                    openmp_pixel_contrib_colours,
                    openmp_pixel_index[i],
                    openmp_pixel_index[i + 1] - 1
            );
        }
    }
#ifdef VALIDATION
    validate_pixel_index(openmp_pixel_contribs, openmp_pixel_index, openmp_output_image.width,
                         openmp_output_image.height);
    validate_sorted_pairs(openmp_particles, openmp_particles_count, openmp_pixel_index, openmp_output_image.width,
                          openmp_output_image.height, openmp_pixel_contrib_colours, openmp_pixel_contrib_depth);
#endif
}
void openmp_stage3() {
    memset(openmp_output_image.data, 255, openmp_output_image.width * openmp_output_image.height * openmp_output_image.channels * sizeof(unsigned char));

#pragma omp parallel for default(none) shared(openmp_output_image, openmp_pixel_index, openmp_pixel_contrib_colours)
    for (int i = 0; i < openmp_output_image.width * openmp_output_image.height; ++i) {
        for (unsigned int j = openmp_pixel_index[i]; j < openmp_pixel_index[i + 1]; ++j) {
            const float opacity = (float)openmp_pixel_contrib_colours[j * 4 + 3] / (float)255;
            openmp_output_image.data[(i * 3) + 0] = (unsigned char)((float)openmp_pixel_contrib_colours[j * 4 + 0] * opacity + (float)openmp_output_image.data[(i * 3) + 0] * (1 - opacity));
            openmp_output_image.data[(i * 3) + 1] = (unsigned char)((float)openmp_pixel_contrib_colours[j * 4 + 1] * opacity + (float)openmp_output_image.data[(i * 3) + 1] * (1 - opacity));
            openmp_output_image.data[(i * 3) + 2] = (unsigned char)((float)openmp_pixel_contrib_colours[j * 4 + 2] * opacity + (float)openmp_output_image.data[(i * 3) + 2] * (1 - opacity));
        }
    }
#ifdef VALIDATION
    validate_blend(openmp_pixel_index, openmp_pixel_contrib_colours, &openmp_output_image);
#endif
}
void openmp_end(CImage* output_image) {
    // Store return value
    output_image->width = openmp_output_image.width;
    output_image->height = openmp_output_image.height;
    output_image->channels = openmp_output_image.channels;
    memcpy(output_image->data, openmp_output_image.data, openmp_output_image.width * openmp_output_image.height * openmp_output_image.channels * sizeof(unsigned char));
    // Release allocations
    free(openmp_pixel_contrib_depth);
    free(openmp_pixel_contrib_colours);
    free(openmp_output_image.data);
    free(openmp_pixel_index);
    free(openmp_pixel_contribs);
    free(openmp_particles);
    // Return ptrs to nullptr
    openmp_pixel_contrib_depth = 0;
    openmp_pixel_contrib_colours = 0;
    openmp_output_image.data = 0;
    openmp_pixel_index = 0;
    openmp_pixel_contribs = 0;
    openmp_particles = 0;
}