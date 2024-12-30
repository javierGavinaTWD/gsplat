// gaussian_ops.cuh
#ifndef GAUSSIAN_OPS_H
#define GAUSSIAN_OPS_H

#include "types.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float2 computeEllipseIntersection(
    const float4 con_o,
    const float disc,
    const float t,
    const float2 p,
    const bool isY,
    const float coord
) {
    float p_u = isY ? p.y : p.x;
    float p_v = isY ? p.x : p.y;
    float coeff = isY ? con_o.x : con_o.z;

    float h = coord - p_u; // h = y - p.y for y, x - p.x for x
    float h_sq = h * h;
    float sqrt_term = sqrt(disc * h_sq + t * coeff);

    float denom = coeff != 0.0f ? coeff : 1.0f; // Avoid division by zero
    return {
        (-con_o.y * h - sqrt_term) / denom + p_v,
        (-con_o.y * h + sqrt_term) / denom + p_v
    };
}

// __device__ inline uint32_t processTiles(
//     const float4 con_o,
//     const float disc,
//     const float t,
//     const float2 p,
//     float2 bbox_min,
//     float2 bbox_max,
//     float2 bbox_argmin,
//     float2 bbox_argmax,
//     int2 rect_min,
//     int2 rect_max,
//     const uint32_t tile_size,
//     const uint32_t tile_width,
//     const uint32_t tile_height,
//     const bool isY,
//     uint32_t idx,
//     uint32_t off,
//     float depth,
//     uint64_t *gaussian_keys_unsorted,
//     uint32_t *gaussian_values_unsorted
// ) {

//     // ---- AccuTile Code ---- //

//     // Set variables based on the isY flag
//     float BLOCK_U = tile_size;
//     float BLOCK_V = tile_size;

//     if (isY) {
//         rect_min = {rect_min.y, rect_min.x};
//         rect_max = {rect_max.y, rect_max.x};

//         bbox_min = {bbox_min.y, bbox_min.x};
//         bbox_max = {bbox_max.y, bbox_max.x};

//         bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
//         bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
//     }

//     uint32_t tiles_count = 0;
//     float2 intersect_min_line, intersect_max_line;
//     float ellipse_min, ellipse_max;
//     float min_line, max_line;

//     min_line = rect_min.x * BLOCK_U;
//     // Initialize min line intersections.
//     if (bbox_min.x <= min_line) {
//         // Boundary case
//         intersect_min_line = computeEllipseIntersection(
//             con_o, disc, t, p, isY, rect_min.x * BLOCK_U
//         );

//     } else {
//         // Just need the min to be >= all points on the ellipse and
//         // max to be <= all points on the ellipse
//         intersect_min_line = {bbox_max.y, bbox_min.y};
//     }

//     // Loop over either y slices or x slices based on the `isY` flag.
//     for (int u = rect_min.x; u < rect_max.x; ++u) {
//         // Starting from the bottom or left, we will only need to compute
//         // intersections at the next line.
//         max_line = min_line + BLOCK_U;
//         if (max_line <= bbox_max.x) {
//             intersect_max_line =
//                 computeEllipseIntersection(con_o, disc, t, p, isY, max_line);
//         }

//         // If the bbox min is in this slice, then it is the minimum
//         // ellipse point in this slice. Otherwise, the minimum ellipse
//         // point will be the minimum of the intersections of the min/max
//         lines. if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
//             ellipse_min = bbox_min.y;
//         } else {
//             ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
//         }

//         // If the bbox max is in this slice, then it is the maximum
//         // ellipse point in this slice. Otherwise, the maximum ellipse
//         // point will be the maximum of the intersections of the min/max
//         lines. if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
//             ellipse_max = bbox_max.y;
//         } else {
//             ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
//         }

//         // Convert ellipse_min/ellipse_max to tiles touched
//         // First map back to tile coordinates, then subtract.
//         int min_tile_v =
//             max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK_V)));
//         int max_tile_v =
//             min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK_V +
//             1)));

//         tiles_count += max_tile_v - min_tile_v;
//         // Only update keys array if it exists.
//         if (gaussian_keys_unsorted != nullptr) {
//             // Loop over tiles and add to keys array
//             for (int v = min_tile_v; v < max_tile_v; v++) {
//                 // For each tile that the Gaussian overlaps, emit a
//                 // key/value pair. The key is |  tile ID  |      depth |,
//                 // and the value is the ID of the Gaussian. Sorting the
//                 values
//                 // with this key yields Gaussian IDs in a list, such that
//                 they
//                 // are first sorted by tile and then by depth.
//                 uint64_t key =
//                     isY ? (u * tile_width + v) : (v * tile_width + u);
//                 key <<= 32;
//                 key |= *((uint32_t *)&depth);
//                 gaussian_keys_unsorted[off] = key;
//                 gaussian_values_unsorted[off] = idx;
//                 off++;
//             }
//         }
//         // Max line of this tile slice will be min lin of next tile slice
//         intersect_min_line = intersect_max_line;
//         min_line = max_line;
//     }
//     return tiles_count;
// }

// Define a struct to store the tile information
struct TileBounds {
    int2 rect_min; // The minimum x and y coordinates of the touched tiles
    int2 rect_max; // The maximum x and y coordinates of the touched tiles
};

// Updated function that returns TileBounds struct
__device__ __forceinline__ TileBounds duplicateToTilesTouched(
    const float2 p,
    const float4 con_o,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    // Calculate discriminant
    float disc = con_o.y * con_o.y - con_o.x * con_o.z;

    // If ill-formed ellipse, return a struct with zero values
    if (con_o.x <= 0 || con_o.z <= 0 || disc >= 0) {
        return TileBounds{{0, 0}, {0, 0}}; // No tiles touched
    }

    // Threshold: opacity * Gaussian = 1 / 255
    float t = 2.0f * log(con_o.w * 255.0f);

    // Compute bounding box terms
    float inv_disc_x = 1.0f / (disc * con_o.x);
    float inv_disc_y = 1.0f / (disc * con_o.z);
    float x_term = sqrt(-con_o.y * con_o.y * t * inv_disc_x);
    float y_term = sqrt(-con_o.y * con_o.y * t * inv_disc_y);
    x_term = (con_o.y < 0) ? x_term : -x_term;
    y_term = (con_o.y < 0) ? y_term : -y_term;

    float2 bbox_argmin = {p.y - y_term, p.x - x_term};
    float2 bbox_argmax = {p.y + y_term, p.x + x_term};

    float2 bbox_min = {
        computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmin.x).x,
        computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmin.y).x
    };
    float2 bbox_max = {
        computeEllipseIntersection(con_o, disc, t, p, true, bbox_argmax.x).y,
        computeEllipseIntersection(con_o, disc, t, p, false, bbox_argmax.y).y
    };

    // Rectangular tile extent of ellipse
    int2 rect_min = {
        min((int)tile_width, max(0, (int)floorf(bbox_min.x / 16.0f))),
        min((int)tile_height, max(0, (int)floorf(bbox_min.y / 16.0f)))
    };
    int2 rect_max = {
        min((int)tile_width, max(0, (int)ceilf(bbox_max.x / 16.0f + 1.0f))),
        min((int)tile_height, max(0, (int)ceilf(bbox_max.y / 16.0f + 1.0f)))
    };

    return TileBounds{rect_min, rect_max};
}

// // If fewer y tiles, loop over y slices else loop over x slices
// bool isY = y_span < x_span;
// return processTiles(
//     con_o,
//     disc,
//     t,
//     p,
//     bbox_min,
//     bbox_max,
//     bbox_argmin,
//     bbox_argmax,
//     rect_min,
//     rect_max,
//     tile_size,
//     tile_width,
//     tile_height,
//     isY,
//     idx,
//     off,
//     depth,
//     gaussian_keys_unsorted,
//     gaussian_values_unsorted
// );

#endif // GAUSSIAN_OPS_H
