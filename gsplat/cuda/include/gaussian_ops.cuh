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

__device__ inline int32_t processTiles(
    const float4 con_o,
    const float disc,
    const float t,
    const float2 p,
    float2 bbox_min,
    float2 bbox_max,
    float2 bbox_argmin,
    float2 bbox_argmax,
    int2 rect_min,
    int2 rect_max,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    uint32_t idx,
    int64_t off,
    const int64_t cid_enc,
    int64_t depth_id_enc,
    int64_t *isect_ids,
    int32_t *flatten_ids
) {

    // ---- AccuTile Code ---- //

    // Set variables based on the isY flag
    float BLOCK_U = tile_size;

    // Compute isY
    int x_span = rect_max.x - rect_min.x;
    int y_span = rect_max.y - rect_min.y;

    // If no tiles are touched, return 0
    if (y_span * x_span == 0) {
        return 0;
    }

    bool isY = y_span < x_span;

    if (isY) {
        rect_min = {rect_min.y, rect_min.x};
        rect_max = {rect_max.y, rect_max.x};

        bbox_min = {bbox_min.y, bbox_min.x};
        bbox_max = {bbox_max.y, bbox_max.x};

        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    int32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    min_line = rect_min.x * BLOCK_U;

    // Initialize min line intersections.
    if (bbox_min.x <= min_line) {
        // Boundary case
        intersect_min_line = computeEllipseIntersection(
            con_o, disc, t, p, isY, rect_min.x * BLOCK_U
        );

    } else {
        // Just need the min to be >= all points on the ellipse and
        // max to be <= all points on the ellipse
        intersect_min_line = {bbox_max.y, bbox_min.y};
    }

    // Loop over either y slices or x slices based on the isY flag.
    for (int32_t u = rect_min.x; u < rect_max.x; ++u) {
        // Starting from the bottom or left, we will only need to compute
        // intersections at the next line.
        max_line = min_line + BLOCK_U;

        if (max_line <= bbox_max.x) {
            intersect_max_line =
                computeEllipseIntersection(con_o, disc, t, p, isY, max_line);
        }

        // If the bbox min is in this slice, then it is the minimum
        // ellipse point in this slice. Otherwise, the minimum ellipse
        // point will be the minimum of the intersections of the min/maxlines.
        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) {
            ellipse_min = bbox_min.y;
        } else {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        // If the bbox max is in this slice, then it is the maximum
        // ellipse point in this slice. Otherwise, the maximum ellipse
        // point will be the maximum of the intersections of the min/maxlines.
        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) {
            ellipse_max = bbox_max.y;
        } else {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        // Convert ellipse_min/ellipse_max to tiles touched
        // First map back to tile coordinates, then subtract.
        int min_tile_v =
            max(rect_min.y, min(rect_max.y, (int)(ellipse_min / 16.0f)));
        int max_tile_v =
            min(rect_max.y, max(rect_min.y, (int)(ellipse_max / 16.0f + 1.0f)));

        tiles_count += max_tile_v - min_tile_v;
        // Only update keys array if it exists.
        if (isect_ids != nullptr && flatten_ids != nullptr) {
            // Loop over tiles and add to keys array
            for (int32_t v = min_tile_v; v < max_tile_v; v++) {
                // For each tile that the Gaussian overlaps, emit a
                // key/value pair. The key is |  tile ID  |      depth |,
                // and the value is the ID of the Gaussian. Sorting the values
                // with this key yields Gaussian IDs in a list, such that they
                // are first sorted by tile and then by depth.
                int64_t key = isY ? (u * tile_width + v) : (v * tile_width + u);

                isect_ids[off] = cid_enc | (key << 32) | depth_id_enc;
                flatten_ids[off] = static_cast<int32_t>(idx);
                off++;
            }
        }
        // Max line of this tile slice will be min lin of next tile slice
        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }

    return tiles_count;
}

// Define a struct to store the tile information
struct TileBounds {
    int2 rect_min; // The minimum x and y coordinates of the touched tiles
    int2 rect_max; // The maximum x and y coordinates of the touched tiles
    float2 bbox_argmin;
    float2 bbox_argmax;
    float2 bbox_min;
    float2 bbox_max;
    float disc;
    float t;
};

// Updated function that returns TileBounds struct
__device__ inline TileBounds duplicateToTilesTouched(
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
        max(0, min((int)tile_width, (int)(bbox_min.x / 16.0f))),
        max(0, min((int)tile_height, (int)(bbox_min.y / 16.0f)))
    };
    int2 rect_max = {
        max(0, min((int)tile_width, (int)(bbox_max.x / 16.0f + 1.0f))),
        max(0, min((int)tile_height, (int)(bbox_max.y / 16.0f + 1.0f)))
    };

    return TileBounds{
        rect_min,
        rect_max,
        bbox_argmin,
        bbox_argmax,
        bbox_min,
        bbox_max,
        disc,
        t,
    };
}

#endif // GAUSSIAN_OPS_H
