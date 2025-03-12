#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Computes safeguard score for a Gaussian primitive with respect to a ray.
//
// Calculation:
//  score = alpha * transmittance * color error (Manhattan distance)
// Output: a score in the range of [0, 1] for a Gaussian primitive with respect to a ray.
template<uint32_t C>
__device__ float compute_score(
	float opacity,
	float alpha,
	float T,
	const float* __restrict__ gt_color = nullptr,
	const float* prim_color = nullptr)
{
	float activated_color_dist_err = 0.0f;
	float color_dist_err = 0.0f;
	float score = 0.0f;

    score = alpha * T;
    for (int ch = 0; ch < C; ch++) {
        color_dist_err += abs(gt_color[ch] - prim_color[ch]);
    }
    activated_color_dist_err = 1 - color_dist_err / C;
    return score * activated_color_dist_err;
}


// __global__ void markingTopKMasks(
//     int L,
//     int topk,
//     const int64_t* __restrict__ point_list_keys,
//     const int32_t* __restrict__ point_list,
//     bool* __restrict__ masks,
//     int N  // ← NUEVO: tamaño de la máscara para chequear val
// ) {
//     auto idx = cg::this_grid().thread_rank();
//     if (idx >= L) return;

//     uint64_t key = point_list_keys[idx];
//     uint32_t val = point_list[idx]; // índice de la gaussiana

//     if (val >= N) {
//         printf("ERROR: val (%d) >= N (%d), idx = %d\n", static_cast<int>(val), static_cast<int>(N), idx);
//         return; // o "assert(false);" si compilas con CUDA DSA
//     }

//     uint32_t currtile = static_cast<uint32_t>(key >> 32);

//     if (idx < topk) {
//         masks[val] = true;
//     } else {
//         uint32_t prevtile = static_cast<uint32_t>(point_list_keys[idx - topk] >> 32);
//         if (currtile != prevtile) {
//             masks[val] = true;
//         }
//     }
// }

// __global__ void markingTopKMasks(
//     int L,
//     int topk,
//     const int64_t* point_list_keys,
//     const int32_t* point_list_values,
//     bool* mask // safeguard_topk_mask
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= L) return;

//     int64_t key = point_list_keys[idx];
//     int32_t gaussian_idx = point_list_values[idx];

//     uint32_t curr_tile = key >> 32;

//     if (idx < topk) {
//         mask[gaussian_idx] = true;
//     } else {
//         uint32_t prev_tile = point_list_keys[idx - topk] >> 32;
//         if (curr_tile != prev_tile) {
//             mask[gaussian_idx] = true;
//         }
//     }
// }


__global__ void markingTopKMasks(
    int L,
    int topk,
    const int64_t* point_list_keys,
    const int32_t* point_list_values,
    bool* mask // safeguard_topk_mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;

    int64_t key = point_list_keys[idx];
    int32_t gaussian_idx = point_list_values[idx];

    uint32_t curr_tile = key >> 32;

    if (idx < topk) {
        mask[gaussian_idx] = true;
        return;
    }

    // IMPORTANTE: si idx - topk < 0 no hacemos nada
    if (idx - topk < 0) return;

    int64_t prev_key = point_list_keys[idx - topk];
    uint32_t prev_tile = prev_key >> 32;

    if (curr_tile != prev_tile) {
        mask[gaussian_idx] = true;
    }
}

