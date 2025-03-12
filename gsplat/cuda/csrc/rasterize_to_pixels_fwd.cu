#include "bindings.h"
#include "types.cuh"
#include "compute_score_safeguard.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz]
    const S *__restrict__ backgrounds, // [C, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const bool use_safeguard,
    const uint32_t safeguard_prune_topk,
    int64_t *__restrict__ gaussian_keys_unsorted,
    int32_t *__restrict__ gaussian_values_unsorted,
    const float *__restrict__ image_gt, // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_colors, // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    bool *__restrict__ safeguard_topk_mask // [N]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    render_colors += camera_id * image_height * image_width * COLOR_DIM;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    S px = (S)j + 0.5f;
    S py = (S)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]
        ); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    S T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    S pix_out[COLOR_DIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> conic = conic_batch[t];
            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;
            const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
            const S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
            S alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const S next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const S vis = alpha * T;
            const S *c_ptr = colors + g * COLOR_DIM;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            cur_idx = batch_start + t;

            if (use_safeguard) {
                // Compute safeguard score para esta gaussiana
                // Puedes ajustar la implementación dependiendo de cómo quieras activarlo.
                // Ejemplo simple, puedes usar visibilidad `vis` como proxy, o algo más avanzado.
                
                // Por simplicidad uso vis aquí
                float prim_color[COLOR_DIM] = {0};
                for (uint32_t ch = 0; ch < COLOR_DIM; ++ch) {
                    prim_color[ch] = colors[g * COLOR_DIM + ch];
                }
                float safeguard_score = compute_score<COLOR_DIM>(
                    opac, // opacity
                    alpha,
                    T,
                    image_gt,
                    reinterpret_cast<float*>(prim_color)
                );

                uint64_t block_id = block.group_index().y * tile_width + block.group_index().z;
                // Orden descendente -> invertimos el score
                uint32_t scaled_score = __float2uint_rd((1.0f - safeguard_score) * 65535.0f);
            
                // Guardamos clave: [tile_id (high 32 bits)] | [scaled_score (low 32 bits)]
                gaussian_keys_unsorted[cur_idx] = (static_cast<int64_t>(block_id) << 32) | scaled_score;

                int32_t gaussian_idx = g % N;
                gaussian_values_unsorted[cur_idx] = gaussian_idx;
            }
            

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    // safeguard parameters
    const bool use_safeguard,
    const uint32_t safeguard_prune_topk,
    const at::optional<torch::Tensor> &image_gt // [C, image_height, image_width, channels]
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);

    if (use_safeguard) {
        GSPLAT_CHECK_INPUT(image_gt.value());
    }

    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor renders = torch::empty(
        {C, image_height, image_width, channels},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor alphas = torch::empty(
        {C, image_height, image_width, 1},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor last_ids = torch::empty(
        {C, image_height, image_width}, means2d.options().dtype(torch::kInt32)
    );

    torch::Tensor gaussian_keys_unsorted = torch::empty(
        {n_isects},
        means2d.options().dtype(torch::kInt64)
    );

    torch::Tensor gaussian_values_unsorted = torch::empty(
        {n_isects},
        means2d.options().dtype(torch::kInt32)
    );

    torch::Tensor safeguard_topk_mask = torch::empty(
        {N},
        means2d.options().dtype(torch::kBool)
    );

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }
    rasterize_to_pixels_fwd_kernel<CDIM, float>
        <<<blocks, threads, shared_mem, stream>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            use_safeguard,
            safeguard_prune_topk,
            gaussian_keys_unsorted.data_ptr<int64_t>(),
            gaussian_values_unsorted.data_ptr<int32_t>(),
            image_gt.has_value() ? image_gt.value().data_ptr<float>() : nullptr,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            safeguard_topk_mask.data_ptr<bool>()
        );

        if (use_safeguard && n_isects > 0) {
            // 1. Ordena gaussian_keys_unsorted y gaussian_values_unsorted
            size_t temp_storage_bytes = 0;
            void* d_temp_storage = nullptr;
        
            // Primero consulta el tamaño del workspace requerido
            cub::DeviceRadixSort::SortPairs(
                nullptr, temp_storage_bytes,
                gaussian_keys_unsorted.data_ptr<int64_t>(), 
                gaussian_keys_unsorted.data_ptr<int64_t>(), // output buffer igual si in-place
                gaussian_values_unsorted.data_ptr<int32_t>(), 
                gaussian_values_unsorted.data_ptr<int32_t>(), // output buffer igual si in-place
                flatten_ids.size(0) // n_isects
            );
        
            // Reserva almacenamiento temporal
            d_temp_storage = torch::empty(
                {static_cast<int64_t>(temp_storage_bytes)}, 
                means2d.options().dtype(torch::kUInt8)
            ).data_ptr();
        
            // Ejecuta la ordenación
            cub::DeviceRadixSort::SortPairs(
                d_temp_storage, temp_storage_bytes,
                gaussian_keys_unsorted.data_ptr<int64_t>(), 
                gaussian_keys_unsorted.data_ptr<int64_t>(),
                gaussian_values_unsorted.data_ptr<int32_t>(), 
                gaussian_values_unsorted.data_ptr<int32_t>(), 
                flatten_ids.size(0) // n_isects
            );
        
            // 2. Lanza el kernel markingTopKMasks
            const int threads = 256;
            const int blocks = (flatten_ids.size(0) + threads - 1) / threads;
            markingTopKMasks<<<blocks, threads, 0, stream>>>(
                flatten_ids.size(0),
                safeguard_prune_topk,
                gaussian_keys_unsorted.data_ptr<int64_t>(),
                gaussian_values_unsorted.data_ptr<int32_t>(),
                safeguard_topk_mask.data_ptr<bool>()
            );

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {

                printf("MEANS2D SIZE: [%ld, %ld]\n", means2d.size(0), means2d.size(1));
                printf("GAUSSIAN_VALUES_UNSORTED SIZE: [%ld]\n", gaussian_values_unsorted.size(0));
                printf("FLATTEN_IDS SIZE: [%ld]\n", flatten_ids.size(0));
                printf("safeguard_topk_mask SIZE: [%ld]\n", safeguard_topk_mask.size(0));
                printf("CUDA error after markingTopKMasks kernel: %s\n", cudaGetErrorString(err));
            }
        }        
    
    return std::make_tuple(renders, alphas, last_ids, safeguard_topk_mask);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,   // [n_isects]
    const bool use_safeguard,
    const uint32_t safeguard_prune_topk,
    const at::optional<torch::Tensor> &image_gt // [C, image_height, image_width, channels]
) {
    GSPLAT_CHECK_INPUT(colors);
    uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim<N>(                                        \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            use_safeguard,                                                     \
            safeguard_prune_topk,                                              \
            image_gt                                                           \
        );

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    switch (channels) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
}

} // namespace gsplat



// #include "bindings.h"
// #include "types.cuh"
// #include "compute_score_safeguard.cuh"
// #include <cooperative_groups.h>
// #include <cub/cub.cuh>
// #include <thrust/device_ptr.h>
// #include <thrust/sort.h>
// #include <cuda_runtime.h>

// namespace gsplat {

// namespace cg = cooperative_groups;

// /****************************************************************************
//  * Rasterization to Pixels Forward Pass
//  ****************************************************************************/

// template <uint32_t COLOR_DIM, typename S>
// __global__ void rasterize_to_pixels_fwd_kernel(
//     const uint32_t C,
//     const uint32_t N,
//     const uint32_t n_isects,
//     const bool packed,
//     const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
//     const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
//     const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
//     const S *__restrict__ opacities,   // [C, N] or [nnz]
//     const S *__restrict__ backgrounds, // [C, COLOR_DIM]
//     const bool *__restrict__ masks,    // [C, tile_height, tile_width]
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const uint32_t tile_size,
//     const uint32_t tile_width,
//     const uint32_t tile_height,
//     const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
//     const int32_t *__restrict__ flatten_ids,  // [n_isects]
//     const bool use_safeguard,
//     const uint32_t safeguard_prune_topk,
//     int64_t *__restrict__ gaussian_keys_unsorted,
//     int32_t *__restrict__ gaussian_values_unsorted,
//     const float *__restrict__ image_gt, // [C, image_height, image_width, COLOR_DIM]
//     S *__restrict__ render_colors, // [C, image_height, image_width, COLOR_DIM]
//     S *__restrict__ render_alphas, // [C, image_height, image_width, 1]
//     int32_t *__restrict__ last_ids, // [C, image_height, image_width]
//     bool *__restrict__ safeguard_topk_mask // [N]
// ) {
//     // each thread draws one pixel, but also timeshares caching gaussians in a
//     // shared tile

//     // auto block = cg::this_thread_block();
//     // int32_t camera_id = block.group_index().x;
//     // int32_t tile_id =
//     //     block.group_index().y * tile_width + block.group_index().z;
//     // uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
//     // uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

//     /* NUEVO CODIGO*/
//     auto block = cg::this_thread_block();
//     uint32_t horizontal_blocks = (image_width + tile_size - 1) / tile_size;
//     uint64_t block_id = block.group_index().y * horizontal_blocks + block.group_index().z;

//     int32_t camera_id = block.group_index().x;
//     int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;

//     uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
//     uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

//     uint32_t pix_id = i * image_width + j;

//     S px = (S)j + 0.5f;
//     S py = (S)i + 0.5f;
//     // NUEVO CODIGO

//     tile_offsets += camera_id * tile_height * tile_width;
//     render_colors += camera_id * image_height * image_width * COLOR_DIM;
//     render_alphas += camera_id * image_height * image_width;
//     last_ids += camera_id * image_height * image_width;
//     if (backgrounds != nullptr) {
//         backgrounds += camera_id * COLOR_DIM;
//     }
//     if (masks != nullptr) {
//         masks += camera_id * tile_height * tile_width;
//     }
//     // return if out of bounds
//     // keep not rasterizing threads around for reading data
//     bool inside = (i < image_height && j < image_width);
//     bool done = !inside;

//     // when the mask is provided, render the background color and return
//     // if this tile is labeled as False
//     if (masks != nullptr && inside && !masks[tile_id]) {
//         for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//             render_colors[pix_id * COLOR_DIM + k] =
//                 backgrounds == nullptr ? 0.0f : backgrounds[k];
//         }
//         return;
//     }

//     // have all threads in tile process the same gaussians in batches
//     // first collect gaussians between range.x and range.y in batches
//     // which gaussians to look through in this tile
//     int32_t range_start = tile_offsets[tile_id];
//     int32_t range_end =
//         (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
//             ? n_isects
//             : tile_offsets[tile_id + 1];
//     const uint32_t block_size = block.size();
//     uint32_t num_batches =
//         (range_end - range_start + block_size - 1) / block_size;

//     extern __shared__ int s[];
//     int32_t *id_batch = (int32_t *)s; // [block_size]
//     vec3<S> *xy_opacity_batch =
//         reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
//     vec3<S> *conic_batch =
//         reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]
//         ); // [block_size]

//     // current visibility left to render
//     // transmittance is gonna be used in the backward pass which requires a high
//     // numerical precision so we use double for it. However double make bwd 1.5x
//     // slower so we stick with float for now.
//     S T = 1.0f;
//     // index of most recent gaussian to write to this thread's pixel
//     uint32_t cur_idx = 0;

//     // collect and process batches of gaussians
//     // each thread loads one gaussian at a time before rasterizing its
//     // designated pixel
//     uint32_t tr = block.thread_rank();

//     S pix_out[COLOR_DIM] = {0.f};

//     for (uint32_t b = 0; b < num_batches; ++b) {
//         // resync all threads before beginning next batch
//         // end early if entire tile is done
//         if (__syncthreads_count(done) >= block_size) {
//             break;
//         }

//         // each thread fetch 1 gaussian from front to back
//         // index of gaussian to load
//         uint32_t batch_start = range_start + block_size * b;
//         uint32_t idx = batch_start + tr;
//         if (idx < range_end) {
//             int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
//             id_batch[tr] = g;
//             const vec2<S> xy = means2d[g];
//             const S opac = opacities[g];
//             xy_opacity_batch[tr] = {xy.x, xy.y, opac};
//             conic_batch[tr] = conics[g];
//         }

//         // wait for other threads to collect the gaussians in batch
//         block.sync();

//         // process gaussians in the current batch for this pixel
//         uint32_t batch_size = min(block_size, range_end - batch_start);
//         for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
//             const vec3<S> conic = conic_batch[t];
//             const vec3<S> xy_opac = xy_opacity_batch[t];
//             const S opac = xy_opac.z;
//             const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
//             const S sigma = 0.5f * (conic.x * delta.x * delta.x +
//                                     conic.z * delta.y * delta.y) +
//                             conic.y * delta.x * delta.y;
//             S alpha = min(0.999f, opac * __expf(-sigma));
//             if (sigma < 0.f || alpha < 1.f / 255.f) {
//                 continue;
//             }

//             const S next_T = T * (1.0f - alpha);
//             if (next_T <= 1e-4) { // this pixel is done: exclusive
//                 done = true;
//                 break;
//             }

//             int32_t g = id_batch[t];
//             const S vis = alpha * T;
//             if (use_safeguard) {
//                 float prim_color[COLOR_DIM] = {0};
//                 for (uint32_t ch = 0; ch < COLOR_DIM; ++ch) {
//                     prim_color[ch] = colors[g * COLOR_DIM + ch];
//                 }
//                 float score = compute_score<COLOR_DIM>(
//                     opac, // opacity
//                     alpha,
//                     T,
//                     image_gt,
//                     reinterpret_cast<float*>(prim_color)
//                 );
//                 // Convertir el score a un valor entre 0 y 65535, invertir para orden ascendente
//                 uint32_t scaled_score = __float2uint_rd((1.f - score) * 65535);

//                 // Asignar key y value en el array (misma lógica que en renderCUDA_topk_color)
//                 uint32_t global_gaussian_idx = batch_start + t;
//                 gaussian_keys_unsorted[global_gaussian_idx] = (block_id << 32) | scaled_score;
//                 gaussian_values_unsorted[global_gaussian_idx] = g;

//             }
//             const S *c_ptr = colors + g * COLOR_DIM;
//             GSPLAT_PRAGMA_UNROLL
//             for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//                 pix_out[k] += c_ptr[k] * vis;
//             }
//             cur_idx = batch_start + t;

//             T = next_T;
//         }
//     }

//     if (inside) {
//         // Here T is the transmittance AFTER the last gaussian in this pixel.
//         // We (should) store double precision as T would be used in backward
//         // pass and it can be very small and causing large diff in gradients
//         // with float32. However, double precision makes the backward pass 1.5x
//         // slower so we stick with float for now.
//         render_alphas[pix_id] = 1.0f - T;
//         GSPLAT_PRAGMA_UNROLL
//         for (uint32_t k = 0; k < COLOR_DIM; ++k) {
//             render_colors[pix_id * COLOR_DIM + k] =
//                 backgrounds == nullptr ? pix_out[k]
//                                        : (pix_out[k] + T * backgrounds[k]);
//         }
//         // index in bin of last gaussian in this pixel
//         last_ids[pix_id] = static_cast<int32_t>(cur_idx);
//     }
// }

// template <uint32_t CDIM>
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> call_kernel_with_dim(
//     // Gaussian parameters
//     const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
//     const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
//     const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
//     const torch::Tensor &opacities, // [C, N]  or [nnz]
//     const at::optional<torch::Tensor> &backgrounds, // [C, channels]
//     const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
//     // image size
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const uint32_t tile_size,
//     // intersections
//     const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
//     const torch::Tensor &flatten_ids,   // [n_isects]
//     const bool use_safeguard,
//     const uint32_t safeguard_prune_topk,
//     const at::optional<torch::Tensor> &image_gt // [C, image_height, image_width, channels]
// ) {
//     GSPLAT_DEVICE_GUARD(means2d);
//     GSPLAT_CHECK_INPUT(means2d);
//     GSPLAT_CHECK_INPUT(conics);
//     GSPLAT_CHECK_INPUT(colors);
//     GSPLAT_CHECK_INPUT(opacities);
//     GSPLAT_CHECK_INPUT(tile_offsets);
//     GSPLAT_CHECK_INPUT(flatten_ids);
//     if (backgrounds.has_value()) {
//         GSPLAT_CHECK_INPUT(backgrounds.value());
//     }
//     if (masks.has_value()) {
//         GSPLAT_CHECK_INPUT(masks.value());
//     }
//     if (use_safeguard) {
//         GSPLAT_CHECK_INPUT(image_gt.value());
//     }
    
//     bool packed = means2d.dim() == 2;

//     uint32_t C = tile_offsets.size(0);         // number of cameras
//     uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
//     uint32_t channels = colors.size(-1);
//     uint32_t tile_height = tile_offsets.size(1);
//     uint32_t tile_width = tile_offsets.size(2);
//     uint32_t n_isects = flatten_ids.size(0);

//     // Each block covers a tile on the image. In total there are
//     // C * tile_height * tile_width blocks.
//     dim3 threads = {tile_size, tile_size, 1};
//     dim3 blocks = {C, tile_height, tile_width};

//     torch::Tensor renders = torch::empty(
//         {C, image_height, image_width, channels},
//         means2d.options().dtype(torch::kFloat32)
//     );
//     torch::Tensor alphas = torch::empty(
//         {C, image_height, image_width, 1},
//         means2d.options().dtype(torch::kFloat32)
//     );
//     torch::Tensor last_ids = torch::empty(
//         {C, image_height, image_width}, means2d.options().dtype(torch::kInt32)
//     );

//     torch::Tensor gaussian_keys_unsorted = torch::empty(
//         {n_isects},
//         means2d.options().dtype(torch::kInt64)
//     );
    
//     torch::Tensor gaussian_values_unsorted = torch::empty(
//         {n_isects},
//         means2d.options().dtype(torch::kInt32)
//     );
    

//     torch::Tensor safeguard_topk_mask = torch::zeros(
//         {N}, means2d.options().dtype(torch::kBool)
//     );
    
//     at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
//     const uint32_t shared_mem =
//         tile_size * tile_size *
//         (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));

//     // TODO: an optimization can be done by passing the actual number of
//     // channels into the kernel functions and avoid necessary global memory
//     // writes. This requires moving the channel padding from python to C side.
//     if (cudaFuncSetAttribute(
//             rasterize_to_pixels_fwd_kernel<CDIM, float>,
//             cudaFuncAttributeMaxDynamicSharedMemorySize,
//             shared_mem
//         ) != cudaSuccess) {
//         AT_ERROR(
//             "Failed to set maximum shared memory size (requested ",
//             shared_mem,
//             " bytes), try lowering tile_size."
//         );
//     }
    // rasterize_to_pixels_fwd_kernel<CDIM, float>
    //     <<<blocks, threads, shared_mem, stream>>>(
    //         C,
    //         N,
    //         n_isects,
    //         packed,
    //         reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
    //         reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
    //         colors.data_ptr<float>(),
    //         opacities.data_ptr<float>(),
    //         backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
    //                                 : nullptr,
    //         masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
    //         image_width,
    //         image_height,
    //         tile_size,
    //         tile_width,
    //         tile_height,
    //         tile_offsets.data_ptr<int32_t>(),
    //         flatten_ids.data_ptr<int32_t>(),
    //         use_safeguard,
    //         safeguard_prune_topk,
    //         gaussian_keys_unsorted.data_ptr<int64_t>(),
    //         gaussian_values_unsorted.data_ptr<int32_t>(),
    //         image_gt.has_value() ? image_gt.value().data_ptr<float>() : nullptr,
    //         renders.data_ptr<float>(),
    //         alphas.data_ptr<float>(),
    //         last_ids.data_ptr<int32_t>(),
    //         safeguard_topk_mask.data_ptr<bool>()
    //     );

//     // Host code:
//     if (use_safeguard) {
//         // Obtener el stream actual
//         at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
//         cudaStream_t cuda_stream = stream.stream();
        
//         const int n_items = n_isects;
//         const int n_bits = 64; // Estamos usando claves de 64 bits
    
//         // Crear tensores para claves y valores ordenados
//         torch::Tensor gaussian_keys_sorted = torch::empty(
//             {n_items}, means2d.options().dtype(torch::kInt64)
//         );
    
//         torch::Tensor gaussian_values_sorted = torch::empty(
//             {n_items}, means2d.options().dtype(torch::kInt32)
//         );
    
//         // Workspace de CUB para el radix sort
//         size_t temp_storage_bytes = 0;
//         void *d_temp_storage = nullptr;
    
//         // Paso 1: obtener el tamaño del espacio de trabajo necesario
//         cub::DeviceRadixSort::SortPairs(
//             d_temp_storage,
//             temp_storage_bytes,
//             gaussian_keys_unsorted.data_ptr<int64_t>(),  // claves de entrada (uint64_t/int64_t)
//             gaussian_keys_sorted.data_ptr<int64_t>(),    // claves ordenadas
//             gaussian_values_unsorted.data_ptr<int32_t>(),// valores de entrada
//             gaussian_values_sorted.data_ptr<int32_t>(),  // valores ordenados
//             n_items,
//             0,                                           // bit inicial
//             n_bits,                                      // cantidad de bits
//             cuda_stream
//         );
    
//         // Paso 2: reservar el espacio de trabajo
//         d_temp_storage = torch::empty(
//             {static_cast<int64_t>(temp_storage_bytes)},
//             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)
//         ).data_ptr();
    
//         // Paso 3: lanzar el sort real
//         cub::DeviceRadixSort::SortPairs(
//             d_temp_storage,
//             temp_storage_bytes,
//             gaussian_keys_unsorted.data_ptr<int64_t>(),  // claves de entrada
//             gaussian_keys_sorted.data_ptr<int64_t>(),    // claves ordenadas
//             gaussian_values_unsorted.data_ptr<int32_t>(),// valores de entrada
//             gaussian_values_sorted.data_ptr<int32_t>(),  // valores ordenados
//             n_items,
//             0,
//             n_bits,
//             cuda_stream
//         );
    
//         // Paso 4: lanzar el kernel markingTopKMasks
//         const int threads = 256;
//         const int blocks = (n_items + threads - 1) / threads;

//         markingTopKMasks<<<blocks, threads, 0, cuda_stream>>>(
//             n_items,
//             safeguard_prune_topk,
//             gaussian_keys_sorted.data_ptr<int64_t>(),
//             gaussian_values_sorted.data_ptr<int32_t>(),
//             safeguard_topk_mask.data_ptr<bool>(), // Tamaño N
//             N
//         );

//         cudaDeviceSynchronize();
//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             printf("n_items: %d\n", n_items);
//             printf("MEANS2D SIZE: [%ld, %ld]\n", means2d.size(0), means2d.size(1));
//             printf("CUDA error after markingTopKMasks kernel: %s\n", cudaGetErrorString(err));
//         }
//     }

//     return std::make_tuple(renders, alphas, last_ids, safeguard_topk_mask);
//     // return std::make_tuple(renders, alphas, last_ids, safeguard_topk_mask);
// }

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
// rasterize_to_pixels_fwd_tensor(
//     // Gaussian parameters
//     const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
//     const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
//     const torch::Tensor &colors,    // [C, N, channels] or [nnz, channels]
//     const torch::Tensor &opacities, // [C, N]  or [nnz]
//     const at::optional<torch::Tensor> &backgrounds, // [C, channels]
//     const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
//     // image size
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const uint32_t tile_size,
//     // intersections
//     const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
//     const torch::Tensor &flatten_ids,   // [n_isects]
//     const bool use_safeguard,
//     const uint32_t safeguard_prune_topk,
//     const at::optional<torch::Tensor> &image_gt // [C, image_height, image_width, channels]
// ) {
//     GSPLAT_CHECK_INPUT(colors);
//     uint32_t channels = colors.size(-1);

// #define __GS__CALL_(N)                                                         \
//     case N:                                                                    \
//         return call_kernel_with_dim<N>(                                        \
//             means2d,                                                           \
//             conics,                                                            \
//             colors,                                                            \
//             opacities,                                                         \
//             backgrounds,                                                       \
//             masks,                                                             \
//             image_width,                                                       \
//             image_height,                                                      \
//             tile_size,                                                         \
//             tile_offsets,                                                      \
//             flatten_ids,                                                       \
//             use_safeguard,                                                     \
//             safeguard_prune_topk,                                              \
//             image_gt                                                           \
//         );

//     // TODO: an optimization can be done by passing the actual number of
//     // channels into the kernel functions and avoid necessary global memory
//     // writes. This requires moving the channel padding from python to C side.
//     switch (channels) {
//         __GS__CALL_(1)
//         __GS__CALL_(2)
//         __GS__CALL_(3)
//         __GS__CALL_(4)
//         __GS__CALL_(5)
//         __GS__CALL_(8)
//         __GS__CALL_(9)
//         __GS__CALL_(16)
//         __GS__CALL_(17)
//         __GS__CALL_(32)
//         __GS__CALL_(33)
//         __GS__CALL_(64)
//         __GS__CALL_(65)
//         __GS__CALL_(128)
//         __GS__CALL_(129)
//         __GS__CALL_(256)
//         __GS__CALL_(257)
//         __GS__CALL_(512)
//         __GS__CALL_(513)
//     default:
//         AT_ERROR("Unsupported number of channels: ", channels);
//     }
// }

// } // namespace gsplat