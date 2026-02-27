#include "configs.cuh"
#include "exception.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"

namespace deep_ep {

namespace internode_ll {

template <bool use_warp_sync = false>
__forceinline__ __device__ bool is_rank_masked(int* mask_buffer_ptr, int rank) {
    if (mask_buffer_ptr == nullptr) {
        return false;
    }
    if constexpr (use_warp_sync) {
        // 只执行1次全局内存访问 + 硬件广播
        return __shfl_sync(0xffffffff, ld_acquire_global(mask_buffer_ptr + rank), 0) != 0;
    } else {
        return ld_acquire_global(mask_buffer_ptr + rank) != 0;
    }
}

template <int kNumThreads>
__forceinline__ __device__ void barrier(int thread_id, int rank, int num_ranks, int* mask_buffer_ptr, int* sync_buffer_ptr) {
    EP_DEVICE_ASSERT(kNumThreads >= num_ranks);

    // Quiet all QPs
    auto qps_per_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;

    for (int i = thread_id; i < qps_per_rank * (num_ranks - 1); i += kNumThreads) {
        auto dst_rank = (rank + 1 + i / qps_per_rank) % num_ranks;
        auto qp_id = i % qps_per_rank;
        // 等待指定目标 PE 的指定 QP 上所有已提交的 RDMA 操作完成。用于确保在清理缓冲区或进行同步操作前，所有未完成的 RDMA 写操作已完成。
        nvshmemi_ibgda_quiet(dst_rank, qp_id);
    }

    // Update local counter
    if (thread_id == 0)
        atomicAdd(sync_buffer_ptr + rank, -1);
    __syncthreads();

    int cnt = sync_buffer_ptr[rank];
    // Update remote counter and wait for local counter to be updated
    if (thread_id < num_ranks && thread_id != rank) {
        const auto dst_rank = thread_id;
        const auto dst_ptr = reinterpret_cast<uint64_t>(sync_buffer_ptr + rank);
        const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);

        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
            if (dst_p2p_ptr == 0) {
                nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), cnt, dst_rank, 0);
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), cnt);
            }

            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            while (ld_acquire_sys_global(sync_buffer_ptr + dst_rank) != cnt            // remote is not ready
                   && (wait_recv_cost = clock64() - start_time) <= NUM_TIMEOUT_CYCLES  // not timeout
            )
                ;
            // Mask rank if timeout
            if (wait_recv_cost > NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for barrier, rank %d, dst_rank %d\n", rank, dst_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + dst_rank, 1);
            }
        }
    }
    __syncthreads();
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_low_latency_buffer(int* clean_0,
                                                                           int num_clean_int_0,
                                                                           int* clean_1,
                                                                           int num_clean_int_1,
                                                                           int rank,
                                                                           int num_ranks,
                                                                           int* mask_buffer_ptr,
                                                                           int* sync_buffer_ptr) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // Barrier before cleaning (in case of unfinished chunked EP)
    if (sync_buffer_ptr == nullptr)
        nvshmemx_barrier_all_block();
    else
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr);

    // Clean
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure the low-latency mode works fine)
    if (sync_buffer_ptr == nullptr)
        nvshmemx_barrier_all_block();
    else
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr);
}

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer_ptr,
                              int* sync_buffer_ptr,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);

    LAUNCH_KERNEL(&cfg,
                  clean_low_latency_buffer<kNumThreads>,
                  clean_0,
                  num_clean_int_0,
                  clean_1,
                  num_clean_int_1,
                  rank,
                  num_ranks,
                  mask_buffer_ptr,
                  sync_buffer_ptr);
}

template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(void* packed_recv_x,
                                                    void* packed_recv_x_scales,
                                                    int* packed_recv_src_info,
                                                    int64_t* packed_recv_layout_range,
                                                    int* packed_recv_count,
                                                    int* mask_buffer_ptr,
                                                    int* cumulative_local_expert_recv_stats,
                                                    int64_t* dispatch_wait_recv_cost_stats,
                                                    void* rdma_recv_x,
                                                    int* rdma_recv_count,
                                                    void* rdma_x,
                                                    const void* x,
                                                    const topk_idx_t* topk_idx,
                                                    int* atomic_counter_per_expert,
                                                    int* atomic_finish_counter_per_expert,
                                                    int* next_clean,
                                                    int num_next_clean_int,
                                                    int num_tokens,
                                                    int num_max_dispatch_tokens_per_rank,
                                                    int num_topk,
                                                    int num_experts,
                                                    int rank,
                                                    int num_ranks,
                                                    int num_warp_groups,
                                                    int num_warps_per_group,
                                                    bool round_scale,
                                                    int phases) {
    /*
    packed_recv_x:                      要写入的, [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden],    接收端接收到的token组成的tensor。
    packed_recv_x_scales:               要写入的, [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, num_scales],  接收端接收到的token的scale，每个token有num_scales个scale，每个scale表示128/512个float8_e4m3fn。
    packed_recv_src_info:               要写入的, [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank], 表示packed_recv_x中每个token在这个token对应的发送端rank发送的tensor中的src_idx。packed_recv_src_info中会记录来自多个rank的tensor中的token。
    packed_recv_layout_range:           要写入的, [num_local_experts, num_ranks], 接收端本地各个专家接收到的来自各个rank的token数量和起始索引(num_recv_tokens, recv_token_begin_idx)
    packed_recv_count:                  要写入的, [num_local_experts], 接收端各个专家接收到的token数量。
    mask_buffer_ptr:                    要读写的, num_ranks, 表示是否往各个rank发送到接收端rank的数据。
    cumulative_local_expert_recv_stats: 要写入的, num_local_experts, 统计当前rank作为接收端时，本地各个专家累计接收的token数，用于线上服务的EP负载均衡监控。torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
    dispatch_wait_recv_cost_stats:      要写入的, [num_ranks, num_ranks], 用于统计的等待接收各 token 累计耗时，适用于精准检测与定位系统延迟异常。
    rdma_recv_x:                        发写接读, num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg, 参见config.hpp
    rdma_recv_count:                    发写接读, num_experts * sizeof(int), signaling_buffer_bytes_aligned, 参见config.hpp
    rdma_x:                             要写入的, num_max_dispatch_tokens_per_rank 个token数据, std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes), 参见config.hpp
    x:                                  要读写的, [num_tokens, hidden],  当前dispatch发送端要分发的token数据，参见 buffer.py 中的 low_latency_dispatch 函数。写的是num_bytes_per_msg的最开始的int4的第一个int。
    topk_idx:                           要读取的, [num_tokens, num_topk],  当前dispatch发送端要分发的token数据对应的topk_idx。参见 buffer.py 中的 low_latency_dispatch 函数
    atomic_counter_per_expert:          要写入的, num_experts, 目标rank的目标专家缓冲区的slot索引分配器，是属于GPU的全局global_memory中的workspace内存中的一个变量。
    atomic_finish_counter_per_expert:   要写入的, num_experts, 记录传输到对应的全局专家已完成的token数量。
    next_clean:                         要写入的, num_experts, 就是 dispatch_rdma_recv_count_buffer(参见config.hpp)，双缓冲区中另一个buffer的 signaling_buffer_bytes_aligned
    num_next_clean_int:                 要读取的, num_experts, 就是 num_clean_int, 也就是 num_experts, 参见config.hpp
    num_tokens:                         要读取的, 当前 x 的token数量
    num_max_dispatch_tokens_per_rank:   要读取的, 接收端rank的每个专家为每个dispatch发送端rank预分配固定大小的缓冲区空间的大小。一定是大于等于num_tokens的。
    num_topk:                           要读取的, 
    num_experts:                        要读取的, 全局所有的专家数
    rank:                               要读取的, 
    num_ranks:                          要读取的, 
    num_warp_groups:                    要读取的, ceil_div(num_experts, num_device_sms), 一个sm负责处理一个专家组通信，这个专家组有num_warp_groups个专家
    num_warps_per_group:                要读取的, 32 / num_warp_groups, 一个sm的所有warp中，多少个warp处理一个专家的通信
    round_scale:                        要读取的, bool, TODO round_scale 控制 FP8 量化中缩放因子的计算方式，True时将因子舍入为2的幂次方，支持更高效的UE8M0格式存储。
    phases:                             要读取的, 
    */
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);   // 当前任务在GPU上分配到的sm数量（实际是block的数量）
    const auto num_warps = num_warp_groups * num_warps_per_group;  // 一个sm负责处理的专家组通信有多少个warp
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;  // 一个sm的所有warp中，当前warp处理的专家在当前rank要处理的所有专家中的局部编号
    const auto sub_warp_id = warp_id % num_warps_per_group;  // 当前warp处理的这个全局专家的通信的所有warp中，当前warp是第几个warp。
    // sm_id * num_warp_groups是分配到当前sm的全局专家的起始偏移量，而warp_group_id是局部索引。得到当前warp负责的全局专家的全局编号。
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;  // 当前warp负责的全局专家的全局编号。

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    const int num_scales = kHidden / kNumPerChannels;  // 4096 / 128 = 32，表示每个token向量有多少个scale。
    // 不包含scales的hidden数据占据的字节数
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: index at source (int), 3 reserved int fields, hidden data, FP8 scales
    // NOTES: currently we have 3 reserved int fields for future use
    // 当kUseFP8=true时为int2（64位），当kUseFP8=false时为int4（128位）。
    using vec_t = std::conditional_t<kUseFP8, int2, int4>;
    // 与config.hpp中的num_bytes_per_dispatch_msg一致。
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;  // 一个sm最多32个warp
    // 一个sm要发送数据的专家数量不可能超过kNumMaxWarpGroups个。
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)  
        goto LOW_LATENCY_DISPATCH_RECV;  // 跳过发送阶段。实际上phases要么是1要么是3，不会跳过发送阶段，所以这里不会运行

    /* There are 2 kinds of warps in this part:
    1. The first-kind warps for FP8 cast and sending top-k tokens
       - 对于数据类型转换，所有的 (num_warps - 1) 个 warp 都参与。
       - 对于发送 token 到全局专家，只需要 num_topk 个 warp 参与。
       宏观上，每个sm处理一个个的token，每个线程处理这个token的一个int4，每个warp处理这个token数据往一个激活专家的发送，只需要num_topk个warp参与这样的发送。

    2. The last warp for reading `topk_idx` and count for per-expert information
        每个sm的最后一个warp负责清理上一个buffer，和其它的warp合作标记发送token到各个专家的数量。
    */
    if (warp_id < num_warps - 1) {  // 不是最后一个warp，则处理发送阶段。
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);  // 128 / 16 = 8。每次读128比特位，每次读了8个元素。每8个元素就是一个int4
        EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");  // 4096 % (32 * 8) = 0
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");  // 8 * 32 % 128 = 0
        const auto num_threads = (num_warps - 1) * 32;  // 除了最后一个warp之外的所有warp的线程数量。
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;  // 4096 / 8 = 512。表示每个token向量有多少个int4。

        // 同一个sm中的所有warp，每次循环时，都是处理一个token，这个token可能要发送到多个rank的多个专家上，不同的warp用于传输到不同的全局专家。
        // num_sms步进循环，不同sm负责不同的分散开的token
        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            // 当前token的int4地址。
            const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            // token token_idx在发送端缓冲区中的起始地址。
            // 最开始的16个字节的int4是src_idx, 其中的第一个4字节的int表示dispatch要发送的这个token在发送端的输入tensor中的索引
            const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            /* token token_idx的真正有意义的数据。参考 num_bytes_per_msg
            rdma_x_vec: 条件类型，当kUseFP8=true时为int2（64位）,当kUseFP8=false时为int4（128位）。
            */
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            // hidden_bytes是不包含scales的hidden数据占据的字节数。偏移 hidden_bytes 字节，就是scales的地址。
            // kUseFP8 的时候rdma_x_scales才有意义，BF16的时候rdma_x_scales是无效的。
            // rdma_x_scales占据的字节数: num_scales * sizeof(float) 32个float。
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index writes
            // token token_idx激活的第warp_id个全局专家的编号，不同的warp用于传输这个token到不同的全局专家。
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            // 如果当前线程是sm中的第一个线程，则将token_idx写入到rdma_x_src_idx中。在将x作为参数传入dispatch函数时，src_idx是没有被写入的。
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            // FP8 cast
            EP_STATIC_ASSERT(hidden_bf16_int4 % 32 == 0, "Must use the full warp to reduce");
            #pragma unroll
            /* 
            对于这个token的hidden_bf16_int4个int4，每个线程处理一个int4。warp合并访存。
            对于数据类型转换，所有的(num_warps - 1)个warp都参与
            */
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {  // num_threads = (num_warps - 1) * 32
                // Read
                auto int4_value = __ldg(x_int4 + i);  // 每个线程都去读一个int4，即128比特位。

                if constexpr (kUseFP8) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);  // 一个int4对应8个nv_bfloat16。
                    float fp32_values[kNumElemsPerRead];  // 临时数组，存储转换为FP32的BF16数据，为后续量化计算准备32位精度的数据。
                    /*
                    准备量化所需的动态缩放参数
                    amax: 绝对值最大值，初始化为 kFP8Margin（1e-4，避免除零）
                    scale：量化缩放因子（FP32→FP8的乘数）
                    scale_inv：量化缩放因子的倒数（FP8→FP32的乘数）
                    */
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    // 每个线程，转换数据精度并找到8个值中的绝对值最大值
                    for (int j = 0; j < kNumElemsPerRead; ++j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        // fmaxf()：CUDA内置函数，取最大值。fabsf()：CUDA内置函数，取绝对值。
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }

                    // Reduce amax and scale
                    // 8 * 32 / 128 = 256 / 128 = 2，确保向量化参数正确（每个warp处理2个scale值）
                    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                    // warp内归约函数，模板参数16表示每16个线程一组，归约出amax的最大值。
                    // 输入是每个线程的局部amax，输出是warp内的全局最大值
                    amax = warp_reduce_max<16>(amax);
                    /* 计算FP8量化缩放因子的函数
                    scale：输出参数，量化缩放因子
                    scale_inv：输出参数，反量化缩放因子
                    round_scale：布尔参数，是否将scale舍入为2的幂次
                    根据数据范围计算FP8量化的动态缩放参数
                    */
                    calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                    if (lane_id == 0 or lane_id == 16)
                        /* 存储一个token的32个反量化缩放因子scale_inv
                        只有lane_id为0和16的线程写入scale（对应2个scale位置）
                        i = 0, 16, 32, 48, 64, 80, 96, 112, ..., (512-16)
                        i * kNumElemsPerRead = 0, 16*8=128, 32*8=256, 48*8=384, 64*8=512, 80*8=640, 96*8=768, 112*8=896, ..., (512-16)*8=3968
                        i * kNumElemsPerRead / 128 = 0, 1, 2, 3, 4, 5, 6, 7, ..., (512-16)*8/128=31
                        kHidden = hidden_bf16_int4 * kNumElemsPerRead，而 i 就是hidden_bf16_int4的索引。
                        每 128 个元素一个缩放因子，所以i * kNumElemsPerRead / 128就是scale的索引。
                        rdma_x_scales: 指向RDMA缓冲区中scales存储区域的指针。num_scales * sizeof(float)
                        */ 
                        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                    // Cast into send buffer
                    /*
                    准备存储打包的FP8数据
                    int2_value: 条件类型，当kUseFP8=true时为int2（64位）,用于存储量化后的FP8数据。
                    fp8x2_values: 重新解释为FP8x2数组的指针。
                    nv_fp8x2_storage_t：NVIDIA FP8存储类型（内部表示）
                    */
                    vec_t int2_value;  // 条件类型，当kUseFP8=true时为int2（64位）,用于存储量化后的FP8数据
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; j += 2) {  // 每两个FP32转为两个FP8。
                        /* 
                        FP32乘以scale，得到FP8。FP8的范围是[-448, 448]。FP8乘以scale_inv，得到FP32。
                        float2：CUDA内置向量类型，包含两个float
                        */
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                        /*
                        将FP32数据量化为FP8格式
                        nv_cvt_float2_to_fp8x2：NVIDIA内置函数，将float2转换为FP8x2
                        NV_SATFINITE：饱和模式，超出范围的值被钳位
                        NV_E4M3：FP8 E4M3格式（4位指数，3位尾数）
                        */
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                    }
                    rdma_x_vec[i] = int2_value;  // rdma_x_vec也是条件类型。将量化后的数据写入RDMA发送缓冲区
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    // 第一个*表示将指针转换为其指向的值
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            // 1 是barrier_id，num_threads表示当前sm中除了最后一个warp之外的所有线程的同步
            asm volatile("bar.sync 1, %0;" ::"r"(num_threads));

            // Issue IBGDA sends
            /*
            对于发送 token 到全局专家，只需要num_topk个warp参与
            dst_expert_idx: token token_idx 激活的第 warp_id 个专家的全局专家编号。-1表示没有激活任何专家。
            */
            if (dst_expert_idx >= 0) {  // 相当于 warp_id < num_topk
                // 通过原子操作获取下一个可用的插槽位置，保证不同token不会覆盖
                // lane_id为0的线程执行atomicAdd操作，其他线程不执行。
                /*
                atomic_counter_per_expert用于为每个要发送到某个专家的token分配在目标rank对应专家缓冲区中的唯一的slot索引，而不是用于统计或同步。
                通过本地的原子操作，可以保证目标rank对应专家缓冲区中分配的slot索引是唯一的，不会产生冲突。
                */
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                // 都去从lane_id为0的线程中获取slot_idx值。
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;  // 目标专家所在的rank
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;  // 目标专家在目标rank内的局部专家索引
                /*
                uint64_t在这里不是数据宽度，而是地址宽度。转换的目的是将C++指针转换为RDMA操作所需的64位地址整数。64位系统中的所有指针都是64位，uint64_t用于表示64位的内存地址。
                IBGDA函数接口要求: req_lptr 必须是64位整数类型。
                    nvshmemi_ibgda_put_nbi_warp(
                        uint64_t req_rptr,    // 远程地址 (64位)
                        uint64_t req_lptr,    // 本地地址 (64位) ← 需要这个
                        size_t bytes, ...     // 数据大小
                    )
                */
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                
                /*
                接收端缓冲区布局：
                rdma_recv_x: 当前rank的对称堆中的地址，可以据此用来计算得到目标rank的接收缓冲区的地址，
                             计算得到的地址在目标rank的heap_base的偏移量和rdma_recv_x在当前rank的heap_base的偏移量是一样的。
                rdma_recv_x
                ├── 专家0的数据区域 (dst_expert_local_idx = 0)
                │   ├── 来自rank0的token槽位 [0, num_max_dispatch_tokens_per_rank)
                │   ├── 来自rank1的token槽位 [num_max_dispatch_tokens_per_rank, 2*num_max_dispatch_tokens_per_rank)
                │   └── ... 其他rank的槽位
                ├── 专家1的数据区域 (dst_expert_local_idx = 1)  
                │   └── ... 类似的布局
                └── ... 其他专家
                rdma_recv_x: 当前rank的RDMA接收缓冲区起始地址，但可以因为对称堆的缘故理解为是目标rank的接收缓冲区的基地址。
                    整个接收缓冲区的基地址，所有数据都存储在这个缓冲区中。
                    在LowLatencyLayout中定义为dispatch_rdma_recv_data_buffer
                dst_expert_local_idx: 目标专家在目标rank内的局部索引，确定数据属于目标rank的哪个本地专家。
                    dst_expert_local_idx = dst_expert_idx % num_local_experts。取值范围：[0, num_local_experts)
                num_ranks: 整个系统中总的rank数量。
                num_max_dispatch_tokens_per_rank: 接收端的每个专家为每个发送端rank最多能接收的分发token数量上限。
                    接收端rank的每个专家为每个dispatch发送端rank预分配固定大小的缓冲区空间的大小。
                num_bytes_per_msg: 每个token消息占用的字节数。
                rank: 当前发送端rank
                slot_idx: 当前token在目标rank对应专家缓冲区中的位置索引。取值范围: [0, num_max_dispatch_tokens_per_rank)
                num_ranks * num_max_dispatch_tokens_per_rank: 每个专家的缓冲区的token槽位数量上限。
                    每个专家对发送端的各个rank都有固定数量的num_max_dispatch_tokens_per_rank个token槽位。专家数据连续存储，按专家ID顺序排列。
                dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg: 专家级偏移。
                rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg: 在目标专家内，跳转到来自特定发送rank的数据区域的起始位置。
                slot_idx * num_bytes_per_msg: 在目标专家内，跳转到当前token的起始位置。

                目标地址 = 基础地址 + 专家偏移 + rank偏移 + 槽位偏移
                dst_ptr: 目标专家的针对当前rank的接收缓冲区中的slot为slot_idx的token msg的地址。
                */
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                    dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                    rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg + slot_idx * num_bytes_per_msg;
                /*
                检查目标rank是否支持对当前rank的直接P2P内存访问，如果支持返回直接地址，否则返回0。
                dst_p2p_ptr: uint64_t类型，是本地可直接访问的远程内存映射地址。P2P访问的直接内存地址（0表示不支持P2P）
                    映射关系: 远程GPU的地址dst_ptr被映射到本地GPU的地址dst_p2p_ptr。
                dst_ptr: uint64_t类型，原始RDMA目标地址
                rank: int类型，源rank ID
                dst_rank: int类型，目标rank ID

                // TODO 之后深入研究
                */
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {  // 检查目标rank是否被屏蔽（超时）
                    if (dst_p2p_ptr == 0) {  // 0表示不支持NVLink P2P，使用IBGDA进行远程传输
                        // TODO 之后深入研究
                        /*
                        这个调用只是"发起请求"，立即返回。实际传输由 RDMA 硬件完成。
                        */
                        nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                    } else {
                        // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                        // 对于7K隐藏维度，使用8次展开只需要2次加载迭代
                        /*
                        src_int4_ptr是指向一个token的msg，num_bytes_per_msg字节的数据。
                        即使第一个int4的第一个int才是记录src_idx的，但是在config.hpp的num_bytes_per_dispatch_msg中已经对齐为int4了，为什么要align现在知道了。
                        */
                        const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                        const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                        /*
                        ld_nc_global: 只读取当前rank本地数据到当前rank的寄存器中
                        st_na_global: 将当前rank的寄存器中的数据写入到同节点的其他GPU的地址中
                        */
                        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                    }
                }

                // Increase counter after finishing
                // 上面无论是使用IBGDA还是P2P，都使用了当前的一个warp进行的传输，所以需要同步。
                __syncwarp();
                /*
                三个地方会更新atomic_finish_counter_per_expert：
                1、这里是真正发送数据的地方更新，每发送当前rank的一个token数据到对应的专家，就把当前rank上的atomic_finish_counter_per_expert对应的专家加一
                2、在sm_id == 0的sm中的warp_id == num_warps - 1的warp中将下一个buffer的所有位置重置为0之后，
                   就把当前rank上的atomic_finish_counter_per_expert对应的专家加上一个大数FINISHED_SUM_TAG（1024）。
                3、统计实际发送的token数量，减去对应的值，同时也加上FINISHED_SUM_TAG。
                   这样，最终看是否发送成功就只需要看对应值是不是FINISHED_SUM_TAG * 2 了。
                为什么选择1024：
                    足够大：避免与实际token数量冲突
                    2的幂：便于位运算和内存对齐
                    标记作用：明显区别于正常的token计数
                */

                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } else if (warp_id == num_warps - 1) {
        /*
        当前sm的最后一个warp执行，负责清理资源、统计信息并通知后续阶段。
        */
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {  // 第一个SM执行额外的全局清理和通知任务
            // The first sm is also responsible for checking QPs
            /* 确保有足够的QP来处理所有本地专家的通信
            ibgda_get_state(): 获取NVSHMEM IBGDA状态的函数
            num_rc_per_pe：int类型，每个rank的RC QP数量
            RC (Reliable Connection)：
                全称：Reliable Connection
                含义：InfiniBand或RoCE (RDMA over Converged Ethernet) 网络中的可靠连接模式
                特点：提供可靠的数据传输，保证数据包的有序性和完整性
            QP (Queue Pair)：
                全称：Queue Pair
                含义：RDMA网络中的队列对，由发送队列(Send Queue)和接收队列(Receive Queue)组成
                作用：管理RDMA通信的发送和接收操作
            num_local_experts：int类型，当前rank的本地专家数量
            确保每个rank有足够的可靠连接队列对来处理所有本地专家的通信
            */
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);

            // The first sm is also responsible for cleaning the next buffer
            #pragma unroll
            /* 将下一个buffer的所有位置重置为0，为下一次迭代做准备。也就是清理config.hpp中的dispatch_rdma_recv_count_buffer
            num_next_clean_int：int类型，要清理的数组大小（等于num_experts）
            next_clean：int*类型，指向下一个buffer的清理标志数组
            */
            for (int i = lane_id; i < num_next_clean_int; i += 32)  // num_next_clean_int = num_experts
                next_clean[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();  // 前面刚刚出现warp步进循环，因此这里需要使用__syncwarp()同步
            #pragma unroll
            /*
            为全局所有每个专家的计数器加上FINISHED_SUM_TAG。这个值是一个状态同步标记，设计用于多阶段通信的完成确认。
            阶段1：所有发送线程完成后，每个专家计数器加上FINISHED_SUM_TAG。
            阶段2：统计实际发送的token数量，减去对应的值。
            */
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
        }

        // This sm should be responsible for some destination experts, read `topk_idx` for them
        /*
        统计每个专家的token数量。一个sm的所有warp中，num_warps_per_group个warp处理一个专家的通信
        一个sm最多 32 个warp，一个sm处理的专家数不可能超过32个。kNumMaxWarpGroups 就是表示的最多可能的专家数。
        */
        int expert_count[kNumMaxWarpGroups] = {0};  // 注意，这里是每个线程一个长度为32的数组
        const auto expert_begin_idx = sm_id * num_warp_groups; // 当前SM负责的专家起始索引
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);  // 当前SM负责的专家结束索引

        // Per lane count
        #pragma unroll 8
        // 越来越能体会到这种写法了，循环变量的初始值跟某层次内的子层次一一对应，循环步长就是该层次的子层次数量。
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            /* 这里肯定可以优化！！！
            这里的一个warp要访问这么多次的内存，即使warp合并访问，感觉也会有点慢吧？？？num_tokens * num_topk / 32 次warp合并访问
            */
            auto idx = static_cast<int>(__ldg(topk_idx + i));  // 专家的全局索引
            /* 这里限定了更新 expert_count 的范围，所以 expert_count 不用是 shared memory 。
            整个GPU上，不同 sm 只有最后一个warp才执行。一个warp与它要处理的专家的范围 [expert_begin_idx, expert_end_idx) 一一对应
            每个线程都有一个长度为32的数组expert_count，可以用于后面warp_reduce_sum。
            当前warp内的所有线程只会处理expert_count内对应自己的那个位置的数据，而其余位置为初始化的0。
            */
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx]++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
            // 为什么warp内的每个线程都要执行warp_reduce_sum？为什么不只要lane_id == 0的线程执行？
            // 答: 因为warp_reduce_sum是并行执行的，所以每个线程都要执行，否则结果会不正确。
            // sum 表示当前sm处理的局部第(i - expert_begin_idx)个专家要接收当前rank发送的多少个token。
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {  // 一个sm中只有一个线程执行，这个线程就是 warp (num_warps - 1) 中的lane_id == 0的线程
                // shared_num_tokens_sent_per_expert虽然只被一个sm中的唯一一个线程写入，但是它要被后面代码中当前sm内的所有线程访问。因此依然需要设置为shared memory。
                // 而expert_count之所以不需要设置为shared memory，是因为它只是用于在当前warp内使用，后面不需要再被访问。
                // 如果针对专家responsible_expert_idx的warp_group中的所有warp都没有发送token给专家responsible_expert_idx，则 sum 就是0。
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();  // 等待当前sm的所有线程完成上面各自的使命。

    // Issue count sends
    /*
    warp_group_id: 当前warp处理的专家在当前rank要处理的所有专家中的局部编号。
    sub_warp_id: 负责全局专家 responsible_expert_idx 的通信的所有warp中，当前 warp 是第几个warp。
    下面的if表示只需要负责专家 responsible_expert_idx 的通信的所有线程中的一个线程执行即可。
    */
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        // 一个rank有多个专家，dst_rank是专家 responsible_expert_idx 所在的rank
        /*
        responsible_expert_idx 应该理解为是一个"通信对标识符"
        在发送端代码中, responsible_expert_idx 不是单纯专家的概念:
         - 不是"当前rank的局部专家"
         - 不是"发送端rank的专家"
         - 而是"当前发送端rank → 接收端rank的局部专家" 的通信通道
        responsible_expert_idx 的范围是[expert_begin_idx, expert_end_idx - 1]
        */
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        // responsible_expert_idx = sm_id * num_warp_groups + warp_group_id
        // 如果针对专家responsible_expert_idx的warp_group中的所有warp都没有发送token给专家responsible_expert_idx，则num_tokens_sent就是0。
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2)
            ;
        // 此时已经等到了当前rank上的对专家 responsible_expert_idx 的所有token都发送完成了。
        /*
        rdma_recv_count 是rank的rdma buffer中记录往各个全局专家发送的token数量的数组，大小是num_ranks * num_local_experts = num_experts。
        每个rank都会记录本地每个专家接收的来自所有rank的token数量。recv_count^localX_rankY表示本地专家localX接收的来自rankY的token数量。
        按照下面的方式排列的：
        recv_count^local0_rank0, recv_count^local0_rank1, recv_count^local0_rank2, ..., recv_count^local0_rank(num_ranks-1),
        recv_count^local1_rank0, recv_count^local1_rank1, recv_count^local1_rank2, ..., recv_count^local1_rank(num_ranks-1),
        ...
        recv_count^local(num_local_experts-1)_rank0, recv_count^local(num_local_experts-1)_rank1, recv_count^local(num_local_experts-1)_rank2, ..., recv_count^local(num_local_experts-1)_rank(num_ranks-1)
        
        注意: rdma来了，不要把以rdma地址开头的地址运算得到的结果都认为是在这个开头的地址所在的物理内存中!!!
        虽然这里是依照内存在当前rank的buffer rdma_recv_count中的地址计算的，但是这个地址本身也可以用于能p2p的节点内dst_rank和跨节点的需要通过ibgda通信的dst_rank，
        这里只是计算地址，可没说以rdma_recv_count开头就一定是要把数据存储在当前rank的内存rdma_recv_count中。
        如果使用p2p通信，dst_p2p_ptr才有实际意义。如果使用ibgda通信，dst_ptr才有实际意义。

        dst_ptr: 是在dst_rank的内存中的。表示当前rank往rank dst_rank上分配的局部专家dst_expert_local_idx（也就是全局专家 responsible_expert_idx）发送的 token 的数量的记录地址
        dst_ptr是nvshmem对称堆中的地址，可以转换为远程rank的地址，包括可以p2p访问的地址和使用IBGDA访问的地址。
        由发送端写入接收端的接收数据量。写入的值为-num_tokens_sent - 1。
        举个例子：
            假设系统配置: num_ranks = 4, num_local_experts = 2, num_experts = 8
            专家分配: rank 0: 全局专家 0, 1;    rank 1: 全局专家 2, 3;     rank 2: 全局专家 4, 5;    rank 3: 全局专家 6, 7
            rank1 上的rdma_recv_count的索引含义:
            | index |               meaning                 | dst_expert_local_idx | rank |
            |-------|---------------------------------------|----------------------|------|
            |   0   | rank0 向 rank1 的局部专家0 发送的token数 |          0           |   0  |
            |   1   | rank1 向 rank1 的局部专家0 发送的token数 |          0           |   1  |
            |   2   | rank2 向 rank1 的局部专家0 发送的token数 |          0           |   2  |
            |   3   | rank3 向 rank1 的局部专家0 发送的token数 |          0           |   3  |
            |   4   | rank0 向 rank1 的局部专家1 发送的token数 |          1           |   0  |
            |   5   | rank1 向 rank1 的局部专家1 发送的token数 |          1           |   1  |
            |   6   | rank2 向 rank1 的局部专家1 发送的token数 |          1           |   2  |
            |   7   | rank3 向 rank1 的局部专家1 发送的token数 |          1           |   3  |

        注意: 当前rank会作为dispatch发送端会为全局所有专家都写入发送token的数量，即使是0，也要记录为-1。
        */
        auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
        auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {  // 如果rank dst_rank还没有被屏蔽（超时），则进行更新。
            if (dst_p2p_ptr == 0) {  // 0表示不支持NVLink P2P，使用IBGDA进行远程传输
                nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, 
                                                dst_expert_local_idx);
            } else {  // 支持NVLink P2P，使用P2P进行节点内的通信
                // dst_p2p_ptr 可以理解为节点内的所有rank的全局逻辑地址，就是说dst_p2p_ptr可以区分同一个节点内的不同rank。
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
            }
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;  // 为下一次dispatch重置分配器。
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            /* 
            一个sm中满足(responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0)这个条件的线程有且仅有一个。
            一个rank中有多个sm，同一个rank中的sm必定可以对应到整个集群的专家。
            而packed_recv_count的长度是num_local_experts, 而每个rank中的本地专家数都是num_local_experts，
            由于这里只是为了对packed_recv_count的num_local_experts个元素置零，为了避免rank中多个sm在多处都对packed_recv_count的同一个位置的元素都竞争置零，
            所以这里只用对应到某一个dst_rank的所有本地专家就可以。
            实际上这里的写法容易造成歧义，写的好像packed_recv_count是为了统计rank dst_rank要接收的token数量。
            */
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    __syncwarp();

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)  // 如果不想执行接收阶段，直接返回。
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    // 如果当前dispatch函数是send-and-recv kernels，则需要同步grid，确保发送阶段的packed_recv_count对所有线程可见，也就是packed_recv_count都置零了。
    if (phases & LOW_LATENCY_SEND_PHASE)  // 如果执行了发送阶段，则需要同步grid，确保发送阶段的结果对所有线程可见。
        cg::this_grid().sync();

    // Receiving and packing
    /* 如果当前线程要处理的专家是有效的，则进行接收和打包。
    现在当前rank是dispatch接收端，responsible_expert_idx就是dispatch发送端的专家的全局编号，
    src_rank是dispatch发送端的rank编号，local_expert_idx是dispatch发送端的本地专家编号。
    */
    if (responsible_expert_idx < num_experts) {
        /*
        responsible_expert_idx 应该理解为是一个"通信对标识符"，而不是传统的"专家标识符"。
        responsible_expert_idx 不是单纯的概念
            - 不是"当前rank的局部专家"
            - 不是"发送端rank的专家" 
            - 而是编码了"发送端rank → 当前rank的局部专家" 的通信通道

        sm中的每个warp_group处理的"专家"，实际上就是：
            - 不同的通信对(src_rank, local_expert_idx)
            - 每个 warp_group_id 对应一个特定的 responsible_expert_idx 
            - 在同一个SM内: warp_group_id ↔ responsible_expert_idx ↔ (src_rank, local_expert_idx)

        这种双重含义设计非常巧妙：
            1. 代码复用：同一个变量在发送和接收阶段有不同的含义，但计算逻辑一致
            2. 内存效率：不需要额外的变量来存储通信关系
            3. 并行性：自然地将通信任务分配到不同的SM和warp
            4. 负载均衡：可以通过调整SM分配来平衡不同通信对的负载
        */
        const auto src_rank = responsible_expert_idx / num_local_experts;  // 当前线程要接收dispatch发送端rank src_rank的专家 responsible_expert_idx 的token。
        const auto local_expert_idx = responsible_expert_idx % num_local_experts; // dispatch发送端的本地专家编号。
        /* 
        接收端缓冲区rdma_recv_x的布局：
            rdma_recv_x (当前rank的接收缓冲区)
            ├── 专家0的数据区域 (dst_expert_local_idx = 0)
            │   ├── 来自rank0的token槽位 [0, num_max_dispatch_tokens_per_rank)
            │   ├── 来自rank1的token槽位 [num_max_dispatch_tokens_per_rank, 2*num_max_dispatch_tokens_per_rank)
            │   └── ... 其他rank的槽位
            ├── 专家1的数据区域 (dst_expert_local_idx = 1)  
            │   ├── 来自rank0的token槽位 [0, num_max_dispatch_tokens_per_rank)
            │   ├── 来自rank1的token槽位 [num_max_dispatch_tokens_per_rank, 2*num_max_dispatch_tokens_per_rank)
            │   └── ... 其他rank的槽位
            └── ... 其他专家
        rdma_recv_x_uint8 是接收缓冲区中的分配给rank src_rank发往当前rank的局部专家 local_expert_idx 的token槽位的起始地址。
        */
        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        /*
        packed_recv_x: shape: [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]，表示接收端接收到的token组成的tensor。
        recv_x_int4: 表示当前dispatch接收端rank对于本地专家 local_expert_idx 要接收到的token要写入的起始位置。
        */
        const auto recv_x_int4 = static_cast<int4*>(packed_recv_x) + 
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        /*
        packed_recv_src_info: shape:[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank], 
            表示packed_recv_x中每个token在这个token对应的发送端rank发送的tensor中的src_idx。packed_recv_src_info中会记录来自多个rank的tensor中的token的src_idx。
        recv_src_info: shape: [num_ranks * num_max_dispatch_tokens_per_rank], 记录在发送端的tensor中的token的src_idx，
            是为了在combine的时候，知道把每个经过专家模型计算输出的token聚合到之前哪个dispatch发送端。
        */
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        /*
        packed_recv_layout_range: shape: [num_local_experts, num_ranks], 
            表示接收端本地各个专家接收到的来自各个rank的token数量和起始索引(num_recv_tokens, recv_token_begin_idx)
        recv_range 长度为num_ranks。表示本地专家 local_expert_idx 接收到的来自各个rank的token数量和在局部专家local_expert_idx接收的所有token中起始索引起始索引。
        */
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        // 每个token有num_scales个scale，每个scale表示128/512个float8_e4m3fn。
        const auto num_aligned_scales = align_up<int>(num_scales, sizeof(float) / sizeof(scale_t));
        /*
        packed_recv_x_scales: shape: [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, num_scales],
            表示接收端接收到的token的scale，每个token有num_scales个scale，每个scale表示128/512个float8_e4m3fn。
        recv_x_scales shape: [num_ranks * num_max_dispatch_tokens_per_rank, num_scales]
        */
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_aligned_scales;

        // Shared between sub-warps in warp groups
        // 记录当前rank作为接收端处理的num_warp_groups个专家接收
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens = 0, recv_token_begin_idx;
        /*
        sm中处理一个专家的warp的数量不止一个，并且sm中处理专家的warp少于15个。而一个sm中最多只有32个warp
        */
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
        // sub_warp_id: 负责全局专家 responsible_expert_idx 的通信的所有warp中，当前 warp 是第几个warp。
        if (sub_warp_id == 1 and lane_id == 0) {  // sm中处理每个专家的warp中，只有第0个warp的第一个线程才执行下面的代码。
            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {  // 如果dispatch发送端rank src_rank还没有被屏蔽（超时），则进行等待。
                /* 
                num_recv_tokens 表示当前rank的局部专家 local_expert_idx 接收到的来自dispatch发送端rank src_rank 的token数量。
                如果num_recv_tokens是0，则表示当前rank还不知道本地的专家 local_expert_idx 是否会接收到dispatch发送端rank src_rank的token，
                    也就是dispatch发送端rank src_rank还没有处理完给当前rank的局部专家 local_expert_idx 发送token这件事。
                如果发送token数量就是为0，那么num_recv_tokens就会是-1。  所以这两种情况是不一样的。
                如果过了一段时间还是0，则认为dispatch发送端rank src_rank发送超时，这是不正常的。这可能是因为线程调度、网络故障、负载模式等原因导致的。
                */
                while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0                                                               // data not arrived
                       && (wait_recv_cost = clock64() - start_time) <= NUM_TIMEOUT_CYCLES  // not timeout
                );
            }
            // Do not receive tokens if rank timeout or masked
            if (num_recv_tokens == 0)
                num_recv_tokens = -1;
            // Mask rank if timeout
            if (wait_recv_cost > NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for dispatch receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       local_expert_idx,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();  // 当检测到rank src_rank的发送不可用时，但是又无法mask掉这个rank，就让程序崩溃。
                // 自动检测：超时自动触发标记，跳过故障rank，继续与其他正常rank通信。防止单个rank故障导致整个系统挂起
                atomicExch(mask_buffer_ptr + src_rank, 1);  // 标记rank src_rank为超时，后续不再接收来自这个rank的token。
            }

            num_recv_tokens = -num_recv_tokens - 1;
            // packed_recv_count: shape: [num_local_experts], 接收端各个专家接收到的token数量。
            // recv_token_begin_idx 表示src_rank发送给当前rank的局部专家local_expert_idx的token在当前rank的局部专家local_expert_idx接收的来自多个rank的所有token中起始索引,
            // 而当前rank的局部专家local_expert_idx会接收来自多个rank的token。
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);  // 返回加操作之前的值
            // warp_group_id 对应 responsible_expert_idx, 也对应(src_rank, local_expert_idx)
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            // recv_range 长度为num_ranks。表示本地专家 local_expert_idx 接收到的来自各个rank的token的数量和索引。
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);

            // Add stats for diagnosis
            if (cumulative_local_expert_recv_stats != nullptr)
                // 长度是 num_local_experts, 统计当前rank作为接收端时，本地各个专家累计接收的token数，用于线上服务的EP负载均衡监控。
                atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
            if (dispatch_wait_recv_cost_stats != nullptr)
                atomicAdd(reinterpret_cast<unsigned long long*>(dispatch_wait_recv_cost_stats + src_rank), wait_recv_cost);
        }
        // 在发送端已经用了barrier_id 1了。num_warps_per_group是处理responsible_expert_idx的warp数量。
        // 这里是同步阻塞等待负责专家 responsible_expert_idx 的通信的所有warp都完成。
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        // num_recv_tokens 是rank src_rank 发送给当前接收端rank的局部专家 local_expert_idx 的token数量。
        // num_warps_per_group个warp循环步进处理num_recv_tokens个token。
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            // Copy source info
            // rdma_recv_x_uint8 是接收缓冲区中的分配给rank src_rank发往当前rank的局部专家 local_expert_idx 的token槽位的起始地址。
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)  // 一个warp处理一个token，而记录这个token在发送端tensor中的src_idx只需要一个线程写入一次即可。
                /*
                recv_src_info: shape: [num_ranks * num_max_dispatch_tokens_per_rank], 记录在发送端的tensor中该token的src_idx，
                是为了在combine的时候，知道把每个经过专家模型计算输出的token聚合到之前哪个dispatch发送端。

                为什么这里读取src_src_idx的一个int值，就是发送端的tensor中该token的src_idx？
                答: 在dispatch发送端的 rdma_x_src_idx 的数据写入到接收端的rdma_recv_x_uint8中，而rdma_x_src_idx的一开始就是一个int4，
                    这个int4的第一个int就是这个token在发送端的tensor中的src_idx。
                */
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();  // warp中的线程都等lane_id == 0的线程写入recv_src_info后再一起继续执行。

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            // src_data 需要偏移掉记录src_idx的int4，后面的才是token hidden真实的数据
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            /* recv_x_int4: 表示当前dispatch接收端rank对于局部专家 local_expert_idx 要接收到的token要写入的起始位置。
            recv_x_int4 就是接收端接收到的token组成的tensor的地址，也就是将要输入到专家模型计算的token数据。
            */
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if constexpr (kUseFP8) {
                // Equivalent CuTe layout: 等价 CuTe 布局
                //  shape: (num_tokens, (num_packed, num_elems_per_pack))
                //  stride: (num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
                /*
                下面的常规模式布局无法与utils.py中的per_token_cast_back的期望格式兼容。
                内存：[token0_pack0, token0_pack1, ..., token0_pack7, token1_pack0, ...]
                recv_x_scales[token_idx * num_scales + lane_id] = scale;
                
                在per_token_cast_back反量化函数中:
                """
                def per_token_cast_back(x_fp8, x_scales):
                    x_scales = x_scales.view(num_tokens, -1, 1)  # 期望：(24, 32, 1)
                    x_fp32_padded.view(num_tokens, -1, 128)       # 实际：(24, 32, 128)
                    return (x_fp32_padded * x_scales).view(...)   # 广播相乘
                """
                所以这里得按照 per_token_cast_back 的要求来。


                hidden_bytes: token hidden占据的字节数。
                rdma_x_scales 占据的字节数: num_scales * sizeof(float) 32个float。     详细解释819行到844行的代码，逐行解释。
                */
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                // packed_t: uint32_t    scale_t: uint8_t
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i; // 当前token在接收端局部专家 local_expert_idx 接收到的token组成的tensor中的索引。
                const auto token_stride = num_elems_per_pack;
                /*
                pack_stride 表示相邻两个pack在内存中的距离。
                num_ranks * num_max_dispatch_tokens_per_rank：每个局部专家的最大token数量
                * num_elems_per_pack：转换为元素单位的步长
                用途：在packed_recv_x_scales中定位不同pack的位置
                */ 
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
                if (lane_id < num_scales) {
                    /*
                    为什么对recv_x_scales的访问不会越界？
                    答：关键在于pack_idx的计算：
                        lane_id ∈ [0, 31]（一个warp有32个线程）
                        但if (lane_id < num_scales)限制了只处理前32个scale
                        所以pack_idx最大值是 31 / 4 = 7（整数除法），此时elem_idx是 31 % 4 = 3
                    因此，无论如何都会至少留下一个pack_stride的索引范围供token_idx * token_stride + elem_idx去访问。所以不越界。
                    */
                    const auto pack_idx = lane_id / num_elems_per_pack;
                    const auto elem_idx = lane_id % num_elems_per_pack;
                    /*
                    UE8M0格式（kUseUE8M0 = true）
                    UE8M0：8-bit浮点格式，专门为量化scale设计
                    特点：指数部分为0，专门存储2的幂次方（scale通常是2的幂次）
                    优势：高效存储和计算

                    scale压缩存储：float scale → 转换 → uint8_t → 打包 → uint32_t
                    scale在utils.py的per_token_cast_back中反压缩：uint32_t → 解包 → uint8_t → 转换回 float → 用于反量化
                    */
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id));
                    /*
                    recv_x_scales shape: [num_ranks * num_max_dispatch_tokens_per_rank, num_scales], 当前局部专家接收到的所有token的scale数据。
                    token_idx * token_stride: 定位到当前token的起始位置
                    pack_idx * pack_stride: 定位到当前pack的起始位置
                    elem_idx: 在pack内的元素偏移

                    关键理解：
                    num_tokens = num_ranks * num_max_dispatch_tokens_per_rank
                    num_packed = num_scales / num_elems_per_pack
                    num_elems_per_pack = 4（uint32_t包含4个uint8_t）

                    举个例子，假设具体数值：
                    num_ranks = 4; num_max_dispatch_tokens_per_rank = 6; num_scales = 32; num_elems_per_pack = 4
                    那么: num_tokens = 24; num_packed = 8; token_stride = 4; pack_stride = 24 * 4 = 96
                    范围分析:
                    pack_idx * pack_stride 的范围: [0, 7*96] = [0, 672], 其中pack_idx ∈ [0, 7]（32个scale分为8个pack）
                    token_idx * token_stride 的范围: [0, 23*4] = [0, 92], 其中token_idx ∈ [0, 23]（24个token）
                    elem_idx 的范围: elem_idx ∈ [0, 3]（每个pack有4个元素）
                    总索引范围: 最大值: 672 + 92 + 3 = 767;  理论最大: 24 * 32 - 1 = 767
                    */
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
                if (lane_id + 32 < num_scales) {  // num_scales 不会超过64
                    const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
                    const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id + 32));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
            }
        }
    }
}

void dispatch(void* packed_recv_x,
              void* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer_ptr,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              int* next_clean,  // 就是dispatch_rdma_recv_count_buffer，参见config.hpp
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              void* workspace,
              int num_device_sms,  // 当前GPU的sm数量
              cudaStream_t stream,
              int phases) {
    constexpr int kNumMaxTopK = 11;
    // 相邻的专家在同一个rank，使用同一个warp往往会命中同一个专家，造成合并访问的效果。
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);  // 一个sm负责处理的专家组通信有多少个专家
    // 一个sm的所有warp中，平均多少个warp处理一个专家的通信。一个sm最多有32个warp。
    const int num_warps_per_group = 32 / num_warp_groups; // 一个sm(严格说应该是blcok)最多有1024个线程，每个warp32个线程，也就是最多有32个warp。
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = static_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    // FP8 checks
    if (use_ue8m0)
        EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden)                         \
    {                                                        \
        auto dispatch_func = dispatch<false, false, hidden>; \
        if (use_fp8 and not use_ue8m0)                       \
            dispatch_func = dispatch<true, false, hidden>;   \
        if (use_fp8 and use_ue8m0)                           \
            dispatch_func = dispatch<true, true, hidden>;    \
        LAUNCH_KERNEL(&cfg,                                  \
                      dispatch_func,                         \
                      packed_recv_x,                         \
                      packed_recv_x_scales,                  \
                      packed_recv_src_info,                  \
                      packed_recv_layout_range,              \
                      packed_recv_count,                     \
                      mask_buffer_ptr,                       \
                      cumulative_local_expert_recv_stats,    \
                      dispatch_wait_recv_cost_stats,         \
                      rdma_recv_x,                           \
                      rdma_recv_count,                       \
                      rdma_x,                                \
                      x,                                     \
                      topk_idx,                              \
                      atomic_counter_per_expert,             \
                      atomic_finish_counter_per_expert,      \
                      next_clean,                            \
                      num_next_clean_int,                    \
                      num_tokens,                            \
                      num_max_dispatch_tokens_per_rank,      \
                      num_topk,                              \
                      num_experts,                           \
                      rank,                                  \
                      num_ranks,                             \
                      num_warp_groups,                       \
                      num_warps_per_group,                   \
                      round_scale,                           \
                      phases);                               \
    }                                                        \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumSendUnrolls>
__forceinline__ __device__ int logfmt_encode(void* buffer, nv_bfloat162* shared_amaxmin, const int& lane_id) {
    
    /*
    buffer: 大小是 kNumTMABufferBytes + 16（kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls）。
            也就是同一个warp的所有lane调用这个函数时，一起处理的数据量是 sizeof(int4) * 32 * kNumSendUnrolls 字节。
            而每个lane处理的数据量是 sizeof(int4) * kNumSendUnrolls 字节，也就是16个BF16元素(当kNumSendUnrolls=2时)。
            于是每8个连续的lane处理128个BF16元素，也就是一个量化单元。
    shared_amaxmin: 如果不为nullptr，说明当前lane是量化组的leader，需要写入元数据。
    */
    
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);  // kNumElemsPerInt4 = 16 / 2 = 8
    constexpr float kLogThreshold = 0;  // 对数阈值。说明要求取对数之前的原值的绝对值要小于1，也就是经过激活函数后范围在 (-1, 1) 之间的值。
    constexpr float kMinClip = 32;      // `== log_2(2 ^ (2 ^ 5))`，最小剪裁值。
    constexpr int kNumBits = 10;        // 10位量化
    constexpr int kNumValues = 1 << (kNumBits - 1);  // 512个量化值。1 << (10 - 1) = 512

    int4 int4_values[kNumSendUnrolls];  // 存储原始数据，假设kNumSendUnrolls是2
    const auto& uint32_values = reinterpret_cast<uint32_t*>(int4_values);
    // uint32_values设置了值，bf162_values也会跟着设置。
    const auto& bf162_values = reinterpret_cast<nv_bfloat162*>(int4_values);

    // Calculate lane offset
    /*
    buffer是输入缓冲区tma_buffers[stage_idx]，大小是kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls
    ld_buffer: 当前lane的起始偏移位置，大小是 (kNumSendUnrolls * sizeof(int4))字节。
    ld_buffer 和 st_buffer 内存复用。先从ld_buffer读取原始数据，再在st_buffer写入压缩数据，这样通过内存复用可以空间节省。

    注意: 这里是每连续8个lane计算得到128个BF16的量化单元中的最值，并不是一整个warp只计算一个量化单元的最值。
    而连续8个lane能比较出8个float值的最值，是通过warp_reduce_max<kNumLanesToReduce>(amax)中的kNumLanesToReduce=8实现的，
    更精确地说，是warp_reduce_max函数中的warp_reduce执行了三次reduction，即warp_reduce中的:
        value = op(value, __shfl_xor_sync(mask, value, 4));
        value = op(value, __shfl_xor_sync(mask, value, 2));
        value = op(value, __shfl_xor_sync(mask, value, 1));
    */
    const auto& ld_buffer = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4)));
    const auto& st_buffer = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + lane_id * (kNumSendUnrolls * sizeof(int4) * 10 / 16));

    // Local log amax
    // BF16的绝对值的最大值和最小值。
    auto bf162_amax = __nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);  // 初始化为(0, 0)，用于寻找最大值
    auto bf162_amin = __nv_bfloat162(CUDART_INF_BF16, CUDART_INF_BF16);    // 初始化为(+∞, +∞)，用于寻找最小值
    uint32_t local_signs = 0;    // 32位无符号整数，用于记录符号位
    #pragma unroll
    for (int k = 0; k < kNumSendUnrolls * kNumElemsPerInt4 / 2; ++k) {  // k = 0到7，共8次迭代，处理16个BF16值
        // TODO: eliminate bank conflicts
        // MoE模型设计：使用合适的激活函数和归一化层，所以BF16的绝对值都在(0, 1)之间。
        uint32_values[k] = ld_buffer[k];    // 读取32位数据
        /*
        ((uint32_values[k] >> 15) & 1): 取第15位符号位，<< (k * 2) 表示左移k * 2位，这样就可以和local_signs做或操作时把值放到local_signs的第k * 2位。
        ((uint32_values[k] >> 31) & 1): 取第31位符号位，

        local_signs的低16位每一位都对应一个BF16值的符号位。
        */
        local_signs |= ((uint32_values[k] >> 15) & 1) << (k * 2);       // 第15位符号位
        local_signs |= ((uint32_values[k] >> 31) & 1) << (k * 2 + 1);   // 第31位符号位
        uint32_values[k] &= 0x7fff7fff;  // 清除两个BF16的符号位，保留数值部分

        /*
        __nv_bfloat162 __hmax2(__nv_bfloat162 a, __nv_bfloat162 b);
        对两个__nv_bfloat162向量中的对应元素执行并行最大值比较。
        返回结果向量：result.x = max(a.x, b.x), result.y = max(a.y, b.y)
        */
        bf162_amax = __hmax2(bf162_amax, bf162_values[k]);  // uint32_values设置了值，bf162_values就可以读取了。
        bf162_amin = __hmin2(bf162_amin, bf162_values[k]);
    }

    // Reduce per 128 channels
    // TODO: figure out how hardware do 2-byte min/max
    /* 
    static_cast<float>()：转换为标准32位IEEE 754单精度浮点数。转换为float以获得更高精度的数值计算。
    注意: 执行完下面两行代码后，amax是当前线程处理的（kNumSendUnrolls * kNumElemsPerInt4）个BF16元素的最大值。
    这也是kNumLanesToReduce的由来。因为一个warp处理128个BF16元素，而现在已经将这128个元素减少了（kNumSendUnrolls * kNumElemsPerInt4 = 16）倍，
    也就是说现在只需要在128 / 16 = 8个值中取得最大值和最小值了。而这一步用warp_reduce_max<kNumLanesToReduce>(amax)执行归约操作更快。
    */
    auto amax = std::max(static_cast<float>(bf162_amax.x), static_cast<float>(bf162_amax.y));
    auto amin = std::min(static_cast<float>(bf162_amin.x), static_cast<float>(bf162_amin.y));
    /* 
    归约配置:
        128 * sizeof(nv_bfloat16)表示一个量化单元（128个BF16元素作为一个量化单元）的字节数。
        (kNumSendUnrolls * sizeof(int4))表示每个线程要处理的数据的字节数。
        kNumLanesToReduce = 8: 每8个线程处理128个BF16元素的数据块。
    
    前面是每个线程计算本地最大/最小值
    这里通过warp归约得到全局最大/最小值
    
    */
    constexpr static int kNumLanesToReduce = 128 * sizeof(nv_bfloat16) / (kNumSendUnrolls * sizeof(int4));
    /*
    这里执行了三次reduction，将8个数中的最大值得到并返回。也就是得到了128个BF16的量化单元中的最值。
    */
    amax = warp_reduce_max<kNumLanesToReduce>(amax);
    amin = warp_reduce_min<kNumLanesToReduce>(amin);

    // Write min/max into the shared memory
    /* 将amax和amin打包成nv_bfloat162格式存储到共享内存meta_buffers中，通过网络传递给接收端。 */
    if (shared_amaxmin != nullptr)  // 实际上这个if判断等价于: if(lane_id % kNumLanesToReduce == 0)
        *shared_amaxmin = __nv_bfloat162(amax, amin);
    __syncwarp();

    // Calculate log amin/amax float
    const auto& log_amax = log2f_approx(amax);
    // kMinClip = 32，说明最小值的绝对值如果比2^{log_amax-32} 还小，那么就对最小值的绝对值进行截断位这个值。
    const auto& log_amin = fmaxf(log2f_approx(amin), log_amax - kMinClip);
    /* 
    每个线程都得到"是否8线程组内所有线程都满足条件"的结果。只有当所有线程的数据都适合量化时，才启用LogFMT。
    warp_reduce_and的模板参数kIntergroupReduce是true，即组间归约模式。
    假如初始条件（每个8线程量化组内条件完全一致，因为log_amax和log_amin是组内的最值）：
        组0: [T, T, T, T, T, T, T, T]  // lane 0,1,2,3,4,5,6,7 - 量化组0使用LogFMT
        组1: [F, F, F, F, F, F, F, F]  // lane 8,9,10,11,12,13,14,15 - 量化组1不使用LogFMT
        组2: [T, T, T, T, T, T, T, T]  // lane 16,17,18,19,20,21,22,23 - 量化组2使用LogFMT
        组3: [T, T, T, T, T, T, T, T]  // lane 24,25,26,27,28,29,30,31 - 量化组3使用LogFMT
    
    执行完warp_reduce_and后，32个lane的enable_cast都是false。
    也就是说warp_reduce_and是将一个组内的8个kNumLanesToReduce的值扩散到warp内的其他组。

    所以，需要一个warp中处理的多个组都是满足(log_amax < kLogThreshold and log_amin < log_amax)条件，
    才会使得这个warp的所有lane的enable_cast都是true，才需要进行LogFMT-10量化。
    注意: 不是只要一个128个BF16的量化组满足条件，就能对这个量化组进行LogFMT-10量化。
         也不是非得整个token hidden的kHidden个元素都满足条件，才能对这kHidden个元素进行LogFMT-10量化。
         而是需要一个warp中处理的多个组都是满足(log_amax < kLogThreshold and log_amin < log_amax)条件，
         才会使得这个warp的所有lane的enable_cast都是true，才需要对这个warp中的 (32/8) 个量化组进行LogFMT-10量化。
         实际的量化还是以组为单位进行
    */
    const bool& enable_cast = warp_reduce_and<kNumLanesToReduce, true>(log_amax < kLogThreshold and log_amin < log_amax);

    // Case into LogFMT-10 if satisfied
    if (enable_cast) {
        /*
        假设log_amax = -1, log_amin = -4
        step = (-1 - (-4)) / 510 ≈ 3 / 510 ≈ 0.00588
        step_inv ≈ 170
        rounding ≈ 2.0 - log₂((1 + 2^0.00588) × 0.5) × 170
        fused_rounding = rounding - (-4) × 170 ≈ rounding + 680
        */
        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        const auto step_inv = 1.0f / step;
        const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * step_inv;
        const auto fused_rounding = rounding - log_amin * step_inv;

        // Pack every 256 bits into 160 bits
        EP_STATIC_ASSERT(kNumSendUnrolls == 2 or kNumSendUnrolls == 4, "kNumSendUnrolls == 2 or 4 only");
        uint32_t encoded[kNumElemsPerInt4 * 2];  // kNumElemsPerInt4 * 2 = 16 个量化值
        #pragma unroll 1
        for (int i = 0; i < kNumSendUnrolls / 2; ++i) {
            #pragma unroll
            for (int k = 0; k < kNumElemsPerInt4; ++k) {
                // __bfloat1622float2 的第二个2是"to"的意思，就是将__nv_bfloat162转换为float2。
                const auto& [x, y] = __bfloat1622float2(bf162_values[i * kNumElemsPerInt4 + k]);
                // __float2uint_rd 是将float向下取整到uint32_t。
                encoded[k * 2 + 0] = __float2uint_rd(fmaxf(log2f_approx(x) * step_inv + fused_rounding, 0));
                encoded[k * 2 + 1] = __float2uint_rd(fmaxf(log2f_approx(y) * step_inv + fused_rounding, 0));
            }
            st_buffer[i * 5 + 0] = (encoded[0] >> 0) | (encoded[1] << 9) | (encoded[2] << 18) | (encoded[3] << 27);
            st_buffer[i * 5 + 1] = (encoded[3] >> 5) | (encoded[4] << 4) | (encoded[5] << 13) | (encoded[6] << 22) | (encoded[7] << 31);
            st_buffer[i * 5 + 2] = (encoded[7] >> 1) | (encoded[8] << 8) | (encoded[9] << 17) | (encoded[10] << 26);
            st_buffer[i * 5 + 3] = (encoded[10] >> 6) | (encoded[11] << 3) | (encoded[12] << 12) | (encoded[13] << 21) | (encoded[14] << 30);
            st_buffer[i * 5 + 4] = (encoded[14] >> 2) | (encoded[15] << 7) | ((i == 0) ? (local_signs << 16) : (local_signs & 0xffff0000u));
        }
        tma_store_fence();
        __syncwarp();
    }

    // Return TMA copy bytes
    return enable_cast ? (32 * (kNumSendUnrolls * sizeof(int4) * 8 * 10 / 16 / 8)) : (32 * (kNumSendUnrolls * sizeof(int4)));
}

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmt_check_amaxmin(
    uint8_t* meta_buffer, float2* shared_log_amax, float2* shared_log_amin, int* shared_cast_info, const int lane_id) {
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32;  // `== log_2(2 ^ (2 ^ 5))`

    bool enable_cast = true;
    if (lane_id < kNumLanes) {
        // Calculate log amin/amax float
        auto amaxmin2 = reinterpret_cast<uint64_t*>(meta_buffer)[lane_id];
        const auto& bf162_amaxmin = reinterpret_cast<__nv_bfloat162*>(&amaxmin2);
        float log_amax[2], log_amin[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            auto amax = static_cast<float>(bf162_amaxmin[i].x);
            auto amin = static_cast<float>(bf162_amaxmin[i].y);
            log_amax[i] = log2f_approx(amax);
            log_amin[i] = amin == 0 ? log_amax[i] - kMinClip : fmaxf(log2f_approx(amin), log_amax[i] - kMinClip);
            enable_cast = enable_cast and log_amax[i] < kLogThreshold and log_amin[i] < log_amax[i];
        }
        shared_log_amax[lane_id] = make_float2(log_amax[0], log_amax[1]);
        shared_log_amin[lane_id] = make_float2(log_amin[0], log_amin[1]);
    }

    const auto& casted = warp_reduce_and<kNumSendUnrolls>(enable_cast) ? 1u << (lane_id / kNumRecvUnrolls) : 0u;
    const auto& num_casted_prefix = __popc(warp_reduce_or<kNumRecvUnrolls, true>(casted) & ((1u << (lane_id / kNumRecvUnrolls)) - 1));

    if (lane_id < kNumLanes and lane_id % kNumRecvUnrolls == 0)
        shared_cast_info[lane_id / kNumRecvUnrolls] = (num_casted_prefix << 1) | (casted ? 1u : 0u);
    __syncwarp();
}

template <int kNumRecvUnrolls>
__forceinline__ __device__ void decode_and_accumulate(
    uint32_t* ld_buffer, float* accum, const float& log_amax, const float& log_amin, const bool& enable_cast, const float& weight) {
    /*
    accum的长度是[kNumElemsPerInt4 * kNumRecvUnrolls]，初始化为0.0f。
    */
    if (enable_cast) {
        constexpr int kNumBits = 10;
        constexpr int kNumValues = 1 << (kNumBits - 1);

        const auto& step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        auto decode = [=](const uint32_t& encoded, const uint32_t& sign) {
            const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + log_amin);
            return sign ? -decoded : decoded;
        };

        EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
        #pragma unroll
        for (int i = 0; i < kNumRecvUnrolls / 2; ++i) {
            uint32_t concat[6];
            concat[0] = ld_buffer[i * 5];
            #pragma unroll
            for (int k = 1; k < 5; ++k)
                concat[k] = (ld_buffer[i * 5 + k - 1] >> (32 - k * 5)) | (ld_buffer[i * 5 + k] << (k * 5));
            concat[5] = ld_buffer[i * 5 + 4] >> 7;

            const uint32_t& local_signs = ld_buffer[i * 5 + 4] >> 16;
            #pragma unroll
            for (int k = 0; k < 5; ++k) {
                accum[i * 16 + k * 3 + 0] += decode((concat[k] >> 0) & 0x1ff, (local_signs >> (k * 3 + 0)) & 1) * weight;
                accum[i * 16 + k * 3 + 1] += decode((concat[k] >> 9) & 0x1ff, (local_signs >> (k * 3 + 1)) & 1) * weight;
                accum[i * 16 + k * 3 + 2] += decode((concat[k] >> 18) & 0x1ff, (local_signs >> (k * 3 + 2)) & 1) * weight;
            }
            accum[i * 16 + 15] += decode(concat[5] & 0x1ff, (local_signs >> 15) & 1) * weight;
        }
    } else {
        // 此时，循环展开必定是kNumRecvUnrolls = 2的情况。
        #pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {  // 这里的4表示每个int4包含4个__nv_bfloat162元素，也就是8个BF16元素。
            // 这里的k表示是第几个__nv_bfloat162元素。
            auto bf16_pack = *reinterpret_cast<__nv_bfloat162*>(ld_buffer + k);
            // 这里的k * 2表示是第几个int4元素，0和1分别表示kNumRecvUnrolls = 2中的循环展开的轮数。
            accum[k * 2 + 0] += static_cast<float>(bf16_pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16_pack.y) * weight;
        }
    }
}

/*
void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy)
*/
template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls>
__global__ __launch_bounds__(1024, 1) void combine(void* combined_x,
                                                   void* rdma_recv_x,
                                                   int* rdma_recv_flag,
                                                   void* rdma_send_x,
                                                   const void* x,
                                                   const topk_idx_t* topk_idx,
                                                   const float* topk_weights,
                                                   const int* src_info,
                                                   const int64_t* layout_range,
                                                   int* mask_buffer_ptr,
                                                   int64_t* combine_wait_recv_cost_stats,
                                                   int* next_clean,
                                                   int num_next_clean_int,
                                                   int* atomic_clean_flag,
                                                   int num_combined_tokens,
                                                   int hidden,
                                                   int num_topk,
                                                   int num_max_dispatch_tokens_per_rank,
                                                   int num_experts,
                                                   int rank,
                                                   int num_ranks,
                                                   int num_warp_groups,
                                                   int num_warps_per_group,
                                                   int phases,
                                                   bool zero_copy) {
    /*
    combined_x:     要写入的, [num_combined_tokens, hidden],  就是combine输出端输出的tensor，这里是将同一个token经过多个激活专家输出后的结果进行聚合后的结果。
    rdma_recv_x:    发写接读, num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg, 参见config.hpp 的 combine_recv_buffer_bytes, 接收端缓冲区
    rdma_recv_flag: 发写接读, num_experts * sizeof(int),
    rdma_send_x:    要读取的, num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg, 参见config.hpp 的 combine_send_buffer_bytes。发起IBGDA send需要先写入到当前rank的这个对称内存缓冲区中，才能发起IBGDA send。
    x:              要读取的, [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden], 当前rank本地专家的输出。
        与用于高吞吐的训练阶段的MoE通信不同，dispatch发送端的一个token如果激活了当前rank的多个本地专家，这里的x并没有先把这些本地专家对这个token的输出先进行聚合。
    topk_idx:       要读取的, [num_combined_tokens, num_topk], 当前rank作为combine发送端（也就是dispatch接收端），要发送给其他rank的token的topk的idx。
    topk_weights:   要读取的, [num_combined_tokens, num_topk], 当前rank作为combine发送端（也就是dispatch接收端），要发送给其他rank的token的topk权重。
        注意: 在internode_ll.cu中，topk_weights 根本就没有在dispatch阶段进行传输，也就是说专家模型输出后，在进行combine前并没有先将同一token激活多个本地专家的输出进行聚合，而是原封不动地在combine阶段传输，在combine接收端才进行聚合。
    src_info:       要读取的, [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank], 就是dispatch中的packed_recv_src_info, 表示packed_recv_x中每个token在这个token对应的dispatch发送端rank发送的tensor中的src_idx。
    layout_range:   要读取的, [num_local_experts, num_ranks],  就是dispatch中的packed_recv_layout_range, combine发送端本地各个专家在dispatch阶段接收到的来自各个rank的token数量和起始索引(num_recv_tokens, recv_token_begin_idx)
    mask_buffer_ptr:要写入的, num_ranks, 表示是否往各个rank发送到接收端rank的数据。
    combine_wait_recv_cost_stats:要写入的, [num_ranks],  表示全局每个rank在combine发送阶段发送到当前rank的等待时间之和。
    next_clean:         要写入的, num_experts, 就是 dispatch_rdma_recv_count_buffer(参见config.hpp)，双缓冲区中另一个buffer的 signaling_buffer_bytes_aligned
    num_next_clean_int: 要读取的, num_experts, 就是 num_clean_int, 也就是 num_experts, 参见config.hpp
    atomic_clean_flag:  要写入的, 1, workspace中的一个int变量，用于原子操作。TODO
    num_combined_tokens:要读取的, 1, combine接收端接收到的token数量。其实就是当前rank作为dispatch发送端时发送出去的token数量。
    hidden:             要读取的,
    num_topk:           要读取的,
    num_max_dispatch_tokens_per_rank:要读取的,
    num_experts:        要读取的, 
    rank:               要读取的,
    num_ranks:          要读取的,
    num_warp_groups:    要读取的,
    num_warps_per_group:要读取的,
    phases:             要读取的,
    zero_copy:          要读取的,
    */
    const auto sm_id = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);  // 只有lane 0实际访问blockIdx.x，减少对寄存器的访问。
    const auto num_sms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warp_id = __shfl_sync(0xffffffff, thread_id / 32, 0), lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;  // 当前warp处理的专家在当前rank要处理的所有专家中的局部编号。
    const auto sub_warp_id = warp_id % num_warps_per_group;    // 负责全局专家 responsible_expert_idx 的通信的所有warp中，当前 warp 是第几个warp。
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;  // responsible_expert_idx 应该理解为是一个"通信对标识符"

    /*
    在CUDA kernel中，extern __shared__声明了一个外部shared memory数组，其含义是：
    extern：表示这个变量的存储空间由外部提供（CUDA runtime负责分配和管理）。
    不占用编译时确定的内存空间，大小在kernel启动时动态指定，就是下面的sharedMemSize。
        """
        // 声明：告诉编译器存在这样一个数组，但不分配空间
        extern __shared__ uint8_t smem_buffer[];

        // 使用：CUDA runtime在启动kernel时分配实际空间
        kernel<<<grid, block, sharedMemSize>>>(...);
        """

    __align__(1024)表示内存对齐要求:
        smem_buffer的起始地址必须对齐到1024字节边界, 也就是 1KB对齐。
        这是为了TMA (Tensor Memory Accelerator) 和其他硬件单元的优化要求
    */
    extern __shared__ __align__(1024) uint8_t smem_buffer[];  // 数组大小在kernel启动时动态指定

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);  // 128 / 16 = 8。每次读128比特位，每次读了8个元素。每8个元素就是一个int4
    constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;  // 4096 / 8 = 512。表示每个token向量有多少个int4。

    // Use different unroll factors for send and recv phases
    /*
    32 * sizeof(int4) / sizeof(nv_bfloat16) 表示每个warp的32个线程对应的数据每次用int4格式能封装多少个nv_bfloat16。
    乘以4，表示的是发送端的循环展开因子。如果kHidden能被乘以4后整除，则kNumSendUnrolls为4，否则为2。
    循环展开因子（Unroll Factor）：单次展开后，原循环体被复制的次数（即 “展开倍数”）；
    主循环（Main Loop）：展开后处理 “能被展开因子整除的迭代次数” 的核心循环；
    剩余部分（Remainder/Residual Loop）：展开后处理 “无法被展开因子整除的剩余迭代次数” 的小循环（或直出代码）。
    */
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    // 对齐TMA
    constexpr int hidden_bf16_int4_pad = align_up(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;  // 每个token的量化组的数量，就是 num_scales
    // 每个token的量化组的元数据大小，每一组需要记录 min/max 两个float。
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    /* 
    num_bytes_per_slot 就是 config.hpp 中的 num_bytes_per_combine_msg，计算方式都是一样的。
    size_t num_bytes_per_combine_msg = num_scales * sizeof(nv_bfloat162) + hidden * sizeof(nv_bfloat16);
    */
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");  

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)  // 如果不需要执行发送阶段，直接跳转到接收阶段
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {  // 一个grid中有且仅有一个warp来清理下一个buffer
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)  // 一个grid中有且仅有一个线程执行
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        /*
        responsible_expert_idx 应该理解为是一个"通信对标识符"，而不是传统的"专家标识符"。
            - 不是"当前rank的局部专家"
            - 不是"发送端rank的专家" 
            - 而是编码了"当前发送端rank → 接收端rank的局部专家" 的通信通道
        */
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;  // 当前rank的本地专家 local_expert_idx 在全局专家中的编号。
        // layout_range: [num_local_experts, num_ranks]
        // layout 表示当前rank的局部专家local_expert_idx在dispatch阶段接收到的来自rank dst_rank的token数量和起始索引(num_recv_tokens, recv_token_begin_idx)
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        /*
        x: [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden], 当前rank本地专家的输出。
        local_x 表示当前rank的局部专家local_expert_idx的所有输出的token的起始地址，大小是 num_max_dispatch_tokens_per_rank * num_ranks * hidden_bf16_int4，
                但是可能并没有写满num_max_dispatch_tokens_per_rank * num_ranks多个token。
        */
        const auto local_x = static_cast<const int4*>(x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        /*
        src_info: [num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank], 就是dispatch中的packed_recv_src_info, 
                    表示packed_recv_x中每个token在这个token对应的dispatch发送端rank发送的tensor中的src_idx。
        local_src_info: 表示当前rank的局部专家local_expert_idx的所有输出的token在这个token对应的dispatch发送端rank发送的tensor中的src_idx的**地址**，
                        大小是 num_ranks * num_max_dispatch_tokens_per_rank。很可能没有写满。
        */
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        /*
        rdma_send_x: num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg, 参见config.hpp 的 combine_send_buffer_bytes。
                     发起IBGDA send需要先写入到当前rank的nvshmem对称内存缓冲区中，才能发起IBGDA send。
                     把 num_experts 拆解为 num_local_experts * num_ranks 比较好理解。
        num_bytes_per_slot: 就是 config.hpp 中的 num_bytes_per_combine_msg。
        rdma_send_x_vec 表示当前rank的局部专家local_expert_idx的所有要输出的token的起始地址，大小是 num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot。
        */
        const auto rdma_send_x_vec = static_cast<uint8_t*>(rdma_send_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        /* 相当于dispatch接收端的 recv_token_begin_idx 和 num_recv_tokens
        offset:             表示当前rank的局部专家local_expert_idx在dispatch阶段接收到的多个rank的所有token中的来自rank dst_rank的token的起始索引。
        num_tokens_to_send: 表示当前rank的局部专家local_expert_idx在dispatch阶段接收到的多个rank的所有token中的来自rank dst_rank的token数量。
        */

        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        // TMA stuffs
        // 主循环部分，单次每个warp的32个线程对应的数据每次用int4格式需要内存空间的大小。
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        /*
        smem_ptr 表示当前warp的共享内存起始地址。
        (kNumTMABufferBytes + 16): 每个warp的每个stage的共享内存大小，每个stage需要16字节的额外空间,
            注意: 这个16字节是干什么用的？
            答: 🤣，用于给每个warp的每个stage都设置一个uint64_t的mbarrier。实际上mbarrier只需要8字节，但是这里预留了16字节。
                这个冗余这是为了和16字节的int4对齐。TMA的cp.async.bulk指令规定: dstMem和srcMem地址和传输的数据大小都必须按16字节对齐。

        kNumStages * (kNumTMABufferBytes + 16): 一个token数据的hidden部分用kNumStages个stage并行传输，但是元数据不需要用stage传输。
                                                这表示的是每个warp需要有针对hidden的kNumStages个stage的共享内存大小。
        kNumMetaBytes: 每个token的量化组的元数据大小，每一组需要记录 min/max 两个float。大小是 kNumDivisions * sizeof(nv_bfloat162)。
        warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes) 表示当前warp的共享内存起始地址。
        */
        auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        uint32_t tma_phase = 0;
        /*
        Lambda 表达式：
        完整的Lambda 表达式：
        auto func = [=](const int& i) { 
            return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); 
        };
        [ = ]：捕获子句（Capture Clause）。= 表示按值捕获所有外部变量。smem_ptr、kNumTMABufferBytes 等外部变量被拷贝到lambda中。
        ( const int& i )：参数列表。const int& i：参数类型和名字。

        其他的捕获方式：
        int x = 10, y = 20;
        auto lambda1 = [=]() { return x + y; };  // [=] 按值捕获所有
        auto lambda2 = [&]() { x = 30; };        // [&] 按引用捕获所有。可以修改外部x
        auto lambda3 = [x, &y]() { y = x; };     // [x, &y] x按值，y按引用
        
        class MyClass {
            int member = 42;
            auto func() {
                return [this]() { return member; };  // [this] 捕获当前对象的指针
            }
        };

        i 指的是stage_idx，表示第几个stage。
        tma_buffers[stage_idx] 表示当前warp的第stage_idx个stage的共享内存起始地址。大小是kNumTMABufferBytes + 16。
        */
        auto tma_buffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smem_ptr + i * (kNumTMABufferBytes + 16)); });
        /*
        full_barriers[stage_idx] 表示的是当前warp的第stage_idx个stage的64位的mbarrier。大小是16字节，紧跟在tma_buffers[stage_idx] 之后。
        */
        auto full_barriers = PatternVisitor(
            [=](const int& i) { return reinterpret_cast<uint64_t*>(smem_ptr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        /*
        只有使用了LogFMT量化，才会有量化组的元数据，否则就把meta_buffers设置为nullptr。
        存储位置在所有的tma_buffers之后。大小是kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
        */
        auto meta_buffers = kUseLogFMT ? reinterpret_cast<nv_bfloat162*>(smem_ptr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (lane_id < kNumStages) {  // 和我预料的一样，就应该一个land对应一个stage
            /*
            mbarrier_init是创建一个CTA级的mbarrier，PTX指令中已经说明了，而不是cluster级的mbarrier。
            mbarrier_init(tma_mbarrier, 1)中的arrive_count = 1 实际上设置的是Expected_Arrive_Count，
            表示mbarrier期望有且仅有1个TMA写共享内存操作，mbarrier_wait函数到时候只用等待一个TMA操作完成，屏障就会打开。
            读共享内存不需要mbarrier。
            而每个TMA写共享内存操作的线程到底需要各自传输多少字节的数据，
            则是由每个线程调用mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_bytes)来告知mbarrier的。
            */
            mbarrier_init(full_barriers[lane_id], 1);
            /*
            fence.mbarrier_init.release.cluster: release前不到后。 cluster作用域，确保cluster内所有SM都能看到一致的初始化状态。
            因为使用mbarrier的tma_load_1d是将全局内存的数据写入shared::cluster，而不是shared::cta。
            发布内存屏障，确保之前的mbarrier.init操作对所有线程可见（特别是cluster内的其他SM）。 
            */
            fence_barrier_init();
        }
        // warp中的lane_id >= kNumStages 的线程虽然不会发起TMA任务，但是会为TMA任务提供支持: 把自己负责的数据写入到共享内存。所以也需要同步。
        __syncwarp();

        /* 
        // 主循环部分，单次每个warp的32个线程对应的数据每次用int4格式需要内存空间的大小。
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;

        kNumIters: 表示一个hidden_bf16_int4_pad要被一个warp用kNumTMABufferBytes大小的共享内存缓冲区传输多少次。
        */
        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);

        /*
        [&] 按引用捕获所有。可以修改外部的值: tma_buffers, gmem_ptr, full_barriers
        tma_load_and_arrive(stage_idx, gmem_ptr, num_bytes): 
        将全局内存的gmem_ptr地址处的连续num_bytes字节的数据写入共享内存tma_buffers[stage_idx]的地址处，
        并告知full_barriers[stage_idx]: arrive_count 增加 1，而初始化的时候Expected_Arrive_Count就是1，因此已经满足了arrive条件。 
        而Transaction_Count设置为num_bytes，表示期望写入num_bytes字节的数据。而修改Transaction_Count的值由tma_load_1d完成后修改Transaction_Count的相反数增加num_bytes。
        */
        auto tma_load_and_arrive = [&](const int& stage_idx, const int4* gmem_ptr, const int& num_bytes) {
            tma_load_1d(tma_buffers[stage_idx], gmem_ptr, full_barriers[stage_idx], num_bytes);
            mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_bytes);
        };
        auto get_num_tma_bytes = [&](const int& offset_int4) {
            // kNumTMABufferBytes 是每个warp的每个stage的共享内存大小，是常规大小。
            // 而(hidden_bf16_int4 - offset_int4) * sizeof(int4) 可能是最后小循环或者直出代码要处理的数据部分。
            return min(kNumTMABufferBytes, static_cast<int>((hidden_bf16_int4 - offset_int4) * sizeof(int4)));
        };

        // Issue IBGDA send
        if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {  // 如果rank dst_rank还没有被屏蔽（超时），则对dst_rank发送token数据。
            /*
            offset:             表示当前rank的局部专家local_expert_idx在dispatch阶段接收到的多个rank的所有token中的来自rank dst_rank的token的起始索引。
            num_tokens_to_send: 表示当前rank的局部专家local_expert_idx在dispatch阶段接收到的多个rank的所有token中的来自rank dst_rank的token数量。
            sub_warp_id:        当前warp处理的这个全局专家的通信的所有warp中，当前warp是第几个warp。
            token_idx:          当前warp要发送给的rank dst_rank的token的索引。
            num_warps_per_group:处理rank dst_rank上的全局专家responsible_expert_idx的通信的所有warp的数量。
                一个warp传输一个token。现在是当前要往rank dst_rank上的全局专家responsible_expert_idx发送token数据。
                因此，token_idx每次循环后增加的大小也就是num_warps_per_group。
            */
            for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
                /*
                local_x 表示当前rank的局部专家local_expert_idx的所有输出的token的起始地址，大小是 num_max_dispatch_tokens_per_rank * num_ranks * hidden_bf16_int4，
                        但是可能并没有写满num_max_dispatch_tokens_per_rank * num_ranks多个token。
                x_int4: 表示当前warp要发送给rank dst_rank上的全局专家responsible_expert_idx的token token_idx在local_x中的起始地址。
                rdma_send_x_vec: 表示当前rank的局部专家local_expert_idx的所有要输出的token的起始地址，大小是 num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot。
                
                */
                const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
                // rdma缓冲区中将要被写入x_int4的地址
                const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
                const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

                // Copy directly to local rank, or copy to buffer and issue RDMA
                /*
                local_src_info: 表示当前rank的局部专家local_expert_idx的所有输出的token在这个token对应的dispatch发送端rank发送的tensor中的src_idx的**地址**，
                                大小是 num_ranks * num_max_dispatch_tokens_per_rank。很可能没有写满。
                src_idx: 表示当前token在dispatch发送端rank发送的tensor中的src_idx。
                */
                const auto src_idx = __shfl_sync(0xffffffff, __ldg(local_src_info + token_idx), 0);
                /*
                x_int4将要写入当前rank的rdma缓冲区的地址。
                */
                const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
                /*
                rdma_recv_x: 发写接读, num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg, 
                             参见config.hpp 的 combine_recv_buffer_bytes, combine接收端要接收的token数据的缓冲区大小。
                             combine接收端也就是dispatch发送端，下面的src_idx就体现了这一点。
                注意: combine接收端为全局所有的每个专家都预留了 num_max_dispatch_tokens_per_rank 个接收token的缓冲区空间。
                dst_ptr: 表示x_int4在combine接收端rank的接收缓冲区中token的缓冲区中的地址。
                */
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                     (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
                /*
                如果是P2P，dst_p2p_ptr是ptr对应的在rank dst_rank的heap_base中的地址。
                */
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                int num_send_bytes = hidden * sizeof(nv_bfloat16);

                // 如果不使用zero_copy，或者能使用NVLink P2P，则使用P2P通信，否则使用IBGDA通信。
                if (not zero_copy or dst_p2p_ptr != 0) {
                    // Read from `cpy_src_int4_ptr` and copy into `cpy_dst_int4_ptr`
                    /*
                    zero copy、zero_copy的最核心原理:
                    torch::from_blob 可以将外部内存块（如 C 数组、NumPy 数组、缓冲区等）“包装” 为 PyTorch 张量（Tensor），
                    而无需拷贝数据（默认情况），实现数据在 PyTorch 与其他系统之间的高效共享。

                    非 Zero-Copy 模式: x_int4 (expert输出) -> [TMA拷贝] -> buf_ptr (发送缓冲区) -> [IBGDA] -> dst_ptr (接收端)
                    Zero-Copy 模式: buf_ptr (expert直接输出到这里) -> [IBGDA] -> dst_ptr (接收端)

                    在推理引擎（如SGLang）中，当需要极低延迟时，可以预先获取RDMA缓冲区tensor（通过 torch::from_blob 实现），
                    然后直接将模型输出写入缓冲区，最后调用零拷贝模式的combine，立即开始combine通信。
                    */
                    const auto cpy_src_int4_ptr = zero_copy ? reinterpret_cast<int4*>(buf_ptr) : x_int4;
                    
                    const auto cpy_dst_int4_ptr = dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);

                    // Prefetch
                    if (elect_one_sync())
                        // 传输当前warp的第0个stage的共享内存数据，传输的字节数是get_num_tma_bytes(0)。
                        // 第0个stage传输的字节数一般都是kNumTMABufferBytes，否则就不用循环展开了。
                        tma_load_and_arrive(0, cpy_src_int4_ptr, get_num_tma_bytes(0));
                    /* 
                    这里只是发起了TMA任务，并没有等待TMA任务完成。就是没有调用mbarrier_wait<true>(full_barriers[0], tma_phase, 0);
                    为什么呢？因为Prefetch是为了给多stage并行先启动第一个stage的TMA任务。
                    */
                    __syncwarp();
                    // stagei_landj_urk
                    // kNumMetaBytes: 每个token的量化组的元数据大小，每一组需要记录 min/max 两个float。大小是 kNumDivisions * sizeof(nv_bfloat162)。
                    int tma_offset_bytes = kNumMetaBytes;
                    /*
                    #pragma unroll 是代码的循环展开。而使用kNumSendUnrolls是进行手动循环展开，每次循环传输kNumSendUnrolls个int4数据。
                    // kNumTMABufferBytes: 主循环部分，单次每个warp的32个线程对应的数据每次用int4格式需要内存空间的大小。
                    constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;  所以这是32个线程也就是一个warp要传输的大小
                    在kNumSendUnrolls的作用下，每个 land 要传输连续的 kNumSendUnrolls 个 int4 数据。而如果不用kNumSendUnrolls，则每个 land 只需要传输 1 个 int4 数据。

                    hidden_bf16_int4_pad: 把 hidden_bf16_int4 对齐到 (32 * kNumSendUnrolls) 的整数倍的大小。这是为了配合对齐TMA传输长度。
                    
                    假设 kNumSendUnrolls = 2
                    land_0:  则 i = 0, 64, 128, 192, ...
                    land_1:  则 i = 1, 65, 129, 193, ...
                    land_2:  则 i = 2, 66, 130, 194, ...
                    land_3:  则 i = 3, 67, 131, 195, ...
                    ...
                    land_31: 则 i = 31, 95, 159, 223, ...
                    
                    iter_idx = 0: 则 stage_idx = 0, next_stage_idx = 1
                    iter_idx = 1: 则 stage_idx = 1, next_stage_idx = 2
                    iter_idx = 2: 则 stage_idx = 2, next_stage_idx = 0
                    */
                    #pragma unroll
                    for (int i = lane_id * kNumSendUnrolls, iter_idx = 0; i < hidden_bf16_int4_pad; i += 32 * kNumSendUnrolls, ++iter_idx) {
                        // Load the next iteration
                        /*
                        iter_idx, stage_idx 和 next_stage_idx, 都是作为共享内存中实际为每个stage分配的内存地址的索引，需要循环使用。
                        */
                        const int& stage_idx = iter_idx % kNumStages;  // 循环取值，0,1,2,0,1,2,...
                        const int& next_stage_idx = (iter_idx + 1) % kNumStages;  // 下一个stage循环取值，1,2,0,1,2,0,...
                        /* 
                        // 每个warp的32个线程对应的数据每次stage用int4格式需要内存空间的大小。
                        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;

                        kNumIters: 表示一个hidden_bf16_int4_pad要被一个warp用kNumTMABufferBytes大小的共享内存缓冲区传输多少次。
                        constexpr int kNumIters = hidden_bf16_int4_pad / (32 * kNumSendUnrolls);

                        共享内存布局：auto smem_ptr = smem_buffer + warp_id * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);

                        要求 iter_idx + 1 < kNumIters，说明最后一次循环不执行下面if里面的代码，也就是不用读取共享内存中的数据了，
                        因为倒数第二次循环时，就已经把最后一次循环需要写入共享内存的tma_load_1d操作发起过了。
                        最后一次循环只需要先mbarrier_wait等待上一轮循环发起的写入共享内存的操作，然后把共享内存的数据调用tma_store_1d写入到全局内存中即可。

                        elect_one_sync()会使得每次选上的都是lane 0。
                        */
                        if (iter_idx + 1 < kNumIters and elect_one_sync()) {
                            tma_store_wait<kNumStages - kNumPrefetch - 1>();
                            /*
                            一个warp的一个stage是要发送 (32 * kNumSendUnrolls) 个 int4 数据。
                            而 tma_store_wait 已经等到了上一个stage的TMA任务读取共享内存完成，因此可以发起下一个stage的TMA任务，也就是将全局内存的数据复制到共享内存。

                            offset_int4 = i + 32 * kNumSendUnrolls 
                                        = lane_id * kNumSendUnrolls + iter_idx * 32 * kNumSendUnrolls + 32 * kNumSendUnrolls
                                        = lane_id * kNumSendUnrolls + (iter_idx + 1) * 32 * kNumSendUnrolls
                            lane_id * kNumSendUnrolls 这个部分不会导致偏差吗？
                            注意: elect_one_sync()会使得每次选上的都是lane 0，因此就等价于:
                            const auto& offset_int4 = (iter_idx + 1) * 32 * kNumSendUnrolls
                            这里应该就是: const auto& offset_int4 = (iter_idx + 1) * 32 * kNumSendUnrolls; 才对吧？
                            */
                            // const auto& offset_int4 = i + 32 * kNumSendUnrolls;
                            const auto& offset_int4 = (iter_idx + 1) * 32 * kNumSendUnrolls;
                            /* 
                            get_num_tma_bytes(offset_int4) 就是 kNumTMABufferBytes
                            发起下一个stage的全局内存数据写入共享内存中这一TMA任务。
                            cpy_src_int4_ptr 就先简单理解为 x_int4，就是要发送的token_idx的地址。
                            */
                            // /*
                            // tma_load_1d(tma_buffers[next_stage_idx], cpy_src_int4_ptr + offset_int4, full_barriers[next_stage_idx], get_num_tma_bytes(offset_int4));
                            // mbarrier_arrive_and_expect_tx(full_barriers[next_stage_idx], get_num_tma_bytes(offset_int4));
                            // */
                            tma_load_and_arrive(next_stage_idx, cpy_src_int4_ptr + offset_int4, get_num_tma_bytes(offset_int4));
                        }
                        __syncwarp();

                        // Wait the current TMA arrival
                        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
                        mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);   // 等待往当前stage的共享内存写入数据的TMA任务完成
                        if constexpr (kUseLogFMT) {
                            // Cast if possible
                            // kNumInt4PerDivision: 32.  每个量化组包含多少个int4数据。128个元素 / 每个int4包含8个nv_bfloat16元素
                            constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                            /*
                            在低延迟模式下，通信时间减少带来的收益 > 量化计算开销。
                            而在训练模式下，重点需要高吞吐，计算是主要瓶颈，而不是通信。

                            整体流程:
                            发送端 (TMA warp): 原始BF16数据 → logfmt_encode → 压缩数据 + 元数据 → 网络传输
                            接收端 (TMA warp): 压缩数据 + 元数据 → logfmt_check_amaxmin → 存储量化参数
                            接收端 (Reduction warp): 量化参数 → decode_and_accumulate → 解码 + 累加 → 最终结果
                            */
                            int num_tma_bytes = logfmt_encode<kNumSendUnrolls>(
                                // tma_buffers[stage_idx] 表示当前warp的第stage_idx个stage的共享内存起始地址。
                                // 大小是 kNumTMABufferBytes + 16（kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls）。
                                tma_buffers[stage_idx],
                                // NOTES: only the leader lane will write the result
                                /* 
                                i 表示的是 int4 的编号，只有for循环中遇到每个量化组的第一个int4时需要写入元数据（也就是这个128个BF16组成的量化组的两个最值）。
                                meta_buffers存储位置在所有的tma_buffers之后。大小是 kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
                                既然i % kNumInt4PerDivision == 0，那么 (i / kNumInt4PerDivision) 表示的是量化组的编号，也就是meta_buffers的索引。
                                */
                                (i % kNumInt4PerDivision == 0) ? meta_buffers + i / kNumInt4PerDivision : nullptr, lane_id);
                            if (elect_one_sync())
                                /* 
                                缓冲区布局: [metadata (kNumMetaBytes)] [token数据...]
                                - metadata写入到 cpy_dst_int4_ptr（缓冲区开头）
                                - token数据写入到 cpy_dst_int4_ptr + tma_offset_bytes，而tma_offset_bytes初始化为kNumMetaBytes
                                这样设计是因为接收端需要先读取metadata来解码token数据（见第1811行：buffer + kNumMetaBytes开始读取token数据）
                                */
                                tma_store_1d(tma_buffers[stage_idx], reinterpret_cast<uint8_t*>(cpy_dst_int4_ptr) + tma_offset_bytes, num_tma_bytes);
                            tma_offset_bytes += num_tma_bytes;
                        } else {
                            // BF16 original values
                            if (elect_one_sync())
                                /* 
                                NVLink P2P 节点内的跨GPU通信的情况.
                                cpy_dst_int4_ptr 是接收端接收token token_idx的缓冲区地址，i 表示第几个 int4。
                                
                                */
                                tma_store_1d(tma_buffers[stage_idx], cpy_dst_int4_ptr + i, get_num_tma_bytes(i));
                        }
                        __syncwarp();
                    }

                    // Store metadata (min/max values) for LogFMT
                    if constexpr (kUseLogFMT) {
                        num_send_bytes = tma_offset_bytes;
                        if (elect_one_sync())
                            // kNumMetaBytes: 每个token的量化组的元数据大小，每一组需要记录 min/max 两个float。
                            // 大小是 kNumDivisions * sizeof(nv_bfloat162)。
                            tma_store_1d(meta_buffers, cpy_dst_int4_ptr, kNumMetaBytes);
                    }

                    // Flush all stores
                    tma_store_wait<0>();  // 等待把上面for循环中的所有tma_store_1d都执行完
                    __syncwarp();
                }

                // Issue RDMA
                // NOTES: for zero-copy mode, we assume the data is already in the send buffer (buf_ptr)
                // 在zero-copy模式下，数据已经预先写入了发送缓冲区buf_ptr，因此可以直接从buf_ptr发送，无需额外的拷贝步骤。
                // 在非zero-copy模式下，上面的TMA操作已经将数据从x_int4拷贝到了buf_ptr，所以这里也是从buf_ptr发送。
                if (dst_p2p_ptr == 0)
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, token_idx - offset);
            }
        }

        // Put the finishing flag
        /*
        num_warps_per_group > 1：每个专家组至少需要 2 个 warp（一个用于发送，一个用于放置标志）
        16是因为一个SM中最多只能有16个硬件barrier（Barrier Units）
        */
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1), "r"(num_warps_per_group * 32)); // 同步负责当前rank上的同一个专家的所有warp，确保数据发送完成。
        if (sub_warp_id == 1 and lane_id == 0) {  // 每个专家组只用一个线程放置完成标志，避免竞争
            /*
            防御性编程，确保 atomic_add_release_global(atomic_clean_flag, num_experts); 执行的结果在当前可见
            */
            while (ld_acquire_global(atomic_clean_flag) == 0)
                ;
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            // 告诉远程rank dst_rank，当前rank已经完成发送，可以开始接收。
            if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
                if (dst_p2p_ptr == 0) {
                    // IBGDA路径：使用原子操作写入值 1 到远程内存。
                    nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
                } else {
                    st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
                }
            }
            atomic_add_release_global(atomic_clean_flag, -1);  // 每个专家组排出一个线程
        }
        __syncwarp();

        // Destroy m-barriers
        if (lane_id < kNumStages) {
            mbarrier_inval(full_barriers[lane_id]);  // 清理 TMA 使用的内存屏障。每个lane处理一个stage的mbarrier清理。
            fence_barrier_init();  // 确保mbarrier初始化操作在cluster全局可见。
        }
        __syncwarp();
    }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            const auto src_rank = responsible_expert_idx / num_local_experts;
            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
                // 等待responsible_expert_idx对应的发送端发完数据
                while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0  // recv not ready
                       && (wait_recv_cost = clock64() - start_time) <= NUM_TIMEOUT_CYCLES   // not timeout
                );
            }
            // Mask rank if timeout
            if (wait_recv_cost > NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for combine receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       responsible_expert_idx % num_local_experts,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + src_rank, 1);
            }

            if (combine_wait_recv_cost_stats != nullptr) {
                atomicAdd(reinterpret_cast<unsigned long long*>(combine_wait_recv_cost_stats + src_rank), wait_recv_cost);
            }
        }
    }
    cg::this_grid().sync();
    // 此时，所有rank都完成了发送，可以开始接收。

    // Reassign warp groups
    constexpr int kMaxNumGroups = 2;
    /* 每个Reduction warp负责处理 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2 字节的数据。
     总的隐藏层大小除以每个warp处理的元素数量，得到需要的Reduction warp数量。*/
    const int num_decode_warps = hidden_bf16_int4_pad / (kNumRecvUnrolls * 32);
    /*
    num_decode_warps + 1: 每个组有num_decode_warps个Reduction warp和1个TMA warp。
    (num_threads / 32) / (num_decode_warps + 1): 总warp数除以每组warp数得到理论组数。
    这种设计确保了组的数量不会超过可用资源，同时限制了最大并行度以避免资源竞争。
    */
    const int num_groups = min(kMaxNumGroups, (num_threads / 32) / (num_decode_warps + 1));
    // 如果decode_warp_idx == num_decode_warps，则是TMA warp，否则是Reduction warp。
    const int decode_warp_idx = __shfl_sync(0xffffffff, warp_id % (num_decode_warps + 1), 0);
    const int group_idx = __shfl_sync(0xffffffff, warp_id / (num_decode_warps + 1), 0);  // 组索引
    // 32 * kNumElemsPerInt4 是每个warp处理的基本单元大小。这个断言确保了内存访问的对齐和向量化效率。
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    // 每个warp需要处理num_topk个专家的权重，这个限制确保了warp内的并行处理不会超过硬件限制。
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(num_groups > 0); // 确保至少有一个有效的组。

    if (group_idx < num_groups) {
        constexpr int kNumStages = 3;
        /*
        16 * 2 = 32: 两个内存屏障的空间（full_barrier和empty_barrier各8字节，但对齐到16字节），但是后面作者又是只用了一个16字节作为两个mbarrier。
        kHidden * 2: 隐藏层大小的两倍（BF16格式，2字节/元素）。可以存储完整的隐藏层BF16格式向量，这个空间比num_tma_bytes大。
        kNumTMABufferBytes: 每个TMA缓冲区的字节大小。
        */
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        /*
        kNumBF16PerWarpBytes 表示每个warp处理BF16数据的字节数，这定义了每个Reduction warp需要存储的BF16数据量大小。
        32：warp中的线程数。 kNumRecvUnrolls = 2：接收展开因子。 kNumElemsPerInt4 = 8：每个int4包含8个BF16元素。 2：BF16的字节数。
        */
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        /*
        LogFMT量化格式下每个warp的压缩数据字节数。LogFMT是一种压缩格式，将16字节的BF16数据压缩为10字节，实现约37.5%的压缩率。
        NumBF16PerWarpBytes / 16 = 1024 / 16 = 64：每16字节的数据块。 64 * 10 = 640：压缩后的字节数
        */
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        /*
        每个量化组的元数据字节数。
        kNumDivisions = kHidden / 128：量化组的数量。 sizeof(uint32_t) = 4：32位无符号整数的字节数。
        combine发送端中，kNumMetaBytes: 每个token的量化组的元数据大小，每一组需要记录 min/max 两个float。大小是 kNumDivisions * sizeof(nv_bfloat162)。
        */
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        /*
        第一部分: 存储full_barriers、empty_barriers和tma_ld_buffers。
        第二部分: 存储tma_st_buffers。预分配的大小是kHidden * 2，这是可能的最耗内存的BF16格式存储方式。因此肯定是充足的。
        第三部分: 存储smem_group_ptr，也就是log_amax_buffers、log_amin_buffers和cast_info_buffers，每个stage的这每个部分都是kNumDivisionBytes的大小。
        */
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;

        // Reallocate shared memory
        const auto smem_group_buffer = smem_buffer + kNumBytesPerGroup * group_idx;
        auto full_barriers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes); });
        auto empty_barriers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smem_group_buffer + i * kNumTMABufferBytes + 8); });

        // tma_ld_buffers用于加载接收数据
        auto tma_ld_buffers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<uint8_t*>(smem_group_buffer + i * kNumTMABufferBytes + 16); });
        /*
        tma_st_buffers用于存储解码结果。tma_st_buffers[decode_warp_idx] 表示每个Reduction warp需要存储的BF16数据量大小。
        
        tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);

        为什么tma_st_buffers与stage无关？而tma_ld_buffers与stage有关？
        1、处理模式不同:
            tma_ld_buffers：需要stage级别的流水线，每个stage加载不同数据
            tma_st_buffers：每个warp独立处理一个token的部分数据，完成后立即写出。
        2、内存复用策略。处理完一个token后，tma_st_buffers缓冲区立即被释放，下一个token可以直接覆盖写入，无需保留多stage数据。
        */
        auto tma_st_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<uint32_t*>(smem_group_buffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes);
        });

        // Redundant when logfmt is disabled。下面这些专门存储logfmt压缩所需的元数据的。
        const auto smem_group_ptr = smem_group_buffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        // log_amax_buffers[stage_idx] 的大小是 kNumDivisionBytes 字节。log_amin_buffers[stage_idx] 和 cast_info_buffers[stage_idx]也是。
        auto log_amax_buffers =
            PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smem_group_ptr + i * kNumDivisionBytes); });
        auto log_amin_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<float*>(smem_group_ptr + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes);
        });
        auto cast_info_buffers = PatternVisitor([=](const int& i) {
            return reinterpret_cast<int*>(smem_group_ptr + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes);
        });

        uint32_t tma_phase = 0;  // 定义为32位的无符号数，每一位可以对应一个stage的相位等待状态。
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        /*
        注意: 这里的条件判断让只有TMA swap的tma_phase一开始所有的stage都设置为1，至关重要。
        这一步决定了同一个stage中TMA swap先执行，Reduction swap得等到TMA swap完成了对tma_ld_buffers[stage_idx]的数据写入
        才可以在mbarrier_wait<true>(full_barriers[stage_idx], ...)中等到屏障释放。

        在mbarrier_wait(mbarriers, tma_phase)中，当tma_phase的比特位的值和mbarriers的第63位不一致的时候，就会立即退出屏障等待。
        如果是一致的，就得等到通过mbarrier_arrive_and_expect_tx 或者 mbarrier_arrive 满足了Expected_Arrive_Count和Arrive_Count这两个条件，
        导致了mbarriers的第63位翻转，然后tma_phase的比特位才算是等到了它当时的比特位的那个相位的状态，才会退出屏障等待。
        */
        if (decode_warp_idx == num_decode_warps)  // 最后一个warp是TMA warp，则所有stage都处于等待状态。
            tma_phase = (1 << kNumStages) - 1;  // 位掩码，kNumStages=3时为0b111（7）。只使用最后的kNumStages个比特位。

        // Initialize m-barriers
        if (decode_warp_idx == num_decode_warps and lane_id < kNumStages) {
            mbarrier_init(full_barriers[lane_id], 1);  // 初始化第lane_id个stage的full屏障，期望1个TMA到达事件
            mbarrier_init(empty_barriers[lane_id], num_decode_warps);  // 初始化第lane_id个stage的empty屏障，期望num_decode_warps个到达事件
        }
        // 组内所有warp同步，确保初始化完成。组内总线程数（Reduction warp + TMA warp）
        asm volatile("bar.sync %0, %1;" ::"r"(group_idx + 1), "r"((num_decode_warps + 1) * 32));

        int stage_idx = 0, topk_idx_by_lane = 0;
        EP_STATIC_ASSERT(kNumMaxTopk <= 32, "Invalid number of topks");
        /*
        最后一个warp是TMA warp。主要有两个作用:
        1、用logfmt_check_amaxmin为LogFMT量化格式的解码做准备，确定数据处理策略。  
        2、读取数据到tma_ld_buffers[stage_idx]中。
        */
        if (decode_warp_idx == num_decode_warps) {
            // TMA load warp
            // 通过sm_id + num_sms * group_idx实现SM和组间的负载分布。 += num_sms * num_groups确保所有token都被均匀处理
            /*
            这种负载均衡的for循环写法有些规律，跳开sm、group来看，假设有两种层次的负载均衡，称为A和B，而对应层次的实际变量就是a和b，那么:
            1、循环变量token_idx的初始化就是token_idx = a + count(A) * b，或者也可以是token_idx = b + count(B) * a；
            2、边界条件: token_idx < token_count;
            3、循环步长: token_idx += count(A) * count(B)。

            如果是有更多层次的负载均衡，那么也是以此类推。如果有三层负载均衡A、B和C，那么：
            1、循环变量token_idx的初始化就是以下几种写法都行：
                A-B-C展开：token_idx = a + count(A) * (b + count(B) * c)
                A-C-B展开：token_idx = a + count(A) * (c + count(C) * b)
                B-A-C展开：token_idx = b + count(B) * (a + count(A) * c)
                B-C-A展开：token_idx = b + count(B) * (c + count(C) * a)
                C-A-B展开：token_idx = c + count(C) * (a + count(A) * b)
                C-B-A展开：token_idx = c + count(C) * (b + count(B) * a)
            2、边界条件: token_idx < token_count;
            3、循环步长: token_idx += count(A) * count(B) * count(C)。
            */
            // 下面的两重for循环是为了读取rdma_recv_x中每个token的每个专家上的token数据到共享内存tma_ld_buffers[stage_idx]中。
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk)
                    /*
                    topk_idx:[num_combined_tokens, num_topk], 当前rank作为combine发送端（也就是dispatch接收端），要发送给其他rank的token的topk的idx。
                    topk_idx_by_lane是当前lane负责的专家索引。下面的lane_id表示是num_topk中的第几个专家。
                    */
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                for (int i = 0; i < num_topk; ++i) {
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)  // 如果索引为负数，表示该位置未激活，则跳过。
                        continue;
                    // topk_idx_reg / num_local_experts计算出目标rank的ID，如果该rank被屏蔽（可能由于通信超时），则跳过处理。
                    if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                        continue;

                    /* empty_barriers[stage_idx] 没有Expected_Arrive_Count，只有Arrive_Count在初始化的时候设置为num_decode_warps。
                    mbarrier_arrive(empty_barriers[stage_idx]) 是在Reduction warps中执行的，因此当前的TMA warp要等待Reduction warps执行完这个stage。
                    */
                    mbarrier_wait<true>(empty_barriers[stage_idx], tma_phase, stage_idx);
                    // rdma_recv_x: 发写接读, num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg
                    auto buffer = static_cast<uint8_t*>(rdma_recv_x) +
                        (topk_idx_reg * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot;
                    if constexpr (kUseLogFMT) {
                    /*
                    kNumDivisions：量化组的数量
                    kNumSendUnrolls、kNumRecvUnrolls：展开因子
                    log_amax_buffers[stage_idx]：存储量化最大值的缓冲区，大小是 kNumDivisionBytes 字节 = kNumDivisions * sizeof(uint32_t);
                    log_amin_buffers[stage_idx]：存储量化最小值的缓冲区，大小是 kNumDivisionBytes 字节。
                    cast_info_buffers[stage_idx]：存储cast信息的缓冲区，大小是 kNumDivisionBytes 字节。
                    */
                        logfmt_check_amaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(
                            buffer,
                            reinterpret_cast<float2*>(log_amax_buffers[stage_idx]),
                            reinterpret_cast<float2*>(log_amin_buffers[stage_idx]),
                            cast_info_buffers[stage_idx],
                            lane_id);
                    }
                    if (elect_one_sync()) {
                        int num_casted = 0;
                        if constexpr (kUseLogFMT) {
                            const auto& info = cast_info_buffers[stage_idx][num_decode_warps - 1];
                            num_casted = (info >> 1) + (info & 1);
                        }
                        /*
                        num_casted: 需要进行cast操作的量化组数量。
                        kNumLogFMTPerWarpBytes = 640: LogFMT压缩格式下每个warp的数据大小。
                        num_decode_warps: Reduction warp的数量（通常等于 hidden_bf16_int4_pad / (kNumRecvUnrolls * 32)）。
                        kNumBF16PerWarpBytes = 1024: BF16格式下每个warp的数据大小。
                        计算逻辑：
                            对于需要cast的量化组，使用压缩的LogFMT格式（640字节/warp）。
                            对于不需要cast的量化组，使用原始BF16格式（1024字节/warp）。
                            总传输字节数 = 压缩数据 + 未压缩数据。
                        */
                        int num_tma_bytes = num_casted * kNumLogFMTPerWarpBytes + (num_decode_warps - num_casted) * kNumBF16PerWarpBytes;
                        // 提交TMA读操作，将buffer中的数据加载到tma_ld_buffers[stage_idx]中。
                        tma_load_1d(
                            tma_ld_buffers[stage_idx], buffer + (kUseLogFMT ? kNumMetaBytes : 0), full_barriers[stage_idx], num_tma_bytes);
                        // num_tma_bytes 是当前线程的TMA读操作要传输的字节数。
                        mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes);
                    }
                    __syncwarp();
                    stage_idx = (stage_idx + 1) % kNumStages;  // 进入下一个stage。
                }
            }
        } else {
            // Reduction warps  负责对TMA加载的数据进行解码、累加计算，并将最终结果写回全局内存。
            float topk_weights_by_lane;  // 当前lane负责的topk权重值
            for (int token_idx = sm_id + num_sms * group_idx; token_idx < num_combined_tokens; token_idx += num_sms * num_groups) {
                if (lane_id < num_topk) {
                    // topk_idx:[num_combined_tokens, num_topk], 当前rank作为combine发送端（也就是dispatch接收端），要发送给其他rank的token的topk的idx。
                    topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                    // topk_weights:[num_combined_tokens, num_topk], 当前rank作为combine发送端（也就是dispatch接收端），要发送给其他rank的token的topk权重。
                    topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
                }
                __syncwarp();

                /* 初始化累加缓冲区，用于存储所有专家贡献的累加结果。每个int4包含kNumElemsPerInt4 = 8个BF16元素，kNumRecvUnrolls = 2：接收展开因子。
                kNumElemsPerInt4 * kNumRecvUnrolls 是每个线程要处理的token_idx的所有专家加权求和后的BF16元素数量。
                */
                float combined_values[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                for (int i = 0; i < num_topk; ++i) {
                    int topk_idx_reg = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (topk_idx_reg < 0)
                        continue;
                    if (is_rank_masked(mask_buffer_ptr, topk_idx_reg / num_local_experts))
                        continue;
                    /*
                    这里为什么topk_weight是引用类型？
                    __shfl_sync 的返回值在某些CUDA版本中可能有特殊的寄存器绑定，使用引用可以确保直接访问而不产生额外的寄存器移动。
                    这样，topk_weight直接绑定到shuffle的结果所在的寄存器，无额外开销。
                    而且，在PTX层面:
                    使用引用: const auto& topk_weight = ...  // 直接使用寄存器
                    使用值:  const auto topk_weight = ...   // 可能产生mov指令
                    */
                    const auto& topk_weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);

                    // 这里已经等到了mbarrier_arrive_and_expect_tx(full_barriers[stage_idx], num_tma_bytes)的所有线程的事务条件和到达条件。
                    mbarrier_wait<true>(full_barriers[stage_idx], tma_phase, stage_idx);
                    if constexpr (kUseLogFMT) {
                        const auto& info = cast_info_buffers[stage_idx][decode_warp_idx];
                        bool enable_cast = info & 1;
                        int num_casted_prefix = info >> 1;
                        int tma_offset =
                            kNumLogFMTPerWarpBytes * num_casted_prefix + kNumBF16PerWarpBytes * (decode_warp_idx - num_casted_prefix);
                        int division_idx = decode_warp_idx * (kNumRecvUnrolls * 2) + lane_id * kNumRecvUnrolls / 16;
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset +
                                                        (enable_cast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * lane_id),
                            combined_values,
                            log_amax_buffers[stage_idx][division_idx],
                            log_amin_buffers[stage_idx][division_idx],
                            enable_cast,
                            topk_weight);
                    } else {
                        // kNumBF16PerWarpBytes 表示每个warp处理BF16数据的字节数，这定义了每个Reduction warp需要存储的BF16数据量大小。
                        // 计算当前解码warp在共享内存缓冲区中的数据偏移量
                        int tma_offset = kNumBF16PerWarpBytes * decode_warp_idx;
                        /*
                        解码量化数据并累加到结果数组combined_values中。读取共享内存中的数据，进行必要的解码，然后按权重累加到combined_values数组中。
                        + tma_offset：加上warp级别的偏移，定位到当前warp的数据块。
                        + kNumBF16PerWarpBytes / 32 * lane_id：加上lane级别的偏移。
                            kNumBF16PerWarpBytes / 32 = 1024 / 32 = 32：每个lane处理的字节数。
                        所以，这里的reinterpret_cast<uint32_t*>(...)指向当前lane负责的BF16数据块的指针。大小是kNumElemsPerInt4 * kNumRecvUnrolls个BF16元素。
                        两个0分别是amax参数和amin参数。
                        false 是enable_cast参数，表示不需要进行cast操作。在非LogFMT模式下，数据已经是BF16格式，无需转换。
                        topk_weight 是 token token_idx对专家topk_idx_reg的权重。在累加过程中，解码后的数据会乘以这个权重后累加到结果中。
                        */
                        decode_and_accumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tma_ld_buffers[stage_idx] + tma_offset + kNumBF16PerWarpBytes / 32 * lane_id),
                            combined_values,
                            0,
                            0,
                            false,
                            topk_weight);
                    }

                    if (elect_one_sync())
                        // 当前Reduction warp已经处理完了一个token，可以通知TMA warp可以开始处理下一个token了。
                        mbarrier_arrive(empty_barriers[stage_idx]);
                    stage_idx = (stage_idx + 1) % kNumStages;  // token token_idx的对应topk专家的下一个专家的权重计算，在下一个stage中进行。
                }
                // 使执行线程进入等待状态，直到当前执行线程最近的批量异步组都执行完成。这里是为了等待之前tma_store_1d提交的批量异步组。
                tma_store_wait<0>();

                #pragma unroll
                /*
                每个线程负责处理kNumRecvUnrolls * 4个__nv_bfloat162元素。
                */
                for (int k = 0; k < kNumRecvUnrolls * 4; ++k) { // 这里的4表示每个int4包含4个__nv_bfloat162元素，也就是8个BF16元素。
                    /* 
                    这里的k表示是第几个__nv_bfloat162元素。
                    这里的k * 2表示是第几个int4元素，0和1分别表示kNumRecvUnrolls = 2中的循环展开的轮数。
                    */
                    auto combined_pack = __nv_bfloat162(combined_values[k * 2], combined_values[k * 2 + 1]);
                    /*
                    *reinterpret_cast<uint32_t*>(&combined_pack)的第一个指针是解引用操作符，从指针中提取值。
                    这是为了将__nv_bfloat162类型的值重新解释为uint32_t类型的值。

                    每个warp负责处理kNumRecvUnrolls * 4 * 32个__nv_bfloat162元素。
                    下面的k表示当前lane处理的kNumRecvUnrolls * 4个__nv_bfloat162元素中的第k个元素。
                    */
                    tma_st_buffers[decode_warp_idx][kNumRecvUnrolls * 4 * lane_id + k] = *reinterpret_cast<uint32_t*>(&combined_pack);
                }
                tma_store_fence();  // 官方文档最权威: 等待共享内存写入操作对 TMA 引擎可见。
                if (elect_one_sync()) {
                    /*
                    combined_x:要写入的, [num_combined_tokens, hidden], 就是combine输出端输出的tensor，将同一个token经过多个激活专家输出后的结果进行聚合后的结果。
                    token_idx * hidden_bf16_int4: 是以int4为单位的token token_idx的hidden在共享内存tma_st_buffers的偏移量。
                    每个线程处理kNumRecvUnrolls个int4数据。32个线程就是kNumRecvUnrolls * 32个int4数据。
                    decode_warp_idx * kNumRecvUnrolls * 32: 是以int4为单位的当前warp在warp级别的偏移量。
                    */
                    tma_store_1d(tma_st_buffers[decode_warp_idx],
                                 static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + decode_warp_idx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                }
                __syncwarp();
            }
        }
    }
}

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy) {
    constexpr int kNumMaxTopk = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms =
        max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm));

    // Check workspace
    // CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream)); 设置了这个值初始化为0
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    // Online cast cannot use zero-copy
    // LogFMT量化需要在线转换（从x_int4读取原始BF16数据并转换为LogFMT格式），
    // 而zero-copy模式假设数据已经预先写入发送缓冲区（buf_ptr），跳过了从x_int4的读取步骤。
    // 因此zero-copy模式与LogFMT量化不兼容。
    EP_HOST_ASSERT(not(zero_copy and use_logfmt));

    constexpr int kNumStages = 3;
    constexpr int kNumMaxUnrolls = 4;
    constexpr int kMaxNumGroups = 2;

    // Send buffer size
    const int num_meta_bytes = hidden / 128 * 4;
    const int num_send_tma_bytes = 32 * sizeof(int4) * kNumMaxUnrolls + 16;
    const int smem_send_size = num_warps * (kNumStages * num_send_tma_bytes + num_meta_bytes);

    // Receive buffer size
    const int num_recv_tma_bytes = 16 + hidden * 2;
    const int smem_recv_size = kMaxNumGroups * (kNumStages * num_recv_tma_bytes + hidden * 2 + kNumStages * num_meta_bytes * 3);

    // Total requirement
    const int smem_size = max(smem_send_size, smem_recv_size);

#define COMBINE_LAUNCH_CASE(hidden)                                                                                                \
    {                                                                                                                              \
        auto combine_func =                                                                                                        \
            use_logfmt ? combine<true, hidden, kNumMaxTopk, kNumMaxUnrolls> : combine<false, hidden, kNumMaxTopk, kNumMaxUnrolls>; \
        SET_SHARED_MEMORY_FOR_TMA(combine_func);                                                                                   \
        LAUNCH_KERNEL(&cfg,                                                                                                        \
                      combine_func,                                                                                                \
                      combined_x,                                                                                                  \
                      rdma_recv_x,                                                                                                 \
                      rdma_recv_flag,                                                                                              \
                      rdma_send_x,                                                                                                 \
                      x,                                                                                                           \
                      topk_idx,                                                                                                    \
                      topk_weights,                                                                                                \
                      src_info,                                                                                                    \
                      layout_range,                                                                                                \
                      mask_buffer_ptr,                                                                                             \
                      combine_wait_recv_cost_stats,                                                                                \
                      next_clean,                                                                                                  \
                      num_next_clean_int,                                                                                          \
                      atomic_clean_flag,                                                                                           \
                      num_combined_tokens,                                                                                         \
                      hidden,                                                                                                      \
                      num_topk,                                                                                                    \
                      num_max_dispatch_tokens_per_rank,                                                                            \
                      num_experts,                                                                                                 \
                      rank,                                                                                                        \
                      num_ranks,                                                                                                   \
                      num_warp_groups,                                                                                             \
                      num_warps_per_group,                                                                                         \
                      phases,                                                                                                      \
                      zero_copy);                                                                                                  \
    }                                                                                                                              \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor) {
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = num_sms * kNumThreads;
    const auto thread_id = sm_id * kNumThreads + static_cast<int>(threadIdx.x);
    for (int rank_id = thread_id; rank_id < num_ranks; rank_id += num_threads) {
        mask_tensor[rank_id] = mask_buffer_ptr[rank_id];
    }
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 1024;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, query_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks, mask_tensor);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    if (sm_id == 0 && thread_id == 0) {
        atomicExch(mask_buffer_ptr + rank_to_mask, mask ? 1 : 0);
    }
}

void update_mask_buffer(int* mask_buffer_ptr, int rank, bool mask, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, update_mask_buffer<kNumThreads>, mask_buffer_ptr, rank, mask);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks) {
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_ranks; i += kNumThreads)
        mask_buffer_ptr[i] = 0;
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks);
}

}  // namespace internode_ll

}  // namespace deep_ep
