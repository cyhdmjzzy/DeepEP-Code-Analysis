#include <functional>
#include <optional>

#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace deep_ep {

namespace internode {

/*
cpu_rdma_team是一个nvshmem_team_t类型的变量，这个变量在其他地方定义（可能在其他源文件中），但在这个文件中会使用。
extern关键字告诉编译器这个变量的定义在外部，不要为它分配内存。
*/
extern nvshmem_team_t cpu_rdma_team;

struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    // TODO: faster encoding
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
        #pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const { return (is_token_in_nvl_rank_bits >> nvl_rank) & 1; }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

int get_source_meta_bytes() {
    return sizeof(SourceMeta);
}

__host__ __device__ __forceinline__ int get_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align_up(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) + num_scales * sizeof(float) +
                                         num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float),
                                     sizeof(int4)));
}

__host__ __device__ __forceinline__ std::pair<int, int> get_rdma_clean_meta(int hidden_int4,
                                                                            int num_scales,
                                                                            int num_topk_idx,
                                                                            int num_topk_weights,
                                                                            int num_rdma_ranks,
                                                                            int num_rdma_recv_buffer_tokens,
                                                                            int num_channels) {
    // Return `int32_t` offset and count to clean
    // return {(get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens *
    //     num_rdma_ranks * 2 * num_channels) /
    //        sizeof(int),
    //    (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels};

    /*
    rdma_clean_offset: RDMA 缓冲区中 token 数据区域分配的字节数，也是结束位置的偏移量（以 int 为单位）。
    get_num_bytes_per_token(...): 每个 token 的字节数。包含：hidden data + SourceMeta + scales + topk_idx + topk_weights。对齐到 16 字节边界
    num_rdma_recv_buffer_tokens: 每个 RDMA rank 每个 channel 的接收缓冲区 token 数量。
    * 2：每个 RDMA rank 有发送和接收两个缓冲区（decoupled 模式）
    / sizeof(int)：转换为 int 单位（4 字节）
    */
    int rdma_clean_offset = (get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) 
                            * num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_channels) / sizeof(int);
    
    /*
    rdma_num_int_clean = (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels
        NUM_MAX_NVL_PEERS * 2 + 4：每个 RDMA rank 对应每个 channel 的元数据大小（以 int 为单位）
            NUM_MAX_NVL_PEERS * 2：NVL rank 相关的元数据（每个 NVL rank 2 个 int）
            + 4：额外的 4 个 int（RDMA 相关的元数据）
        * 2: 每个 RDMA rank 有发送和接收两个缓冲区。
    当前rank是要负责对所有的节点和channel进行清零操作，所以需要清零的int数量是要乘以 rdma_num_int_clean * num_channels。
    */
    int rdma_num_int_clean = (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels
    
    /*
    由上面的计算可知每个rank的SymBuffer的整体内存布局:
    +===============================================================+
    | Token数据区域 (rdma_clean_offset 之前)                          |
    |                                                               |
    | Channel 0:                                                    |
    |   Rank 0 Send: [token 0] [token 1] ... [token N-1]            |
    |   Rank 0 Recv: [token 0] [token 1] ... [token N-1]            |
    |   Rank 1 Send: [token 0] [token 1] ... [token N-1]            |
    |   Rank 1 Recv: [token 0] [token 1] ... [token N-1]            |
    |   ...                                                         |
    |   Rank K Send: [token 0] [token 1] ... [token N-1]            |
    |   Rank K Recv: [token 0] [token 1] ... [token N-1]            |
    |                                                               |
    | Channel 1:                                                    |
    |   Rank 0 Send: [token 0] [token 1] ... [token N-1]            |
    |   Rank 0 Recv: [token 0] [token 1] ... [token N-1]            |
    |   ...                                                         |
    |                                                               |
    | Channel M:                                                    |
    |   ...                                                         |
    +===============================================================+
    | 元数据区域 (rdma_clean_offset 开始，rdma_num_int_clean 个 int)   |
    |                                                               |
    | Channel 0:                                                    |
    |   Rank 0 Send Meta: [(NUM_MAX_NVL_PEERS * 2 + 4) = 20个int]   |
    |   Rank 0 Recv Meta: [(NUM_MAX_NVL_PEERS * 2 + 4) = 20个int]   |
    |   Rank 1 Send Meta: [(NUM_MAX_NVL_PEERS * 2 + 4) = 20个int]   |
    |   ...                                                         |
    |                                                               |
    | Channel M:                                                    |
    |   ...                                                         |
    +===============================================================+
    | Head/Tail指针区域                                              |
    |                                                               |
    | Channel 0:                                                    |
    |   Rank 0 Head: [8 bytes]                                      |
    |   Rank 0 Tail: [8 bytes]                                      |
    |   Rank 1 Head: [8 bytes]                                      |
    |   ...                                                         |
    |                                                               |
    | Channel M:                                                    |
    |   ...                                                         |
    +===============================================================+
    */
    return {rdma_clean_offset, rdma_num_int_clean};
}

__host__ __device__ __forceinline__ std::pair<int, int> get_nvl_clean_meta(int hidden_int4,
                                                                           int num_scales,
                                                                           int num_topk_idx,
                                                                           int num_topk_weights,
                                                                           int num_rdma_ranks,
                                                                           int num_nvl_ranks,
                                                                           int num_nvl_recv_buffer_tokens,
                                                                           int num_channels,
                                                                           bool is_dispatch) {
    // Return `int32_t` offset and to clean
    EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

    return {
        (num_nvl_recv_buffer_tokens * get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_nvl_ranks *
         num_channels) /
            sizeof(int),
        num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels,
    };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    /* 
    实际上，不管是不是kLowLatencyMode，最终还都是不同节点的相同nvl_rank的GPU到GPU之间的通信。
    ll模式就是不同节点间只在相同nvl_rank的GPU之间进行通信。
    而普通模式下是以每个节点为PE，直接节点对节点通信，但是根据PE之间的对称内存性质，发送端节点和接收端节点通信依然是在对应的nvl_rank上通信的。
    */
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

template <bool kLowLatencyMode>
__forceinline__ __device__ void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
    /*
    通过传入特定的 nvshmem_rdma_team，可以控制整个集群中哪些GPU参与RDMA（远程直接内存访问）通信的同步，确保只有相关的GPU进程协同工作，提高通信效率。
    低延迟模式：使用nvshmem_sync(rdma_team)只同步指定 team 内的GPU，也就是不同节点上的具有相同nvl_rank的GPU作为一个team。
    普通模式：使用nvshmem_sync_all()同步所有GPU。

    nvshmem_sync和nvshmem_sync_all会阻塞调用 PE，直到指定team中的所有 PE 都调用了 nvshmem_sync 或 nvshmem_sync_all 后才返回。
    在多线程 NVSHMEM 程序中，仅调用线程会被阻塞。
    与 nvshmem_barrier 不同，nvshmem_sync仅确保先前发起的**内存存储操作（指本地PE对自己内存的常规存储操作）**完成并可见（本地PE自己可见），
    **不确保**通过 NVSHMEM 例程发起的远程内存更新完成。
    */
    kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();
}

template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,
                                int* moe_recv_counter_mapped,
                                int num_ranks,
                                const int* num_tokens_per_rdma_rank,
                                int* moe_recv_rdma_counter_mapped,
                                const int* num_tokens_per_expert,
                                int* moe_recv_expert_counter_mapped,
                                int num_experts,
                                const bool* is_token_in_rank,
                                int num_tokens,
                                int num_worst_tokens,
                                int num_channels,
                                int expert_alignment,
                                const int rdma_clean_offset,
                                const int rdma_num_int_clean,
                                const int nvl_clean_offset,
                                const int nvl_num_int_clean,
                                int* rdma_channel_prefix_matrix,
                                int* recv_rdma_rank_prefix_sum,
                                int* gbl_channel_prefix_matrix,
                                int* recv_gbl_rank_prefix_sum,
                                void* rdma_buffer_ptr,
                                void** buffer_ptrs,
                                int** barrier_signal_ptrs,
                                int rank,
                                const nvshmem_team_t rdma_team) {
    /*
    num_tokens_per_rank:            kNumRanks,                          要读取的。表示当前的rank rank发送到节点内各个rank的token数量。
    moe_recv_counter_mapped:        1,                                  要写入的。表示所有rank发送到rank rank的token数量之和。
    num_ranks:                      1,                                  要读取的。表示整个训练集群中的rank数量。
    num_tokens_per_rdma_rank:       kNumRDMARanks,                      要读取的。表示当前的rank rank发送到节点内各个RDMA rank的token数量。
    moe_recv_rdma_counter_mapped:   1                                   要写入的。
    num_tokens_per_expert:          [kNumRanks, num_experts_per_rank],  要读取的。num_tokens_per_expert[i * num_experts_per_rank + j] 表示rank rank要发送到rank i内的expert j的token数量。
    moe_recv_expert_counter_mapped: num_experts_per_rank,               要写入的。moe_recv_expert_counter_mapped[i] 表示rank rank内的expert i收到的token数量。
    num_experts:                    1,                                  要读取的。表示节点内所有rank的专家数量之和。
    is_token_in_rank:               [num_tokens, kNumRanks],            要读取的。表示当前rank要发送的token是否需要发送到节点内其他rank。
    num_tokens:                     1,                                  要读取的。表示当前rank要发送的token数量。
    num_worst_tokens:               1,                                  要读取的。
    num_channels:                   1,                                  要读取的。表示接下来进行dispatch通信使用的channel数量。
    expert_alignment:               1,                                  要读取的。rank rank内的expert thread_id收到的token数量以expert_alignment为对齐单位。
    rdma_clean_offset:              1,                                  要读取的。表示RDMA缓冲区需要清零的偏移量。
    rdma_num_int_clean:             1,                                  要读取的。表示RDMA缓冲区需要清零的int数量。
    nvl_clean_offset:               1,                                  要读取的。表示NVLink缓冲区需要清零的偏移量。
    nvl_num_int_clean:              1,                                  要读取的。表示NVLink缓冲区需要清零的int数量。
    rdma_channel_prefix_matrix:     [kNumRDMARanks, num_channels],      要写入的。rdma_channel_prefix_matrix[dst_rank * num_channels + i]表示的是dispatch发送端rank rank从channel 0到channel i（包含channel i）累计要发送到rdmarank dst_rank的token数量之和（channel级别前缀和）。
    recv_rdma_rank_prefix_sum:      [kNumRDMARanks, kNumRDMARanks],     要写入的。recv_rdma_rank_prefix_sum[i * kNumRanks + j]表示的是从rdmarank 0到rdmarank i累计发送到rdmarank j的token数量之和。
    gbl_channel_prefix_matrix:      [kNumRanks, num_channels],          要写入的。gbl_channel_prefix_matrix[dst_rank * num_channels + i]表示的是dispatch发送端rank rank从channel 0到channel i（包含channel i）累计要发送到rank dst_rank的token数量之和（channel级别前缀和）。
    recv_gbl_rank_prefix_sum:       [kNumRanks, kNumRanks],             要写入的。recv_gbl_rank_prefix_sum[i * kNumRanks + j]表示的是从rank 0到rank i累计发送到rank j的token数量之和。
    rdma_buffer_ptr:                void*                               要写入的。大小是 int(1e9) 字节。
    buffer_ptrs:                    kNumRanks, void**                   要写入的。节点内所有rank的NVLink buffer。
    barrier_signal_ptrs:            [kNumRanks, kNumRanks],             要读写的。节点内的所有rank同步所需的barrier信号。
    rank:                           1,                                  要读取的。表示当前rank在集群内的唯一编号。数量是节点数乘以8。
    rdma_team:                      nvshmem_team_t                      要读取的。通过传入特定的nvshmem_team_t，可以控制哪些GPU参与RDMA（远程直接内存访问）通信的同步，确保只有相关的GPU进程协同工作，提高通信效率。     
    */
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;

    // rdma_rank: 当前节点编号。nvl_rank: 当前节点内GPU编号（从0到7）。
    auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    // num_rdma_experts: 每个节点负责的专家数量。num_nvl_experts: 节点内每个GPU负责的专家数量。
    auto num_rdma_experts = num_experts / kNumRDMARanks, num_nvl_experts = num_rdma_experts / NUM_MAX_NVL_PEERS;

    if (sm_id == 0) {  // 全局协调任务。负责RDMA通信、数据聚合、同步。
        // Communication with others
        // Global barrier: the first warp does intra-node sync, the second warp does internode sync
        EP_DEVICE_ASSERT(num_warps > 1);
        EP_DEVICE_ASSERT(kNumRDMARanks <= num_threads);

        // waiting for all previous inflight wrs to complete,
        // in case of rewriting cleared rdma_buffer
        /*
        num_rc_per_pe: int类型，每个rank的RC QP数量。
        num_devices_initialized: 当前 PE 选择并成功初始化的 NIC 设备数量。
        qps_per_rdma_rank: qps_per_rdma_rank表示的是当前PE每次往某个节点发送数据的qp数量。
        */
        auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
        /* 
        等待所有之前进行的 RDMA 操作完成，确保缓冲区清理时不会与正在进行的写操作冲突。
        PE不用网自身所在节点发送rdma数据，所以需要减去1。这意味着需要等待的QP总数 = 当前PE对每个目标节点的QP数量 × 目标节点数量（不包括自己）
        */
        for (int i = thread_id; i < qps_per_rdma_rank * (kNumRDMARanks - 1); i += num_threads) {
            /*
            使用模运算的循环偏移（Circular Shift）使得dst_rdma_rank的值在[0, kNumRDMARanks-1]之间循环，
            但是恰好避开rdma_rank，从而实现对所有目标节点的QP进行等待。 
            (i / qps_per_rdma_rank + 1) ∈ [1, kNumRDMARanks - 1]，是变量。
            rdma_rank ∈ [0, kNumRDMARanks - 1]，是常量。
            (i / qps_per_rdma_rank + rdma_rank + 1) ∈ [rdma_rank + 1, rdma_rank + kNumRDMARanks - 1]，是变量。
            这个变量对kNumRDMARanks取模，得到的是[0, kNumRDMARanks-1]中除了rdma_rank的任何一个整数。
            */
            auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % kNumRDMARanks;
            // 在每个目标节点的QP中选择具体的QP编号
            auto qp_id = i % qps_per_rdma_rank;
            // 等待指定目标 PE 的指定 QP 上所有已提交的 RDMA 操作完成。用于确保在清理缓冲区或进行同步操作前，所有未完成的 RDMA 写操作已完成。
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), qp_id);
        }
        __syncthreads();

        if (thread_id == 32)
            /*
            低延迟模式：使用nvshmem_sync(rdma_team)只同步指定 team 内的GPU，也就是不同节点上的具有相同nvl_rank的GPU作为一个team。
            普通模式：使用nvshmem_sync_all()同步所有节点。
            */
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        // 同步节点内的跨 rank 的 barrier， 等待节点内的所有 rank 到达。
        // 第二个模板参数为true表示执行内存屏障：确保所有之前的内存操作对同一个节点内的所有GPU的所有线程可见。
        barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

        // Send numbers of tokens per rank/expert to RDMA ranks
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        /* 
        NUM_MAX_NVL_PEERS + num_rdma_experts + 1: 当前rank对每个节点要传输的数据的字节数。
        rdma_recv_num_tokens_mixed 是属于当前rank的SymBuffer，用于存储当前rank要发送给其他rank的token数量。
        */
        auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS + num_rdma_experts + 1, kNumRDMARanks);

        // Clean up for later data dispatch
        EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));
        #pragma unroll
        /*
        rdma_clean_offset 和 rdma_num_int_clean 的计算参见 get_rdma_clean_meta 函数。
        rdma_num_int_clean 表示: 需要清零的 RDMA 元数据区域的 int 数量。
        rdma_num_int_clean = (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels
        */
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;  // 清零元数据部分

        // Copy to send buffer
        #pragma unroll
        for (int i = thread_id; i < num_ranks; i += num_threads)
            /* 
            存储当前rank要发往每个rank的token数量。每个节点要设置 NUM_MAX_NVL_PEERS 个int值
            (i / NUM_MAX_NVL_PEERS) 表示第几个节点。(i % NUM_MAX_NVL_PEERS) 表示节点内的nvl_rank。
            */
            rdma_recv_num_tokens_mixed.send_buffer(i / NUM_MAX_NVL_PEERS)[i % NUM_MAX_NVL_PEERS] = num_tokens_per_rank[i];
        #pragma unroll
        for (int i = thread_id; i < num_experts; i += num_threads)
            /* 
            存储当前rank要发往每个专家的token数量。每个节点要设置 num_rdma_experts 个int值。
            num_rdma_experts: 每个节点负责的专家数量
            (i / num_rdma_experts): 表示rdma rank编号
            (i % num_rdma_experts): 表示节点内的局部专家编号。加上 NUM_MAX_NVL_PEERS 是因为要偏移上面存储每个rank的token数量的信息。
            */
            rdma_recv_num_tokens_mixed.send_buffer(i / num_rdma_experts)[NUM_MAX_NVL_PEERS + i % num_rdma_experts] =
                num_tokens_per_expert[i];
        if (thread_id < kNumRDMARanks)
            // 存储当前rank要发往每个节点的token数量。每个节点要设置 1 个int值。
            rdma_recv_num_tokens_mixed.send_buffer(thread_id)[NUM_MAX_NVL_PEERS + num_rdma_experts] = num_tokens_per_rdma_rank[thread_id];
        __syncthreads();  // if条件中使用了thread_id，后面往往跟这个。

        // Issue send
        // TODO: more light fence or barrier or signaling
        // TODO: overlap EP barrier and NVL cleaning
        for (int i = warp_id; i < kNumRDMARanks; i += num_warps) {
            if (i != rdma_rank) {  // 需要跨节点传输
                /* 
                把当前rank要发送给节点i的rank nvl_rank的token数量信息，发送给节点i的rank nvl_rank的相应对称内存中。

                rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank) 表示远程节点i的rank nvl_rank的对称内存中存储当前节点rdma_rank的nvl_rank的元数据的指针。
                rdma_recv_num_tokens_mixed.send_buffer(i) 表示当前rank要发送给节点 i的token数量信息的指针。在本地RDMA缓冲区中。
                */
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.send_buffer(i)),
                                                  (NUM_MAX_NVL_PEERS + num_rdma_experts + 1) * sizeof(int),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(i, nvl_rank),
                                                  0,  // qp_id 是0，后面nvshmemi_ibgda_quiet进行等待rdma完成的qp也是0
                                                  lane_id,
                                                  0);
            } else {
                // 不需要跨节点传输，直接在本地RDMA缓冲区中传输。注意: 这里的rdma_recv_num_tokens_mixed是rank级别的对称内存，不是sm级别的。
                UNROLLED_WARP_COPY(1,
                                   lane_id,
                                   NUM_MAX_NVL_PEERS + num_rdma_experts + 1,
                                   rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank),
                                   rdma_recv_num_tokens_mixed.send_buffer(i),
                                   ld_volatile_global,
                                   st_na_global);
            }
        }
        __syncthreads();

        // Wait previous operations to be finished
        if (thread_id < kNumRDMARanks and thread_id != rdma_rank)
            // 等待指定目标 PE 的指定 QP 上所有已提交的 RDMA 操作完成。用于确保上面使用nvshmemi_ibgda_put_nbi_warp的rdma操作都完成。
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank), 0);
        __syncthreads();  // if条件中使用了thread_id，后面往往跟这个。

        // Barrier
        if (thread_id == 0)
            /*
            仅确保先前发起的**内存存储操作（指本地PE对自己内存的常规存储操作）**完成并可见（本地PE自己可见）
            */
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        __syncthreads();  // if条件中使用了thread_id，后面往往跟这个。

        // NVL buffers
        // 注意: nvl_send_buffer 不是当前rank的nvl_rank的缓冲区，而是当前节点的local_rank thread_id的缓冲区。
        auto nvl_send_buffer = thread_id < NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
        auto nvl_recv_buffer = buffer_ptrs[nvl_rank];
        /* 
        nvl_reduced_num_tokens_per_expert的长度是num_rdma_experts，每个元素记录所有节点要发送到当前节点上的专家 thread_id的token数量之和。
        advance_also(nvl_send_buffer) 让nvl_send_buffer也和nvl_recv_buffer一样偏移了num_rdma_experts个int元素的字节。
        */
        auto nvl_reduced_num_tokens_per_expert = Buffer<int>(nvl_recv_buffer, num_rdma_experts).advance_also(nvl_send_buffer);
        /*
        nvl_send_num_tokens_per_rank 的长度是NUM_MAX_NVL_PEERS，每个元素长度是kNumRDMARanks个int。
        注意: 这每个int值表示rank thread_id要发送给节点内的各个rank的来自各个节点的token数量。
        */
        auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);

        // Clean up for later data dispatch
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                             nvl_send_num_tokens_per_expert.total_bytes <=
                         nvl_clean_offset * sizeof(int));
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

        // Reduce number of tokens per expert into the NVL send buffer
        // TODO: may use NVSHMEM reduction
        /*
        一个节点往往有8个GPU，而DeepSeek-V3的每个GPU托管8个专家，所以num_rdma_experts就是8*8=64。
        而 num_threads 是当前sm中的线程数，是512，远大于64.
        */
        EP_DEVICE_ASSERT(num_rdma_experts <= num_threads);
        if (thread_id < num_rdma_experts) {  // thread_id表示节点内的局部专家编号。
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i)
                // 对所有节点要发送到当前rank所在节点的托管的专家编号为thread_id的专家的token数量进行累加。
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + thread_id];
            nvl_reduced_num_tokens_per_expert[thread_id] = sum;
        }
        __syncthreads();

        // Reduce RDMA received tokens
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i) {
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + num_rdma_experts];
                recv_rdma_rank_prefix_sum[i] = sum;
            }
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1)
                    ;
                *moe_recv_rdma_counter_mapped = sum;
            }
        }

        // Send numbers of tokens per rank/expert to NVL ranks
        EP_DEVICE_ASSERT(NUM_MAX_NVL_PEERS <= num_threads);
        if (thread_id < NUM_MAX_NVL_PEERS) {
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i)
                /*
                rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id] 表示节点i的local_rank nvl_rank要发送给当前节点的local_rank thread_id的token数量。
                但是这个发送要先发送到当前节点的local_rank nvl_rank，然后由当前节点的local_rank nvl_rank发送给当前节点的local_rank thread_id。
                
                nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] 表示的是 i nvl_rank要发送给rdma_rank thread_id的token数据量。
                注意: nvl_send_num_tokens_per_rank 是当前节点的local_rank thread_id的nvlink缓冲区。
                */
                nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] = rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id];
            #pragma unroll
            for (int i = 0; i < num_nvl_experts; ++i)
                nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] = nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i];
        }
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);

        // Reduce the number of tokens per rank/expert
        EP_DEVICE_ASSERT(num_nvl_experts <= num_threads);
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < num_ranks; ++i) {
                int src_rdma_rank = i / NUM_MAX_NVL_PEERS, src_nvl_rank = i % NUM_MAX_NVL_PEERS;
                sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank];
                recv_gbl_rank_prefix_sum[i] = sum;
            }
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_counter_mapped) != -1)
                    ;
                *moe_recv_counter_mapped = sum;
            }
        }
        if (thread_id < num_nvl_experts) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                sum += nvl_recv_num_tokens_per_expert.buffer(i)[thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1)
                    ;
                moe_recv_expert_counter_mapped[thread_id] = sum;
            }
        }

        // Finally barrier
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else {  // 并行计算任务。负责token统计和前缀和计算
        // Calculate meta data
        int dst_rdma_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            // 统计每个channel需要发送到每个目标rdma_rank内每个nvl_rank的token数量
            int total_count = 0, per_nvl_rank_count[NUM_MAX_NVL_PEERS] = {0};
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
                EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
                auto is_token_in_rank_uint64 =
                    *reinterpret_cast<const uint64_t*>(is_token_in_rank + i * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS);
                auto is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_uint64);
                #pragma unroll
                for (int j = 0; j < NUM_MAX_NVL_PEERS; ++j)
                    per_nvl_rank_count[j] += is_token_in_rank_values[j];
                total_count += (is_token_in_rank_uint64 != 0);
            }

            // Warp reduce
            total_count = warp_reduce_sum(total_count);
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                per_nvl_rank_count[i] = warp_reduce_sum(per_nvl_rank_count[i]);

            // Write into channel matrix
            if (elect_one_sync()) {
                #pragma unroll
                // 记录聚合结果到全局矩阵
                for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                    gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + i) * num_channels + channel_id] = per_nvl_rank_count[i];
                rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = total_count;
            }
        }

        // Calculate prefix sum
        __syncthreads();
        if (thread_id == 0) {
            auto prefix_row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];
        }

        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        if (thread_id < NUM_MAX_NVL_PEERS) {
            auto prefix_row = gbl_channel_prefix_matrix + (dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id) * num_channels;
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                prefix_row[i] += prefix_row[i - 1];
        }
    }
}

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_worst_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool low_latency_mode) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                                    \
    {                                                                                                                                  \
        auto notify_dispatch_func = low_latency_mode ? notify_dispatch<true, num_rdma_ranks> : notify_dispatch<false, num_rdma_ranks>; \
        LAUNCH_KERNEL(&cfg,                                                                                                            \
                      notify_dispatch_func,                                                                                            \
                      num_tokens_per_rank,                                                                                             \
                      moe_recv_counter_mapped,                                                                                         \
                      num_ranks,                                                                                                       \
                      num_tokens_per_rdma_rank,                                                                                        \
                      moe_recv_rdma_counter_mapped,                                                                                    \
                      num_tokens_per_expert,                                                                                           \
                      moe_recv_expert_counter_mapped,                                                                                  \
                      num_experts,                                                                                                     \
                      is_token_in_rank,                                                                                                \
                      num_tokens,                                                                                                      \
                      num_worst_tokens,                                                                                                \
                      num_channels,                                                                                                    \
                      expert_alignment,                                                                                                \
                      rdma_clean_meta.first,                                                                                           \
                      rdma_clean_meta.second,                                                                                          \
                      nvl_clean_meta.first,                                                                                            \
                      nvl_clean_meta.second,                                                                                           \
                      rdma_channel_prefix_matrix,                                                                                      \
                      recv_rdma_rank_prefix_sum,                                                                                       \
                      gbl_channel_prefix_matrix,                                                                                       \
                      recv_gbl_rank_prefix_sum,                                                                                        \
                      rdma_buffer_ptr,                                                                                                 \
                      buffer_ptrs,                                                                                                     \
                      barrier_signal_ptrs,                                                                                             \
                      rank,                                                                                                            \
                      cpu_rdma_team);                                                                                                  \
    }                                                                                                                                  \
    break

    constexpr int kNumThreads = 512;
    const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

    // Get clean meta
    auto rdma_clean_meta =
        get_rdma_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4,
                                             num_scales,
                                             num_topk,
                                             num_topk,
                                             num_rdma_ranks,
                                             NUM_MAX_NVL_PEERS,
                                             num_max_nvl_chunked_recv_tokens,
                                             num_channels,
                                             true);
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());

    // Launch kernel
    SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
    SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

template <bool kLowLatencyMode,
          int kNumRDMARanks,
          bool kCachedMode,
          int kNumTMABytesPerWarp,
          int kNumDispatchRDMASenderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32), 1) dispatch(
             int4* recv_x,
             float* recv_x_scales,
             topk_idx_t* recv_topk_idx,
             float* recv_topk_weights,
             SourceMeta* recv_src_meta,
             const int4* x,
             const float* x_scales,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             int* send_rdma_head,
             int* send_nvl_head,
             int* recv_rdma_channel_prefix_matrix,
             int* recv_gbl_channel_prefix_matrix,
             const int* rdma_channel_prefix_matrix,
             const int* recv_rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             const int* recv_gbl_rank_prefix_sum,
             const bool* is_token_in_rank,
             int num_tokens,
             int num_worst_tokens,
             int hidden_int4,
             int num_scales,
             int num_topk,
             int num_experts,
             int scale_token_stride,
             int scale_hidden_stride,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks) {

    enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };

    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
    const bool is_forwarder = sm_id % 2 == 0;
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;

    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms);

    const auto role_meta = [=]() -> std::pair<WarpRole, int> {
        if (is_forwarder) {  // 偶数sm_id的warp负责RDMA缓冲区到NVL缓冲区的转发
            if (warp_id < NUM_MAX_NVL_PEERS) {  // 前8个warp作为RDMA到NVL转发器
                return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
            } else {  // 偶数sm_id中的其他warp作为RDMA到NVL转发协调器
                return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};
            }
        } else if (warp_id < kNumDispatchRDMASenderWarps) {  // RDMA发送器: 发送数据到远程节点
            return {WarpRole::kRDMASender, -1};
        } else if (warp_id == kNumDispatchRDMASenderWarps) { // RDMA发送协调器: 协调RDMA传输
            return {WarpRole::kRDMASenderCoordinator, -1};
        } else {   // NVL接收器: 从NVL缓冲区接收数据
            return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS};
        }
    }();
    auto warp_role = role_meta.first;
    auto target_rank = role_meta.second;  // Not applicable for RDMA senders
    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS);

    // Data checks
    EP_DEVICE_ASSERT(num_topk <= 32);

    // RDMA symmetric layout
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto scale_bytes = num_scales * sizeof(float);
    auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk);
    auto rdma_channel_data = SymBuffer<uint8_t>(
        rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

    // NVL buffer layouts
    // NOTES: `rs_wr_buffer_ptr` means "Read for Senders, Write for Receivers", `ws_rr_buffer_ptr` means "Write for Senders, Read for
    // Receivers"
    void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
    int rs_wr_rank = 0, ws_rr_rank = 0;
    if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        rs_wr_buffer_ptr = buffer_ptrs[nvl_rank], ws_rr_buffer_ptr = buffer_ptrs[target_rank], rs_wr_rank = nvl_rank,
        ws_rr_rank = target_rank;
    if (warp_role == WarpRole::kNVLReceivers)
        rs_wr_buffer_ptr = buffer_ptrs[target_rank], ws_rr_buffer_ptr = buffer_ptrs[nvl_rank], rs_wr_rank = target_rank,
        ws_rr_rank = nvl_rank;

    // Allocate buffers
    auto nvl_channel_x = AsymBuffer<uint8_t>(ws_rr_buffer_ptr,
                                             num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                             NUM_MAX_NVL_PEERS,
                                             channel_id,
                                             num_channels,
                                             rs_wr_rank)
                             .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_start =
        AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_end = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                      .advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_head =
        AsymBuffer<int>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);
    auto nvl_channel_tail =
        AsymBuffer<int>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);

    // RDMA sender warp synchronization
    // NOTES: `rdma_send_channel_tail` means the latest released tail
    // NOTES: `rdma_send_channel_window` means the ongoing 32 transactions' status
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
    auto sync_rdma_sender_smem = []() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // TMA stuffs
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
    auto tma_buffer = smem_tma_buffer + target_rank * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
    uint32_t tma_phase = 0;
    if ((warp_role == WarpRole::kRDMAAndNVLForwarder or warp_role == WarpRole::kNVLReceivers) and elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();

    // Forward warp synchronization
    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = []() { asm volatile("barrier.sync 1, %0;" ::"r"((NUM_MAX_NVL_PEERS + 1) * 32)); };

    if (warp_role == WarpRole::kRDMASender) {
        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // Send number of tokens in this channel by `-value - 1`
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr =
                dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                dst_ptr[lane_id] =
                    -(channel_id == 0
                          ? 0
                          : gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1]) -
                    1;
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] =
                    -gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) * num_channels +
                                               channel_id] -
                    1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
            }
            __syncwarp();

            // Issue RDMA for non-local ranks
            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                  channel_id,
                                                  lane_id,
                                                  0);
            }
        }
        sync_rdma_sender_smem();

        // Iterate over tokens and copy into buffer
        int64_t token_idx;
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            // Read RDMA rank existence
            uint64_t is_token_in_rank_uint64 = 0;
            if (lane_id < kNumRDMARanks) {
                is_token_in_rank_uint64 =
                    __ldg(reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS));
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);
            }
            __syncwarp();

            // Skip the token which does not belong to this warp
            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

            // Wait the remote buffer to be released
            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));

                // Timeout check
                if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id,
                           rdma_rank,
                           nvl_rank,
                           lane_id,
                           cached_rdma_channel_head,
                           rdma_tail_idx);
                    trap();
                }
            }
            __syncwarp();

            // Store RDMA head for combine
            if (lane_id < kNumRDMARanks and not kCachedMode)
                send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

            // Broadcast tails
            SourceMeta src_meta;
            int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
            void* dst_send_buffers[kNumTopkRDMARanks];
            #pragma unroll
            for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
                if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
                    slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                    topk_ranks[num_topk_ranks] = i;
                    auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
                    auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
                    if (lane_id == num_topk_ranks)
                        src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
                    dst_send_buffers[num_topk_ranks++] =
                        reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_token;
                }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);

            // Copy `x` into symmetric send buffer
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;

            // Copy `x_scales` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_scales; i += 32) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                auto value = ld_nc_global(x_scales + offset);
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);
            }
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;

            // Copy source metadata into symmetric send buffer
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

            // Copy `topk_idx` and `topk_weights` into symmetric send buffer
            #pragma unroll
            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
                auto rank_idx = i / num_topk, copy_idx = i % num_topk;
                auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
                auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);
            }
            __syncwarp();

            // Release the transaction in the window
            if (is_token_in_rank_uint64 != 0) {
                // Acquire lock first
                acquire_lock(rdma_send_channel_lock + lane_id);
                auto latest_tail = rdma_send_channel_tail[lane_id];
                auto offset = rdma_tail_idx - latest_tail;
                while (offset >= 32) {
                    release_lock(rdma_send_channel_lock + lane_id);
                    acquire_lock(rdma_send_channel_lock + lane_id);
                    latest_tail = rdma_send_channel_tail[lane_id];
                    offset = rdma_tail_idx - latest_tail;
                }

                // Release the transaction slot
                // Add the bit and move the ones if possible
                auto window = rdma_send_channel_window[lane_id] | (1u << offset);
                if (offset == 0) {
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;
                    st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);
                    window >>= num_empty_slots;
                }
                rdma_send_channel_window[lane_id] = window;

                // Release lock
                release_lock(rdma_send_channel_lock + lane_id);
            }
            __syncwarp();
        }
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        // NOTES: in case of splitting, the issued put at the end of the buffer
        EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

        // Clean shared memory
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;

        // Synchronize shared memory
        sync_rdma_sender_smem();

        // Get number of tokens to send for each RDMA rank
        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
            if (channel_id > 0)
                num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
        }

        // Iterate all RDMA ranks
        int last_issued_tail = 0;
        auto start_time = clock64();
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                printf("DeepEP RDMA sender coordinator timeout, channel: %d, IB: %d, nvl %d, dst IB: %d, tail: %d, remaining: %d\n",
                       channel_id,
                       rdma_rank,
                       nvl_rank,
                       lane_id,
                       last_issued_tail,
                       num_tokens_to_send);
                trap();
            }

            // TODO: try thread-level `put_nbi`?
            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++i) {
                // To mitigate incast congestion, shuffle the starting index of target rank for different ranks and channels
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);
                if (synced_num_tokens_to_send == 0)
                    continue;

                // Read the latest progress
                // NOTES: `rdma_send_channel_tail` does not need to be protected by lock
                auto processed_tail =
                    __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
                    continue;

                // Issue RDMA send
                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);
                if (dst_rdma_rank != rdma_rank) {
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;
                    const auto dst_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_token);
                    const auto src_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_token);
                        // internode_ll.cu中，nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, token_idx - offset);
                        nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,
                                                      src_ptr,
                                                      num_bytes_per_msg,
                                                      translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                      channel_id,
                                                      lane_id,
                                                      0);
                } else {
                    // Lighter fence for local RDMA rank
                    memory_fence();
                }
                __syncwarp();

                // Update tails
                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send -= num_tokens_to_issue;
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
                                                    num_tokens_to_issue,
                                                    translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                    channel_id,
                                                    dst_rdma_rank == rdma_rank);
                }
                __syncwarp();
            }
        }
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        // RDMA consumers and NVL producers
        const auto dst_nvl_rank = target_rank;

        // Wait counters to arrive
        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    // Notify NVL ranks
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    // Save RDMA channel received token count
                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    if (not kCachedMode)
                        recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf(
                        "DeepEP dispatch forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst NVL: %d, "
                        "meta: %d, %d, %d, %d\n",
                        channel_id,
                        rdma_rank,
                        nvl_rank,
                        lane_id,
                        dst_nvl_rank,
                        meta_0,
                        meta_1,
                        meta_2,
                        meta_3);
                    trap();
                }
            }
        }
        __syncwarp();

        // Shift cached head
        send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

        // Wait shared memory to be cleaned
        sync_forwarder_smem();

        // Forward tokens from RDMA buffer
        // NOTES: always start from the local rank
        int src_rdma_rank = sm_id % kNumRDMARanks;
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            // Check destination queue emptiness, or wait a buffer to be released
            start_time = clock64();
            while (true) {
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens)
                    break;
                cached_nvl_channel_head = __shfl_sync(0xffffffff, ld_volatile_global(nvl_channel_head.buffer()), 0);

                // Timeout check
                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf(
                        "DeepEP dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, head: %d, tail: %d\n",
                        channel_id,
                        rdma_rank,
                        nvl_rank,
                        dst_nvl_rank,
                        ld_volatile_global(nvl_channel_head.buffer()),
                        cached_nvl_channel_tail);
                    trap();
                }
            }

            // Find next source RDMA rank (round-robin)
            start_time = clock64();
            while (true) {
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf(
                        "DeepEP dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA lane: %d, "
                        "head: %d, tail: %d, expected: %d\n",
                        channel_id,
                        rdma_rank,
                        nvl_rank,
                        dst_nvl_rank,
                        lane_id,
                        cached_rdma_channel_head,
                        cached_rdma_channel_tail,
                        num_tokens_to_recv_from_rdma);
                    trap();
                }
            }
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

            // Iterate over every token from the RDMA buffer
            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
                auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
                auto shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
                // 基于SourceMeta位图进行精确路由
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
                if (lane_id == src_rdma_rank) {
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;
                    if (not kCachedMode)
                        send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;
                }
                if (not is_in_dst_nvl_rank)  // 只处理需要接收的token
                    continue;

                // Get an empty slot
                int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;
                auto dst_shifted = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

                // Copy data
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, num_bytes_per_token, false);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, num_bytes_per_token);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);
                if (elect_one_sync())
                    tma_store_1d(tma_buffer, dst_shifted, num_bytes_per_token);
                __syncwarp();

                // In case of insufficient NVL buffers, early stopping
                if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;

                // Wait TMA to be finished
                tma_store_wait<0>();
                __syncwarp();
            }

            // Sync head index
            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

            // Move tail index
            __syncwarp();
            if (elect_one_sync())
                st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
        }

        // Retired
        __syncwarp();
        if (elect_one_sync())
            forward_channel_retired[dst_nvl_rank] = true;
    } else if (warp_role == WarpRole::kForwarderCoordinator) {
        // Extra warps for forwarder coordinator should exit directly
        if (target_rank > 0)
            return;

        // Forward warp coordinator
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        // Clean shared memory
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        #pragma unroll
        for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
            forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
        if (lane_id < NUM_MAX_NVL_PEERS)
            forward_channel_retired[lane_id] = false;
        sync_forwarder_smem();

        int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
        while (true) {
            // Find minimum head
            int min_head = std::numeric_limits<int>::max();
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                if (not forward_channel_retired[i])
                    min_head = min(min_head, forward_channel_head[i][target_rdma]);
            if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
                break;

            // Update remote head
            if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + num_max_rdma_chunked_send_tokens and
                lane_id < kNumRDMARanks) {
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                min_head - last_head,
                                                translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank),
                                                channel_id + num_channels,
                                                lane_id == rdma_rank);
                last_head = min_head;
            }

            // Nanosleep and let other warps work
            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
    } else {
        // NVL consumers
        // Retrieve rank offset from barrier results (each lane's register stores an RDMA rank)
        int src_nvl_rank = target_rank, total_offset = 0;
        const int local_expert_begin = rank * (num_experts / num_ranks);
        const int local_expert_end = local_expert_begin + (num_experts / num_ranks);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

        // Receive channel offsets
        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
            end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            if (start_offset < 0 and end_offset < 0) {
                start_offset = -start_offset - 1, end_offset = -end_offset - 1;
                total_offset += start_offset;
                break;
            }

            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf(
                    "DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d, start: %d, end: %d\n",
                    channel_id,
                    rdma_rank,
                    nvl_rank,
                    lane_id,
                    src_nvl_rank,
                    start_offset,
                    end_offset);
                trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

        // Save for combine usage
        if (lane_id < kNumRDMARanks and not kCachedMode)
            recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
        __syncwarp();

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {
            // Check channel status by lane 0
            start_time = clock64();
            while (true) {
                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                cached_channel_tail_idx = __shfl_sync(0xffffffff, ld_acquire_sys_global(nvl_channel_tail.buffer()), 0);

                // Timeout check
                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, head: %d, tail: %d\n",
                           channel_id,
                           rdma_rank,
                           nvl_rank,
                           src_nvl_rank,
                           cached_channel_head_idx,
                           cached_channel_tail_idx);
                    trap();
                }
            }

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
                int token_idx_in_buffer = (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;
                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                int64_t recv_token_idx = __shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank);
                (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;

                bool scale_aligned = (scale_bytes % 16 == 0);
                auto tma_load_bytes = hidden_bytes + (scale_aligned ? scale_bytes : 0);

                // Copy data
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, tma_load_bytes);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, tma_load_bytes);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);
                if (elect_one_sync()) {
                    tma_store_1d(tma_buffer, recv_x + recv_token_idx * hidden_int4, hidden_bytes, false);
                    if (scale_aligned)
                        tma_store_1d(tma_buffer + hidden_bytes, recv_x_scales + recv_token_idx * num_scales, scale_bytes, false);
                }
                __syncwarp();
                shifted += hidden_bytes;

                // Copy scales
                // TODO: make it as templated
                if (not scale_aligned) {
                    UNROLLED_WARP_COPY(1,
                                       lane_id,
                                       num_scales,
                                       recv_x_scales + recv_token_idx * num_scales,
                                       reinterpret_cast<float*>(shifted),
                                       ld_nc_global,
                                       st_na_global);
                }
                shifted += scale_bytes;

                // Copy source meta
                if (not kCachedMode and elect_one_sync())
                    st_na_global(recv_src_meta + recv_token_idx, meta);
                shifted += sizeof(SourceMeta);

                // Copy `topk_idx` and `topk_weights`
                if (lane_id < num_topk) {
                    // Read
                    auto idx_value = static_cast<topk_idx_t>(ld_nc_global(reinterpret_cast<int*>(shifted) + lane_id));
                    auto weight_value = ld_nc_global(reinterpret_cast<float*>(shifted + sizeof(int) * num_topk) + lane_id);
                    auto recv_idx = recv_token_idx * num_topk + lane_id;

                    // Transform and write
                    idx_value = (idx_value >= local_expert_begin and idx_value < local_expert_end) ? idx_value - local_expert_begin : -1;
                    weight_value = idx_value >= 0 ? weight_value : 0.0f;
                    st_na_global(recv_topk_idx + recv_idx, idx_value);
                    st_na_global(recv_topk_weights + recv_idx, weight_value);
                }

                // Wait TMA to be finished
                tma_store_wait<0>();
                __syncwarp();
            }

            // Move queue
            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }
    }

    // Clean unused `recv_topk_idx` as -1
    if (num_worst_tokens > 0) {
        if (is_forwarder)
            return;
        // get the actual number of num_recv_tokens on the current rank
        int num_recv_tokens = recv_gbl_rank_prefix_sum[num_ranks - 1];
        // some ForwarderCoordinator threads exit early, so we only use non-forwarder in clean-up
        // channel_id * num_threads is the offset of the current non-forwarder sms
        const auto clean_start = num_recv_tokens * num_topk + channel_id * num_threads;
        const auto clean_end = num_worst_tokens * num_topk;
        const auto clean_stride = num_channels * num_threads;
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
            recv_topk_idx[i] = -1;
    }
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              int* send_rdma_head,
              int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_scales,
              int num_topk,
              int num_experts,
              int scale_token_stride,
              int scale_hidden_stride,
              void* rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens,
              int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens,
              int rank,
              int num_ranks,
              bool is_cached_dispatch,
              cudaStream_t stream,
              int num_channels,
              bool low_latency_mode) {
    constexpr int kNumDispatchRDMASenderWarps = 7;
    constexpr int kNumTMABytesPerWarp = 16384;
    constexpr int smem_size = kNumTMABytesPerWarp * NUM_MAX_NVL_PEERS;

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                                   \
    {                                                                                                                          \
        auto dispatch_func = low_latency_mode                                                                                  \
            ? (is_cached_dispatch ? dispatch<true, num_rdma_ranks, true, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>     \
                                  : dispatch<true, num_rdma_ranks, false, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>)   \
            : (is_cached_dispatch ? dispatch<false, num_rdma_ranks, true, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>    \
                                  : dispatch<false, num_rdma_ranks, false, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>); \
        SET_SHARED_MEMORY_FOR_TMA(dispatch_func);                                                                              \
        LAUNCH_KERNEL(&cfg,                                                                                                    \
                      dispatch_func,                                                                                           \
                      reinterpret_cast<int4*>(recv_x),                                                                         \
                      recv_x_scales,                                                                                           \
                      recv_topk_idx,                                                                                           \
                      recv_topk_weights,                                                                                       \
                      reinterpret_cast<SourceMeta*>(recv_src_meta),                                                            \
                      reinterpret_cast<const int4*>(x),                                                                        \
                      x_scales,                                                                                                \
                      topk_idx,                                                                                                \
                      topk_weights,                                                                                            \
                      send_rdma_head,                                                                                          \
                      send_nvl_head,                                                                                           \
                      recv_rdma_channel_prefix_matrix,                                                                         \
                      recv_gbl_channel_prefix_matrix,                                                                          \
                      rdma_channel_prefix_matrix,                                                                              \
                      recv_rdma_rank_prefix_sum,                                                                               \
                      gbl_channel_prefix_matrix,                                                                               \
                      recv_gbl_rank_prefix_sum,                                                                                \
                      is_token_in_rank,                                                                                        \
                      num_tokens,                                                                                              \
                      num_worst_tokens,                                                                                        \
                      hidden_int4,                                                                                             \
                      num_scales,                                                                                              \
                      num_topk,                                                                                                \
                      num_experts,                                                                                             \
                      scale_token_stride,                                                                                      \
                      scale_hidden_stride,                                                                                     \
                      rdma_buffer_ptr,                                                                                         \
                      num_max_rdma_chunked_send_tokens,                                                                        \
                      num_max_rdma_chunked_recv_tokens,                                                                        \
                      buffer_ptrs,                                                                                             \
                      num_max_nvl_chunked_send_tokens,                                                                         \
                      num_max_nvl_chunked_recv_tokens,                                                                         \
                      rank,                                                                                                    \
                      num_ranks);                                                                                              \
    }                                                                                                                          \
    break

    EP_HOST_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32, stream);
    SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <bool kLowLatencyMode, int kNumTMABytesPerWarp>
__global__ void cached_notify(const int rdma_clean_offset,
                              const int rdma_num_int_clean,
                              const int nvl_clean_offset,
                              const int nvl_num_int_clean,
                              int* combined_rdma_head,
                              int num_combined_tokens,
                              int num_channels,
                              const int* rdma_channel_prefix_matrix,
                              const int* rdma_rank_prefix_sum,
                              int* combined_nvl_head,
                              void* rdma_buffer_ptr,
                              void** buffer_ptrs,
                              int** barrier_signal_ptrs,
                              int rank,
                              int num_ranks,
                              bool is_cached_dispatch,
                              const nvshmem_team_t rdma_team) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);
    auto num_warps = num_threads / 32;
    auto warp_id = thread_id / 32;
    auto lane_id = get_lane_id();

    auto nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    auto rdma_rank = rank / NUM_MAX_NVL_PEERS;

    // Using two SMs, which clean the RDMA/NVL buffer respectively
    if (sm_id == 0) {
        auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
        for (int i = thread_id; i < qps_per_rdma_rank * (num_rdma_ranks - 1); i += num_threads) {
            auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % num_rdma_ranks;
            auto qp_id = i % qps_per_rdma_rank;
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), qp_id);
        }
        __syncthreads();

        // Barrier for RDMA
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);

        // Barrier for NVL
        barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

        // Clean RDMA buffer
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        #pragma unroll
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

        // Clean NVL buffer
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;
        __syncthreads();

        // Barrier again
        if (thread_id == 32)
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else if (sm_id == 1) {
        if (is_cached_dispatch)
            return;

        EP_DEVICE_ASSERT(num_warps >= num_channels);
        EP_DEVICE_ASSERT(num_rdma_ranks <= 32);

        // Iterate in reverse order
        if (lane_id < num_rdma_ranks and warp_id < num_channels) {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, warp_id, token_start_idx, token_end_idx);

            // NOTES: `1 << 25` is a heuristic large number
            int last_head = 1 << 25;
            for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                auto current_head = __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id);
                if (current_head < 0) {
                    combined_rdma_head[token_idx * num_rdma_ranks + lane_id] = -last_head - 1;
                } else {
                    last_head = current_head;
                }
            }
        }
    } else {
        if (is_cached_dispatch)
            return;

        EP_DEVICE_ASSERT(num_warps >= num_channels);
        EP_DEVICE_ASSERT(rdma_channel_prefix_matrix != nullptr and rdma_rank_prefix_sum != nullptr);
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Too many NVL peers");

        if (warp_id < num_channels) {
            constexpr int tma_batch_size = kNumTMABytesPerWarp - sizeof(uint64_t);
            constexpr int num_bytes_per_token = sizeof(int) * NUM_MAX_NVL_PEERS;
            constexpr int num_tokens_per_batch = tma_batch_size / num_bytes_per_token;
            EP_STATIC_ASSERT(num_bytes_per_token % 16 == 0, "num_bytes_per_token should be divisible by 16");

            // TMA stuffs
            extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
            auto tma_buffer = smem_tma_buffer + warp_id * kNumTMABytesPerWarp;
            auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + tma_batch_size);
            uint32_t tma_phase = 0;
            if (elect_one_sync()) {
                mbarrier_init(tma_mbarrier, 1);
                fence_barrier_init();
            }
            __syncwarp();

            for (int dst_rdma_rank = sm_id - 2; dst_rdma_rank < num_rdma_ranks; dst_rdma_rank += num_channels * 2 - 2) {
                // Iterate in reverse order
                int token_start_idx = warp_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id - 1];
                int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id];
                int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                token_start_idx += shift, token_end_idx += shift;

                // NOTES: `1 << 25` is a heuristic large number
                int last_head = 1 << 25;
                for (int batch_end_idx = token_end_idx; batch_end_idx > token_start_idx; batch_end_idx -= num_tokens_per_batch) {
                    auto batch_start_idx = max(token_start_idx, batch_end_idx - num_tokens_per_batch);

                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer,
                                    combined_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS,
                                    tma_mbarrier,
                                    (batch_end_idx - batch_start_idx) * num_bytes_per_token);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, (batch_end_idx - batch_start_idx) * num_bytes_per_token);
                    }
                    mbarrier_wait(tma_mbarrier, tma_phase);
                    __syncwarp();

                    for (int token_idx = batch_end_idx - 1; token_idx >= batch_start_idx; --token_idx) {
                        if (lane_id < NUM_MAX_NVL_PEERS) {
                            auto current_head =
                                reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id];
                            if (current_head < 0) {
                                reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id] =
                                    -last_head - 1;
                            } else {
                                last_head = current_head;
                            }
                        }
                    }
                    tma_store_fence();
                    __syncwarp();

                    if (elect_one_sync())
                        tma_store_1d(tma_buffer,
                                     combined_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS,
                                     (batch_end_idx - batch_start_idx) * num_bytes_per_token);
                    tma_store_wait<0>();
                    __syncwarp();
                }
            }
        }
    }
}

void cached_notify(int hidden_int4,
                   int num_scales,
                   int num_topk_idx,
                   int num_topk_weights,
                   int num_ranks,
                   int num_channels,
                   int num_combined_tokens,
                   int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix,
                   const int* rdma_rank_prefix_sum,
                   int* combined_nvl_head,
                   void* rdma_buffer_ptr,
                   int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs,
                   int rank,
                   cudaStream_t stream,
                   int64_t num_rdma_bytes,
                   int64_t num_nvl_bytes,
                   bool is_cached_dispatch,
                   bool low_latency_mode) {
    const int num_threads = std::max(128, 32 * num_channels);
    const int num_warps = num_threads / 32;
    const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    const int kNumTMABytesPerWarp = 8192;
    const int smem_size = kNumTMABytesPerWarp * num_warps;

    // Get clean meta
    auto rdma_clean_meta = get_rdma_clean_meta(
        hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4,
                                             num_scales,
                                             num_topk_idx,
                                             num_topk_weights,
                                             num_rdma_ranks,
                                             NUM_MAX_NVL_PEERS,
                                             num_max_nvl_chunked_recv_tokens,
                                             num_channels,
                                             is_cached_dispatch);
    EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_channels * 2 > 3);

    // Launch kernel
    auto cached_notify_func = low_latency_mode ? cached_notify<true, kNumTMABytesPerWarp> : cached_notify<false, kNumTMABytesPerWarp>;
    SETUP_LAUNCH_CONFIG(num_channels * 2, num_threads, stream);
    SET_SHARED_MEMORY_FOR_TMA(cached_notify_func);
    LAUNCH_KERNEL(&cfg,
                  cached_notify_func,
                  rdma_clean_meta.first,
                  rdma_clean_meta.second,
                  nvl_clean_meta.first,
                  nvl_clean_meta.second,
                  combined_rdma_head,
                  num_combined_tokens,
                  num_channels,
                  rdma_channel_prefix_matrix,
                  rdma_rank_prefix_sum,
                  combined_nvl_head,
                  rdma_buffer_ptr,
                  buffer_ptrs,
                  barrier_signal_ptrs,
                  rank,
                  num_ranks,
                  is_cached_dispatch,
                  cpu_rdma_team);
}

template <int kNumRanks,
          bool kMaybeWithBias,
          typename dtype_t,
          int kMaxNumRanks,
          bool kUseTMA,
          int kNumStages,
          int kNumTMALoadBytes = 0,
          typename GetAddrFn,
          typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank,
                             int head_idx,
                             int lane_id,
                             int hidden_int4,
                             int num_topk,
                             int4* combined_row,
                             float* combined_topk_weights,
                             const int4* bias_0_int4,
                             const int4* bias_1_int4,
                             int num_max_recv_tokens,
                             const GetAddrFn& get_addr_fn,
                             const ReceiveTWFn& recv_tw_fn,
                             uint8_t* smem_ptr,
                             uint32_t (&tma_phase)[kNumStages]) {
    constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

    // Broadcast current heads
    // Lane `i` holds the head of rank `i` and `is_token_in_rank`
    EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
    int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
    #pragma unroll
    for (int i = 0; i < kNumRanks; ++i)
        if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
            slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
            topk_ranks[num_topk_ranks++] = i;
        }
    EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);
    EP_STATIC_ASSERT(not(kUseTMA and kMaybeWithBias), "TMA cannot be used by receiver warps");
    EP_STATIC_ASSERT(kNumStages == 2, "Only support 2 stages now");

    // Reduce data
    if constexpr (kUseTMA) {
        constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
        EP_DEVICE_ASSERT(hidden_int4 % 32 == 0);

        auto tma_load_buffer = [=](const int& i, const int& j) -> int4* {
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + j * kNumTMALoadBytes);
        };
        auto tma_store_buffer = [=](const int& i) -> int4* {
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + NUM_MAX_NVL_PEERS * kNumTMALoadBytes);
        };
        auto tma_mbarrier = [=](const int& i) -> uint64_t* {
            return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + (NUM_MAX_NVL_PEERS + 1) * kNumTMALoadBytes);
        };

        // Prefetch
        if (lane_id < num_topk_ranks)
            tma_load_1d(
                tma_load_buffer(0, lane_id), get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], 0), tma_mbarrier(0), kNumTMALoadBytes);
        mbarrier_arrive_and_expect_tx(tma_mbarrier(0), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
        __syncwarp();

        for (int shifted = 0, iter = 0; shifted < hidden_int4; shifted += 32, iter += 1) {
            const int stage_idx = iter % kNumStages;
            const int next_stage_idx = (iter + 1) % kNumStages;

            // Prefetch next stage
            if (shifted + 32 < hidden_int4) {
                if (lane_id < num_topk_ranks)
                    tma_load_1d(tma_load_buffer(next_stage_idx, lane_id),
                                get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], shifted + 32),
                                tma_mbarrier(next_stage_idx),
                                kNumTMALoadBytes);
                mbarrier_arrive_and_expect_tx(tma_mbarrier(next_stage_idx), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
                __syncwarp();
            }

            mbarrier_wait(tma_mbarrier(stage_idx), tma_phase[stage_idx]);
            float values[kDtypePerInt4] = {0};
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j) {
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(tma_load_buffer(stage_idx, j) + lane_id);
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    values[k] += static_cast<float>(recv_value_dtypes[k]);
            }

            // Wait shared memory to be released
            tma_store_wait<kNumStages - 1>();

            // Copy into shared and issue TMA
            auto out_dtypes = reinterpret_cast<dtype_t*>(tma_store_buffer(stage_idx) + lane_id);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                out_dtypes[j] = static_cast<dtype_t>(values[j]);
            tma_store_fence();
            __syncwarp();

            if (elect_one_sync())
                tma_store_1d(tma_store_buffer(stage_idx), combined_row + shifted, kNumTMALoadBytes);
            __syncwarp();
        }

        // Flush all writes
        tma_store_wait<0>();
    } else {
        #pragma unroll
        for (int i = lane_id; i < hidden_int4; i += 32) {
            // Read bias
            // TODO: make it as a finer-grained template
            int4 bias_0_value_int4, bias_1_value_int4;
            if constexpr (kMaybeWithBias) {
                bias_0_value_int4 = bias_0_int4 != nullptr ? ld_nc_global(bias_0_int4 + i) : make_int4(0, 0, 0, 0);
                bias_1_value_int4 = bias_1_int4 != nullptr ? ld_nc_global(bias_1_int4 + i) : make_int4(0, 0, 0, 0);
            }

            // Read buffers
            // TODO: maybe too many registers here
            int4 recv_value_int4[kMaxNumRanks];
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j)
                recv_value_int4[j] = ld_nc_global(get_addr_fn(topk_ranks[j], slot_indices[j], i));

            // Clean
            // Reduce bias
            float values[kDtypePerInt4] = {0};
            if constexpr (kMaybeWithBias) {
                auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
                auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
                #pragma unroll
                for (int j = 0; j < kDtypePerInt4; ++j)
                    values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);
            }

            // Reduce all-to-all results
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j) {
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    values[k] += static_cast<float>(recv_value_dtypes[k]);
            }

            // Cast back to `dtype_t` and write
            int4 out_int4;
            auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                out_dtypes[j] = static_cast<dtype_t>(values[j]);
            st_na_global(combined_row + i, out_int4);
        }
    }

    // Reduce `topk_weights`
    if (lane_id < num_topk) {
        float value = 0;
        #pragma unroll
        for (int i = 0; i < num_topk_ranks; ++i)
            value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
        st_na_global(combined_topk_weights + lane_id, value);
    }

    // Return the minimum top-k rank
    return topk_ranks[0];
}

template <bool kLowLatencyMode,
          int kNumRDMARanks,
          typename dtype_t,
          int kNumCombineForwarderWarps,
          int kNumTMABytesPerSenderWarp,
          int kNumTMABytesPerForwarderWarp,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
          int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1,
          int kNumForwarders = kNumRDMARanks* kNumWarpsPerForwarder,
          int kNumRDMAReceivers = kNumForwarders - NUM_MAX_NVL_PEERS>
__global__ void __launch_bounds__((kNumForwarders + 1) * 32, 1) combine(int4* combined_x,
                                                                        float* combined_topk_weights,
                                                                        const bool* is_combined_token_in_rank,
                                                                        const int4* x,
                                                                        const float* topk_weights,
                                                                        const int4* bias_0,
                                                                        const int4* bias_1,
                                                                        const int* combined_rdma_head,
                                                                        const int* combined_nvl_head,
                                                                        const SourceMeta* src_meta,
                                                                        const int* rdma_channel_prefix_matrix,
                                                                        const int* rdma_rank_prefix_sum,
                                                                        const int* gbl_channel_prefix_matrix,
                                                                        int num_tokens,
                                                                        int num_combined_tokens,
                                                                        int hidden,
                                                                        int num_topk,
                                                                        void* rdma_buffer_ptr,
                                                                        int num_max_rdma_chunked_send_tokens,
                                                                        int num_max_rdma_chunked_recv_tokens,
                                                                        void** buffer_ptrs,
                                                                        int num_max_nvl_chunked_send_tokens,
                                                                        int num_max_nvl_chunked_recv_tokens,
                                                                        int rank,
                                                                        int num_ranks) {
    enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
    const bool is_forwarder_sm = sm_id % 2 == 1;

    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));
    const auto hidden_bytes = hidden_int4 * sizeof(int4);
    const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);

    // NOTES: we decouple a channel into 2 SMs
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        auto warp_id = thread_id / 32;
        if (not is_forwarder_sm) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                auto shuffled_warp_id = warp_id;
                shuffled_warp_id = (shuffled_warp_id + channel_id) % NUM_MAX_NVL_PEERS;
                return {WarpRole::kNVLSender, shuffled_warp_id};
            } else if (warp_id < kNumForwarders) {
                return {WarpRole::kRDMAReceiver, warp_id - NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        } else {
            if (warp_id < kNumForwarders) {
                auto shuffled_warp_id = (warp_id + channel_id) % kNumForwarders;
                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        }
    }();
    auto warp_role = role_meta.first;
    auto warp_id = role_meta.second;

    EP_DEVICE_ASSERT(num_warps == kNumForwarders + 1);
    auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;

    if (warp_role == WarpRole::kNVLSender) {
        // NVL producers
        const auto dst_nvl_rank = warp_id;

        // NVL layouts
        // NOTES: to avoid deadlocks, we use separate NVL buffers for different RDMA sources
        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
        auto nvl_channel_x = AsymBuffer<uint8_t>(dst_buffer_ptr,
                                                 num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                                 NUM_MAX_NVL_PEERS,
                                                 channel_id,
                                                 num_channels,
                                                 nvl_rank)
                                 .advance_also(local_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, dst_nvl_rank)
                                    .advance_also(dst_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                                    .advance_also(local_buffer_ptr);

        // TMA stuffs
        extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
        auto tma_buffer = smem_tma_buffer + dst_nvl_rank * kNumTMABytesPerSenderWarp;
        auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
        uint32_t tma_phase = 0;
        if (elect_one_sync()) {
            mbarrier_init(tma_mbarrier, 1);
            fence_barrier_init();
            EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerSenderWarp);
        }
        __syncwarp();

        // Get tasks for each RDMA lane
        int token_start_idx = 0, token_end_idx = 0;
        if (lane_id < kNumRDMARanks) {
            int prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;
            token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
            token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1];
        }
        __syncwarp();

        // NOTES: here the cached value of each lane is only responsible for a single RDMA buffer
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        // Iterate over all tokens and send by chunks
        int current_rdma_idx = channel_id % kNumRDMARanks;
        while (true) {
            // Exit if possible
            if (__all_sync(0xffffffff, token_start_idx >= token_end_idx))
                break;

            // Decide the next RDMA buffer to send
            bool is_lane_ready = false;
            auto start_time = clock64();
            while (true) {
                int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and
                    num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
                if (__any_sync(0xffffffff, is_lane_ready))
                    break;

                // Retry
                if (lane_id < kNumRDMARanks and token_start_idx < token_end_idx)
                    cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id);

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf(
                        "DeepEP combine NVL sender timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, RDMA lane: %d, head: %d, tail: "
                        "%d, start: %d, end: %d\n",
                        channel_id,
                        rdma_rank,
                        nvl_rank,
                        dst_nvl_rank,
                        lane_id,
                        ld_volatile_global(nvl_channel_head.buffer() + lane_id),
                        cached_channel_tail_idx,
                        token_start_idx,
                        token_end_idx);
                    trap();
                }
            }

            // Sync token start index and count
            for (int i = 0; i < kNumRDMARanks; ++i) {
                current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;

                // Sync token start index
                auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));
                int num_tokens_in_chunk =
                    __shfl_sync(0xffffffff, min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx);

                // Send by chunk
                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                    // Get an empty slot
                    int dst_slot_idx = 0;
                    if (lane_id == current_rdma_idx) {
                        dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
                    }
                    dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

                    // Load data
                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;
                    auto shifted_x = x + token_idx * hidden_int4;
                    tma_store_wait<0>();
                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer, shifted_x, tma_mbarrier, hidden_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);
                    }
                    __syncwarp();
                    mbarrier_wait(tma_mbarrier, tma_phase);

                    // Load source meta
                    if (lane_id == num_topk)
                        *reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes) = ld_nc_global(src_meta + token_idx);

                    // Load `topk_weights`
                    if (lane_id < num_topk)
                        *reinterpret_cast<float*>(tma_buffer + hidden_bytes + sizeof(SourceMeta) + lane_id * sizeof(float)) =
                            ld_nc_global(topk_weights + token_idx * num_topk + lane_id);

                    // Issue TMA store
                    tma_store_fence();
                    __syncwarp();
                    if (elect_one_sync())
                        tma_store_1d(tma_buffer, shifted_x_buffers, num_bytes_per_token, false);
                }
                lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
            }

            // Move queue tail
            tma_store_wait<0>();
            __syncwarp();
            if (lane_id < kNumRDMARanks and is_lane_ready)
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
        }
    } else {
        // Combiners and coordinators
        // RDMA symmetric layout
        auto rdma_channel_data = SymBuffer<int8_t>(
            rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

        // NVL layouts
        void* local_nvl_buffer = buffer_ptrs[nvl_rank];
        void* nvl_buffers[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
            nvl_buffers[i] = buffer_ptrs[i];
        auto nvl_channel_x =
            AsymBuffer<uint8_t>(
                local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, channel_id, num_channels)
                .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_head =
            AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                .advance_also(local_nvl_buffer);
        auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels)
                                    .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

        // Combiner warp synchronization
        __shared__ volatile int forwarder_nvl_head[kNumForwarders][NUM_MAX_NVL_PEERS];
        __shared__ volatile bool forwarder_retired[kNumForwarders];
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
        auto sync_forwarder_smem = [=]() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumForwarders + 1) * 32)); };
        auto sync_rdma_receiver_smem = [=]() { asm volatile("barrier.sync 1, %0;" ::"r"((kNumRDMAReceivers + 1) * 32)); };

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
            // Receive from NVL ranks and forward to RDMA ranks
            // NOTES: this part is using "large warps" for each RDMA ranks
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
            auto send_buffer =
                dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            auto sync_large_warp = [=]() {
                if (kNumWarpsPerForwarder == 1) {
                    __syncwarp();
                } else {
                    asm volatile("bar.sync %0, %1;" ::"r"(dst_rdma_rank + 2), "r"(kNumWarpsPerForwarder * 32));
                }
            };
            EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough");

            // TMA stuffs
            constexpr int kNumStages = 2;
            constexpr int kNumTMALoadBytes = sizeof(int4) * 32;
            constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
            EP_STATIC_ASSERT(kNumTMABufferBytesPerStage * kNumStages <= kNumTMABytesPerForwarderWarp, "TMA buffer is not larger enough");

            extern __shared__ __align__(1024) uint8_t smem_buffer[];
            auto smem_ptr = smem_buffer + warp_id * kNumStages * kNumTMABufferBytesPerStage;
            auto tma_mbarrier = [=](const int& i) {
                return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1));
            };
            uint32_t tma_phase[kNumStages] = {0};
            if (lane_id < kNumStages) {
                mbarrier_init(tma_mbarrier(lane_id), 32);
                fence_barrier_init();
            }
            __syncwarp();

            // Advance to the corresponding NVL buffer
            nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token);
            nvl_channel_head.advance(dst_rdma_rank);
            nvl_channel_tail.advance(dst_rdma_rank);

            // Clean shared memory and sync
            EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
            lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
            sync_forwarder_smem();

            // Get count and cached head
            int cached_nvl_channel_tail_idx = 0;
            int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            num_tokens_to_combine -= num_tokens_prefix;
            num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS;

            // Iterate over all tokens and combine by chunks
            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                // Check destination queue emptiness, or wait a buffer to be released
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                auto start_time = clock64();
                while (sub_warp_id == 0 and lane_id == 0) {
                    // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
                    // Here, `token_start_idx` is the actual tail
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
                        break;

                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf(
                            "DeepEP combine forwarder (RDMA check) timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, head: %ld, tail: "
                            "%d, chunked: %d\n",
                            channel_id,
                            rdma_rank,
                            nvl_rank,
                            dst_rdma_rank,
                            ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)),
                            token_start_idx,
                            num_chunked_tokens);
                        trap();
                    }
                }
                sync_large_warp();

                // Combine and write to the RDMA buffer
                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
                    // Read expected head
                    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS) {
                        expected_head = ld_nc_global(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head);
                    }

                    // Wait lanes to be ready
                    start_time = clock64();
                    while (cached_nvl_channel_tail_idx <= expected_head) {
                        cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

                        // Timeout check
                        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS) {
                            printf(
                                "DeepEP combine forwarder (NVL check) timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, dst RDMA: %d, "
                                "tail: %d, waiting: %d, total: %d, sub: %d, large: %d, expected: %d\n",
                                channel_id,
                                rdma_rank,
                                nvl_rank,
                                lane_id,
                                dst_rdma_rank,
                                cached_nvl_channel_tail_idx,
                                token_idx,
                                num_tokens_to_combine,
                                sub_warp_id,
                                kNumWarpsPerForwarder,
                                expected_head);
                            trap();
                        }
                    }

                    // Combine current token
                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_token;
                    auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                        return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) +
                            hidden_int4_idx;
                    };
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
                        return ld_nc_global(reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token +
                                                                     hidden_bytes + sizeof(SourceMeta)) +
                                            topk_idx);
                    };
                    combine_token<NUM_MAX_NVL_PEERS, false, dtype_t, NUM_MAX_NVL_PEERS, true, kNumStages, kNumTMALoadBytes>(
                        expected_head >= 0,
                        expected_head,
                        lane_id,
                        hidden_int4,
                        num_topk,
                        static_cast<int4*>(shifted),
                        reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
                        nullptr,
                        nullptr,
                        num_max_nvl_chunked_recv_tokens_per_rdma,
                        get_addr_fn,
                        recv_tw_fn,
                        smem_ptr,
                        tma_phase);

                    // Update head
                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
                }
                sync_large_warp();

                // Issue RDMA send
                if (sub_warp_id == kNumWarpsPerForwarder - 1) {
                    if (dst_rdma_rank != rdma_rank) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;
                        const auto dst_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        const auto src_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,
                                                          src_ptr,
                                                          num_bytes_per_msg,
                                                          translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                          channel_id,
                                                          lane_id,
                                                          0);
                    } else {
                        memory_fence();
                    }

                    // Write new RDMA tail
                    __syncwarp();
                    if (elect_one_sync()) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
                                                        num_chunked_tokens,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                        channel_id,
                                                        dst_rdma_rank == rdma_rank);
                    }
                }
            }

            // Retired
            __syncwarp();
            if (elect_one_sync())
                forwarder_retired[warp_id] = true;
        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // Receive from RDMA ranks and write to the output tensor
            // Clean shared memory and sync
            EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
            lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_smem();

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
                // Read expected head
                EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                int expected_head = -1;
                if (lane_id < kNumRDMARanks) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1)
                                        : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
                }

                // Wait lanes to be ready
                auto start_time = clock64();
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf(
                            "DeepEP combine RDMA receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, tail: %d, waiting: %ld, "
                            "expect: %d\n",
                            channel_id,
                            rdma_rank,
                            nvl_rank,
                            lane_id,
                            cached_channel_tail_idx,
                            token_idx,
                            expected_head);
                        trap();
                    }
                }
                __syncwarp();

                // Combine current token
                auto get_addr_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                    return reinterpret_cast<int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token) +
                        hidden_int4_idx;
                };
                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
                    return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) +
                                                                       slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) +
                                        topk_idx);
                };
                uint32_t dummy_tma_phases[2];
                combine_token<kNumRDMARanks, true, dtype_t, kNumTopkRDMARanks, false, 2>(
                    expected_head >= 0,
                    expected_head,
                    lane_id,
                    hidden_int4,
                    num_topk,
                    combined_x + token_idx * hidden_int4,
                    combined_topk_weights + token_idx * num_topk,
                    bias_0 == nullptr ? nullptr : bias_0 + token_idx * hidden_int4,
                    bias_1 == nullptr ? nullptr : bias_1 + token_idx * hidden_int4,
                    num_max_rdma_chunked_recv_tokens,
                    get_addr_fn,
                    recv_tw_fn,
                    nullptr,
                    dummy_tma_phases);
            }

            // Retired
            __syncwarp();
            if (elect_one_sync())
                rdma_receiver_retired[warp_id] = true;
        } else {
            // Coordinator
            // Sync shared memory status
            is_forwarder_sm ? sync_forwarder_smem() : sync_rdma_receiver_smem();
            const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

            int last_rdma_head = 0;
            int last_nvl_head[kNumRDMARanks] = {0};
            int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
            int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;
            EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps");
            while (true) {
                // Retired
                if (not is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
                    break;
                if (is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumForwarders or forwarder_retired[lane_id]))
                    break;

                // Find minimum head for RDMA ranks
                if (not is_forwarder_sm) {
                    int min_head = std::numeric_limits<int>::max();
                    #pragma unroll
                    for (int i = 0; i < kNumRDMAReceivers; ++i)
                        if (not rdma_receiver_retired[i])
                            min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
                    if (min_head != std::numeric_limits<int>::max() and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and
                        lane_id < kNumRDMARanks) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                        min_head - last_rdma_head,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                        channel_id + num_channels,
                                                        dst_rdma_rank == rdma_rank);
                        last_rdma_head = min_head;
                    }
                } else {
                    // Find minimum head for NVL ranks
                    #pragma unroll
                    for (int i = 0; i < kNumRDMARanks; ++i) {
                        int min_head = std::numeric_limits<int>::max();
                        #pragma unroll
                        for (int j = 0; j < num_warps_per_rdma_rank; ++j)
                            if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                                min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);
                        if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and lane_id < NUM_MAX_NVL_PEERS)
                            st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);
                    }
                }

                // Nanosleep and let other warps work
                __nanosleep(NUM_WAIT_NANOSECONDS);
            }
        }
    }
}

void combine(cudaDataType_t type,
             void* combined_x,
             float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
             const void* src_meta,
             const int* rdma_channel_prefix_matrix,
             const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_channels,
             bool low_latency_mode) {
    constexpr int kNumCombineForwarderWarps = 24;
    constexpr int kNumTMABytesPerSenderWarp = 16384;
    constexpr int kNumTMABytesPerForwarderWarp = 9248;
    constexpr int smem_size =
        std::max(kNumTMABytesPerSenderWarp * NUM_MAX_NVL_PEERS, kNumTMABytesPerForwarderWarp * kNumCombineForwarderWarps);

#define COMBINE_LAUNCH_CASE(num_rdma_ranks)                                           \
    {                                                                                 \
        auto combine_func = low_latency_mode ? combine<true,                          \
                                                       num_rdma_ranks,                \
                                                       nv_bfloat16,                   \
                                                       kNumCombineForwarderWarps,     \
                                                       kNumTMABytesPerSenderWarp,     \
                                                       kNumTMABytesPerForwarderWarp>  \
                                             : combine<false,                         \
                                                       num_rdma_ranks,                \
                                                       nv_bfloat16,                   \
                                                       kNumCombineForwarderWarps,     \
                                                       kNumTMABytesPerSenderWarp,     \
                                                       kNumTMABytesPerForwarderWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(combine_func);                                      \
        LAUNCH_KERNEL(&cfg,                                                           \
                      combine_func,                                                   \
                      reinterpret_cast<int4*>(combined_x),                            \
                      combined_topk_weights,                                          \
                      is_combined_token_in_rank,                                      \
                      reinterpret_cast<const int4*>(x),                               \
                      topk_weights,                                                   \
                      reinterpret_cast<const int4*>(bias_0),                          \
                      reinterpret_cast<const int4*>(bias_1),                          \
                      combined_rdma_head,                                             \
                      combined_nvl_head,                                              \
                      reinterpret_cast<const SourceMeta*>(src_meta),                  \
                      rdma_channel_prefix_matrix,                                     \
                      rdma_rank_prefix_sum,                                           \
                      gbl_channel_prefix_matrix,                                      \
                      num_tokens,                                                     \
                      num_combined_tokens,                                            \
                      hidden,                                                         \
                      num_topk,                                                       \
                      rdma_buffer_ptr,                                                \
                      num_max_rdma_chunked_send_tokens,                               \
                      num_max_rdma_chunked_recv_tokens,                               \
                      buffer_ptrs,                                                    \
                      num_max_nvl_chunked_send_tokens,                                \
                      num_max_nvl_chunked_recv_tokens,                                \
                      rank,                                                           \
                      num_ranks);                                                     \
    }                                                                                 \
    break

    int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
    int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
    EP_HOST_ASSERT(num_rdma_ranks <= kNumCombineForwarderWarps);
    EP_HOST_ASSERT(num_forwarder_warps > NUM_MAX_NVL_PEERS and num_forwarder_warps % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks >
                   std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
    EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks - num_warps_per_forwarder >= num_max_nvl_chunked_send_tokens);
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder);
    EP_HOST_ASSERT(type == CUDA_R_16BF);

    SETUP_LAUNCH_CONFIG(num_channels * 2, (num_forwarder_warps + 1) * 32, stream);
    SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

}  // namespace internode

}  // namespace deep_ep
