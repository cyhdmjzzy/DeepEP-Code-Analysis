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
    /* 
    src_rdma_rank: 表示token token_idx是从哪个节点上发出的。这是用于在dispatch阶段token发送到dispatch接收端之后，在combine阶段确定这个token要原路返回发往哪个节点。
    is_token_in_nvl_rank_bits: 表示token token_idx是否要发送到的目标节点上的 8 个local_rank的布尔值。
        每个节点有8个rank，这里用32比特的is_token_in_nvl_rank_bits有点浪费（8比特足矣）。但是ibgda传输时数据要对齐到int4单位，所以这里也不得不浪费掉一些比特位。
    */
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    // TODO: faster encoding
    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];  // 初始化时要么是0，要么是1。但是除了最右边比特位以外其余比特位都是0。
        #pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i)
            // 先左移is_token_in_nvl_ranks[i]，右边用0填充，然后和is_token_in_nvl_rank_bits的对应位做按位与。
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const { return (is_token_in_nvl_rank_bits >> nvl_rank) & 1; }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

int get_source_meta_bytes() {
    return sizeof(SourceMeta);
}

__host__ __device__ __forceinline__ int get_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align_up(hidden_int4 * sizeof(int4) + num_scales * sizeof(float) + sizeof(SourceMeta) +
                                     num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float), sizeof(int4)));
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
    get_num_bytes_per_token(...): 每个 token 的字节数。包含：hidden data + scales + SourceMeta + topk_idx + topk_weights。对齐到 16 字节边界
    num_rdma_recv_buffer_tokens: 每个节点为每个 channel 准备的接收缓冲区 token 数量。
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

    // return {
    //     (num_nvl_recv_buffer_tokens * get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) 
    //     * num_nvl_ranks * num_channels) / sizeof(int),
    //     num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels,
    // };

    /*
    nvl_clean_offset 是数据区域和元数据区域的分界点，表示数据区域的结束位置（以int为单位）。
    num_nvl_recv_buffer_tokens: 每个NVLink接收缓冲区的token数量
    get_num_bytes_per_token(...): 每个token的字节数
    num_nvl_ranks: 节点内GPU数量（通常是8）
    num_channels: channel数量
    结果: 数据区域的总大小（以int为单位）
    */
    int nvl_clean_offset = (num_nvl_recv_buffer_tokens * get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) 
                        * num_nvl_ranks * num_channels) / sizeof(int);
    
    /*
    需要清零的元数据区域的总大小（以int为单位）
    */
    int nvl_num_int_clean = num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels;

    return {nvl_clean_offset, nvl_num_int_clean};
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    /* 
    实际上，不管是不是kLowLatencyMode，最终还都是不同节点的相同nvl_rank的GPU到GPU之间的通信。
    ll模式就是不同节点间只在相同nvl_rank的GPU之间进行通信。
    而普通模式下是以每个节点为PE，直接节点对节点通信，但是根据PE之间的对称内存性质，发送端节点和接收端节点通信依然是在对应的nvl_rank上通信的。
        也就是相当于整个节点的所有GPU的参与同步内存的内存区域构成该节点的对称内存空间。在runtime.cu 的 init()中创建了该对称内存空间。
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
    moe_recv_rdma_counter_mapped:   1                                   要写入的。记录当前rank收到的来自所有节点要发送给当前rank的token数量之和
    num_tokens_per_expert:          [kNumRanks, num_experts_per_rank],  要读取的。num_tokens_per_expert[i * num_experts_per_rank + j] 表示rank rank要发送到rank i内的expert j的token数量。
    moe_recv_expert_counter_mapped: num_nvl_experts,                    要写入的。moe_recv_expert_counter_mapped[i] 表示当前rank内的expert i收到的来自所有节点发送的token数量之和。
    num_experts:                    1,                                  要读取的。表示节点内所有rank的专家数量之和。
    is_token_in_rank:               [num_tokens, kNumRanks],            要读取的。表示当前rank要发送的token是否需要发送到节点内其他rank。
    num_tokens:                     1,                                  要读取的。表示当前rank要发送的token数量。
    num_worst_tokens:               1,                                  要读取的。表示最坏情况下当前rank要接收的token数量之和。用于与分配接收空间的大小。
    num_channels:                   1,                                  要读取的。表示接下来进行dispatch通信使用的channel数量。
    expert_alignment:               1,                                  要读取的。表示当前rank内的expert thread_id收到的token数量以expert_alignment为对齐单位。
    rdma_clean_offset:              1,                                  要读取的。表示RDMA缓冲区需要清零的偏移量。
    rdma_num_int_clean:             1,                                  要读取的。表示RDMA缓冲区需要清零的int数量。
    nvl_clean_offset:               1,                                  要读取的。表示NVLink缓冲区需要清零的偏移量。
    nvl_num_int_clean:              1,                                  要读取的。表示NVLink缓冲区需要清零的int数量。
    rdma_channel_prefix_matrix:     [kNumRDMARanks, num_channels],      要写入的。当前rank要发送token，所以跟channel有关。rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + i]表示当前rank作为dispatch发送端从channel 0到channel i（包含）累计要发送到节点dst_rdma_rank的token数量之和（r2n的channel级别前缀和）。
    recv_rdma_rank_prefix_sum:      [kNumRDMARanks],                    要写入的。当前rank要接收token，所以跟channel无关。recv_rdma_rank_prefix_sum[i]表示的是从节点 0到节点 i（包含）累计发送到当前rank的token数量之和，每个节点只有和当前rank的local_rank相同的rank才发送。
    gbl_channel_prefix_matrix:      [kNumRanks, num_channels],          要写入的。当前rank要发送token，所以跟channel有关。gbl_channel_prefix_matrix[dst_rank * num_channels + i]表示当前rank作为dispatch发送端从channel 0到channel i（包含）累计要发送到rank dst_rank的token数量之和（r2r的channel级别前缀和）。
    recv_gbl_rank_prefix_sum:       [kNumRanks],                        要写入的。当前rank要接收token，所以跟channel无关。recv_gbl_rank_prefix_sum[i] 表示从全局的rank 0到rank i（包含）累计发送到当前rank的token数量之和。
    rdma_buffer_ptr:                void*                               要写入的。rdma缓冲区。大小是 int(1e9) 字节。
    buffer_ptrs:                    NUM_MAX_NVL_PEERS, void**           要写入的。节点内所有rank的NVLink buffer。
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
        /*
        sm_id 只做了一点微小的工作，讲五句话:
        1、为后面的dispatch清零的元数据区域。
        2、写入: recv_rdma_rank_prefix_sum[i] 表示的是从节点 0到节点 i（包含）累计发送到当前rank的token数量之和。
        3、写入: recv_gbl_rank_prefix_sum[i] 表示从全局的rank 0到rank i（包含）累计发送到当前rank的token数量之和。
        4、写入: moe_recv_rdma_counter_mapped: 记录当前rank收到的来自所有节点要发送给当前rank的token数量之和。
        5、写入: moe_recv_expert_counter_mapped[i] 表示当前rank内的expert i收到的来自所有节点发送的token数量之和。
        */

        // Communication with others
        // Global barrier: the first warp does intra-node sync, the second warp does internode sync
        EP_DEVICE_ASSERT(num_warps > 1);
        EP_DEVICE_ASSERT(kNumRDMARanks <= num_threads);

        // waiting for all previous inflight wrs to complete,
        // in case of rewriting cleared rdma_buffer
        /*
        num_rc_per_pe: int类型，每个rank的RC QP数量。
        num_devices_initialized: 当前 PE 选择并成功初始化的 NIC 设备数量。
        qps_per_rdma_rank: 表示的是当前PE每次往某个节点发送数据的qp数量。
        */
        auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
        /* 
        等待所有之前进行的 RDMA 操作完成，确保缓冲区清理时不会与正在进行的写操作冲突。
        为什么减 1: PE不用往自身所在节点发送rdma数据，所以需要减去 1 。这意味着需要等待的QP总数 = 当前PE对每个目标节点的QP数量 × 目标节点数量（不包括自己）
        */
        for (int i = thread_id; i < qps_per_rdma_rank * (kNumRDMARanks - 1); i += num_threads) {
            /*
            使用模运算的循环偏移（Circular Shift）使得dst_rdma_rank的值在[0, kNumRDMARanks-1]之间循环，
            但是恰好避开rdma_rank，从而实现对所有目标节点的QP进行等待。 
            (i / qps_per_rdma_rank + 1) ∈ [1, kNumRDMARanks - 1]，是变量。
            rdma_rank ∈ [0, kNumRDMARanks - 1]，是常量。
            (i / qps_per_rdma_rank + rdma_rank + 1) ∈ [rdma_rank + 1, rdma_rank + kNumRDMARanks - 1]，是变量。
            将这个范围分为两部分: [rdma_rank + 1, kNumRDMARanks] 和 [kNumRDMARanks + 1, rdma_rank + kNumRDMARanks - 1]。
            第一部分对kNumRDMARanks取模，得到的是{rdma_rank + 1, rdma_rank + 2, ..., 0} 中的一个整数。
            第二部分对kNumRDMARanks取模，得到的是{1, 2, ..., rdma_rank - 1} 中的一个整数。
            因此，这个变量对kNumRDMARanks取模，得到的是[0, kNumRDMARanks-1]中除了rdma_rank的任何一个整数。
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
            // 存储当前rank要发往节点thread_id的token数量。每个节点要设置 1 个int值。
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
        /*
        buffer_ptrs[nvl_rank]的内存布局:
        +------------------------------------------------------------------------------------------------+
        | 1. nvl_reduced_num_tokens_per_expert  大小: num_rdma_experts * sizeof(int) 字节                 |
        +------------------------------------------------------------------------------------------------+
        | 2. nvl_recv_num_tokens_per_rank  大小: NUM_MAX_NVL_PEERS * kNumRDMARanks * sizeof(int) 字节     |
        +------------------------------------------------------------------------------------------------+
        | 3. nvl_recv_num_tokens_per_expert  大小: NUM_MAX_NVL_PEERS * num_nvl_experts * sizeof(int) 字节 |
        +------------------------------------------------------------------------------------------------+
        */
        auto nvl_recv_buffer = buffer_ptrs[nvl_rank];
        /* 
        nvl_reduced_num_tokens_per_expert的长度是num_rdma_experts。
        nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i] 表示
        所有节点要先发送到当前节点的rank nvl_rank再发送到的rank thread_id上的专家i的token数量之和。
        advance_also(nvl_send_buffer) 让nvl_send_buffer也和nvl_recv_buffer一样偏移了num_rdma_experts个int元素的字节。
        这是为了后面的nvl_recv_num_tokens_per_rank和nvl_send_num_tokens_per_rank在同节点的不同的rank的buffer_ptrs中指针索引相同，
        nvl_send_num_tokens_per_expert和nvl_recv_num_tokens_per_expert在同节点的不同的rank的buffer_ptrs中指针索引也相同。
        */
        auto nvl_reduced_num_tokens_per_expert = Buffer<int>(nvl_recv_buffer, num_rdma_experts).advance_also(nvl_send_buffer);
        /*
        nvl_send_num_tokens_per_rank 的长度是NUM_MAX_NVL_PEERS，每个元素长度是kNumRDMARanks个int。
        nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] 表示的是节点i的nvl_rank发送给当前节点的nvl_rank后再发送给当前节点的thread_id的token数据量。
        注意: 这每个int值表示rank thread_id要发送给节点内的各个rank的来自各个节点的token数量。
        注意: nvl_send_num_tokens_per_rank 的字段命名中含有send，是从当前rank（local_rank nvl_rink）的角度出发的，
             因为最后是当前rank发往当前节点的local_rank thread_id。
        */
        auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        /* 
        记录当前rank要发送给当前节点的各个rank的各个专家的token数量。
        nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] 表示所有节点要发送给当前节点的rank thread_id之后再发送给rank thread_id上的专家i的token数量。
        */
        auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);
        /*
        同节点内的其它rank会给当前rank的nvl_recv_num_tokens_per_rank的内存中写数据的，
        就像当前rank给local_rank thread_id的nvl_send_num_tokens_per_rank写数据一样。
        nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank]
        表示节点src_rdma_rank的src_nvl_rank先发送给当前节点的local_rank src_nvl_rank再转发给当前节点的local_rank nvl_rank（当前rank）的token数量。
        */
        auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
        /*
        nvl_recv_num_tokens_per_expert.buffer(i)[thread_id] 表示所有节点要发送给当前节点的rank i之后再发送给rank nvl_rank上的专家thread_id的token数量。
        */
        auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);

        // Clean up for later data dispatch
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        // 这几个元数据肯定是小于nvl_clean_offset的。
        EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                             nvl_send_num_tokens_per_expert.total_bytes <= nvl_clean_offset * sizeof(int));
        #pragma unroll
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            // 为后面的dispatch清零的元数据区域的大小（以int为单位）
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
                /* 
                rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + thread_id] 表示节点i的专家thread_id要发送给当前节点的token数量。
                sum 表示对所有节点要发送到当前节点的专家thread_id的token数量之和。
                */
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + thread_id];
            nvl_reduced_num_tokens_per_expert[thread_id] = sum;
        }
        __syncthreads();

        // Reduce RDMA received tokens
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumRDMARanks; ++i) {
                /* 
                rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + num_rdma_experts] 表示节点i要发送给当前rank的token数量。
                 */
                sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + num_rdma_experts];
                // recv_rdma_rank_prefix_sum[i]表示的是从节点 0到节点 i（包含）累计发送到当前rank的token数量之和。
                recv_rdma_rank_prefix_sum[i] = sum;
            }
            if (num_worst_tokens == 0) {  // 如果不是用Worst case模式进行预分配。
                // 直到等到moe_recv_rdma_counter_mapped是-1，说明前一次dispatch过程已经完成。现在需要设置moe_recv_rdma_counter_mapped新一轮的值。
                while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1)
                    ;
                // 记录当前rank收到的所有节点要发送给当前rank的token数量之和。
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
                
                nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] 表示的是节点i的nvl_rank发送给当前节点的nvl_rank后再发送给当前节点的thread_id的token数据量。
                注意: nvl_send_num_tokens_per_rank 是当前节点的local_rank thread_id的nvlink缓冲区。
                注意: 命名中含有send，都是从当前rank（local_rank nvl_rink）的角度出发的，因为最后是当前rank发往当前节点的local_rank thread_id。
                注意: 当前rank写节点内的rank thread_id的nvl缓冲区，而节点内的其它rank也在往rank nvl_rank的nvl缓冲区写入。
                */
                nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] = rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id];
            #pragma unroll
            for (int i = 0; i < num_nvl_experts; ++i)
                /*
                nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i] 表示所有节点要发送到当前节点的rank thread_id的rank thread_id上的专家i的token数量之和。
                nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] 表示所有节点要发送给当前节点的rank thread_id之后再发送给rank thread_id上的专家i的token数量。
                */
                nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] = nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i];
        }
        /* 
        同步节点内的跨 rank 的 barrier， 等待节点内的所有rank的所有线程到达，还要保持节点内的内存屏障。
        因为前面有对节点内的其它rank的nvl缓冲区buffer_ptrs[thread_id]的写入。
        */
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);

        // Reduce the number of tokens per rank/expert
        EP_DEVICE_ASSERT(num_nvl_experts <= num_threads);
        if (thread_id == 0) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < num_ranks; ++i) {
                int src_rdma_rank = i / NUM_MAX_NVL_PEERS, src_nvl_rank = i % NUM_MAX_NVL_PEERS;
                /* 
                nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank]
                表示节点src_rdma_rank的src_nvl_rank先发送给当前节点的src_nvl_rank再发送给当前节点的当前rank（rank nvl_rank）的token数量。
                注意: 这里之所以能直接读取nvl_recv_num_tokens_per_rank，是因为别的rank在前面往对应当前的rank的nvl_send_num_tokens_per_rank写入数据了。
                */
                sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank];
                /*
                recv_gbl_rank_prefix_sum[i] 表示从全局的rank 0到rank i（包含）累计发送到当前rank的token数量之和。
                */
                recv_gbl_rank_prefix_sum[i] = sum;
            }
            if (num_worst_tokens == 0) {
                while (ld_volatile_global(moe_recv_counter_mapped) != -1)
                    ;
                // 记录当前rank收到的所有rank发送给当前rank的token数量之和。
                *moe_recv_counter_mapped = sum;
            }
        }
        if (thread_id < num_nvl_experts) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                /*
                nvl_recv_num_tokens_per_expert.buffer(i)[thread_id] 表示所有节点要发送给当前节点的rank i之后再发送给当前rank nvl_rank上的专家thread_id的token数量。
                sum求和之后表示的是所有节点要发送给当前rank nvl_rank上的专家thread_id的token数量之和。
                注意: 这里之所以能直接读取nvl_recv_num_tokens_per_expert，是因为别的rank在前面往对应当前的rank的nvl_send_num_tokens_per_expert写入数据了。
                */
                sum += nvl_recv_num_tokens_per_expert.buffer(i)[thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment; // 对齐到expert_alignment的倍数。
            if (num_worst_tokens == 0) {
                /* 
                不等于-1了，说明moe_recv_expert_counter_mapped[thread_id]的值已经被读取。
                设置-1在 deep_cp.cpp中的internode::notify_dispatch前面的moe_recv_expert_counter[i] = -1中。
                */
                while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1)
                    ;
                // moe_recv_expert_counter_mapped[i] 表示当前rank内的expert i收到的来自所有节点发送的token数量之和。
                moe_recv_expert_counter_mapped[thread_id] = sum; 
            }
        }

        // Finally barrier
        if (thread_id == 32) // 保证内存可见性，只需要有一个线程发起即可。因为这不是屏障同步。
            // 仅确保先前发起的**内存存储操作（指本地PE对自己内存的常规存储操作）**完成并可见（本地PE自己可见）
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        // 执行节点内的屏障同步，并且实现节点内跨rank之间的内存可见。但是不跨越节点可见。
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else {  
        /* 
        其余所有的sm“只做了一点微小的工作，讲两句话”:
        1、计算 (全局rank, channel) 级别的前缀和。写入: 
            gbl_channel_prefix_matrix[dst_rank * num_channels + i] 表示当前rank作为dispatch发送端从channel 0到
            channel i（包含）累计要发送到rank dst_rank的token数量之和（r2r的channel级别前缀和）。
        2、计算 (节点, channel) 级别的前缀和。写入: 
            rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + i] 表示当前rank作为dispatch发送端从channel 0到
            channel i（包含）累计要发送到节点dst_rdma_rank的token数量之和（r2n的channel级别前缀和）。
        */
        // Calculate meta data
        // num_sms 默认值是20，sm_id=0执行上面的任务了，所以这里的 sm_id 的取值范围是0到18，dst_rdma_rank 的取值范围是0到18。
        int dst_rdma_rank = sm_id - 1;
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;  // 记录当前channel要发送的token的开始索引和结束索引。
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            // 统计每个channel需要发送到每个目标rdma_rank内每个nvl_rank的token数量
            int total_count = 0, per_nvl_rank_count[NUM_MAX_NVL_PEERS] = {0};
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) { // 线程按warp循环步进
                // sizeof(bool)是一字节
                EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
                /*
                is_token_in_rank的数据类型是bool数组，shape是[num_tokens, kNumRanks], 表示当前rank要发送的token是否需要发送到节点内其他rank。
                is_token_in_rank_uint64 表示token i是否要发送到节点dst_rdma_rank上的8个rank。
                */
                auto is_token_in_rank_uint64 =
                    *reinterpret_cast<const uint64_t*>(is_token_in_rank + i * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS);
                auto is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_uint64);
                #pragma unroll
                for (int j = 0; j < NUM_MAX_NVL_PEERS; ++j)
                    // per_nvl_rank_count[j] 表示当前rank要发送到节点dst_rdma_rank上的rank j的token数量。
                    per_nvl_rank_count[j] += is_token_in_rank_values[j];
                // is_token_in_rank_uint64的8个bool值中只要有一个不是0，则total_count就加1。
                // 统计当前rank要发送到节点dst_rdma_rank上的token数量。
                total_count += (is_token_in_rank_uint64 != 0);
            }

            // Warp reduce
            // 计算得到了当前rank要发往 (节点dst_rdma_rank, channel channel_id) 的token数量。
            total_count = warp_reduce_sum(total_count);
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                // 计算得到了当前rank要发往 (节点dst_rdma_rank上的rank i, channel channel_id) 的token数量。
                per_nvl_rank_count[i] = warp_reduce_sum(per_nvl_rank_count[i]);

            // Write into channel matrix
            /* 
            当前sm对应一个目的地节点和该节点上的所有rank，而一个warp对应一个channel。
            现在需要记录 (节点dst_rdma_rank, channel channel_id) 的token数量，所以只需用当前warp上的一个线程执行即可。
            */
            if (elect_one_sync()) {
                #pragma unroll
                // 记录聚合结果到全局矩阵
                for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                    // 记录当前rank要发往 (节点dst_rdma_rank上的rank i, channel channel_id) 的token数量。此时的记录还不是前缀和。
                    gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + i) * num_channels + channel_id] = per_nvl_rank_count[i];
                // 记录当前rank要发往 (节点dst_rdma_rank, channel channel_id) 的token数量。此时的记录还不是前缀和。
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
    auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, 
                                               num_max_rdma_chunked_recv_tokens, num_channels);
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
/* 
V3每个MoE层有256个专家，而每个GPU托管8个专家，每个节点托管8*8=64个专家。一共只需4个节点就可以托管256个专家。
V3使用了节点限制的路由，每个token最多发送到$$M=4$$个节点。
*/
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

template <bool kLowLatencyMode,
          int kNumRDMARanks,
          bool kCachedMode,
          int kNumTMABytesPerWarp,   // 16384
          int kNumDispatchRDMASenderWarps,  // 7
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
/* 
__launch_bounds__(max_threads_per_block, min_blocks_per_multiprocessor)
这是 CUDA 的编译指令，用于向编译器提供 kernel 启动配置的提示，以优化寄存器分配和占用率。
1、max_threads_per_block: 表示每个block的最大线程数。
   (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32 = (7 + 1 + 8) * 32 = 512 表示每个sm的线程数量。
2、min_blocks_per_multiprocessor: 表示每个 SM 至少驻留的block数量。
   这里的1表示每个SM至少需要1个block，因为每个SM至少需要1个block来运行内核。
*/
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
    /*
    recv_x:                              要写入的, [num_recv_tokens, hidden], 接收端接收到的token组成的tensor，最终返回给expert模型使用。
    recv_x_scales:                       要写入的, [num_recv_tokens, num_scales], 接收端接收到的token的scale，每个token有num_scales个scale，每个scale表示128个float8_e4m3fn。接收端用它来还原FP8数据到原始精度。
    recv_topk_idx:                       要写入的, [num_recv_tokens, num_topk], 接收的每个token激活的专家在当前接收端rank中的专家的局部专家ID，不在这个rank中的专家就是-1。
    recv_topk_weights:                   要写入的, [num_recv_tokens, num_topk], 与recv_topk_idx对应，表示接收的每个token激活的专家在当前接收端rank中的专家的权重，不在这个rank中的专家权重就是0。
    recv_src_meta:                       要写入的, num_recv_tokens, SourceMeta类型。接收端接收到的token的源元数据，包含发送端rank信息和token在节点内的路由信息，用于combine阶段的数据聚合。
    x:                                   要读取的, [num_tokens, hidden], 从MultiHeadAttention传入的token hidden数据，当前dispatch发送端要分发的token数据。
    x_scales:                            要读取的, [num_tokens, num_scales], x_scales[i, j]：第 i 个token的第 j 组（每组128个元素）的反缩放因子。发送端用它来量化FP8数据。
    topk_idx:                            要读取的, [num_tokens, num_topk], 表示每个token要路由到的topk个expert的全局ID。
    topk_weights:                        要读取的, [num_tokens, num_topk], 与topk_idx对应，表示每个token要路由到的topk个expert的权重。
    send_rdma_head:                      要写入的, [num_tokens, kNumRDMARanks], send_rdma_head[token_idx * kNumRDMARanks + dst_rdma_rank]: 当前rank的token token_idx要发送到节点dst_rdma_rank的local_rank nvl_rank的RDMA缓冲区环形队列的head索引，用于combine阶段查询token在缓冲区中的位置。
    send_nvl_head:                       过 要写入的, [num_rdma_recv_tokens, NUM_MAX_NVL_PEERS], send_nvl_head[token_idx * NUM_MAX_NVL_PEERS + dst_nvl_rank]: 从RDMA缓冲区接收到的第token_idx个token转发到节点内rank dst_nvl_rank的NVL环形缓冲区的单调递增head索引，-1表示这个token没有发送到节点内rank dst_nvl_rank。用于combine阶段查询token在缓冲区中的位置。
        注意: num_rdma_recv_tokens 就是在notify_dispatch中记录到 moe_recv_rdma_counter_mapped 中的值，也就是当前rank收到的来自所有节点要发送到当前rank的token数量之和。
    
    
    recv_rdma_channel_prefix_matrix:     要写入的, [kNumRDMARanks, num_channels], recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel 0到channel channel_id（包含）累计要发送到当前节点的token数量之和（r2n的channel级别前缀和）。
    recv_gbl_channel_prefix_matrix:      过 要写入的, [kNumRanks, num_channels], recv_gbl_channel_prefix_matrix[(src_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id]表示
                                                  从全局的rank 0到节点src_rdma_rank的rank src_nvl_rank的从channel 0到channel (channel_id - 1)（包含）累计要发送给当前rank的token数量之和。也就是:
                                                  token数据在接收端输出数组recv_x中的给全局rank (src_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank)的channel channel_id准备的空间的起始索引。
                                                  recv_gbl_channel_prefix_matrix 的意义是在combine的时候能将传给combine的token数据正确地写入到cimbine的输出tensor的对应位置中。
    rdma_channel_prefix_matrix:          过 要读取的, [kNumRDMARanks, num_channels], rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + i]表示的是当前rank作为dispatch发送端从channel 0到channel i（包含channel i）累计要发送到节点dst_rdma_rank的token数量之和（channel级别前缀和）。
    recv_rdma_rank_prefix_sum:           要读取的, [kNumRDMARanks], recv_rdma_rank_prefix_sum[i]表示的是从节点 0的rank nvl_rank到节点 i的rank nvl_rank（包含节点 i的rank nvl_rank）累计发送到当前rank的token数量之和（rank级别前缀和）。
    gbl_channel_prefix_matrix:           过 要读取的, [kNumRanks, num_channels], gbl_channel_prefix_matrix[dst_rank * num_channels + i]表示的是当前rank作为dispatch发送端从channel 0到channel i（包含channel i）累计要发送到rank dst_rank的token数量之和（r2r的channel级别前缀和）。
    recv_gbl_rank_prefix_sum:            要读取的, [kNumRanks], recv_gbl_rank_prefix_sum[i]表示从全局的rank 0到rank i（包含rank i）累计发送到当前rank的token数量之和（rank级别前缀和）。
    is_token_in_rank:                    过 要读取的, bool 类型[num_tokens, kNumRanks], is_token_in_rank[token_idx * kNumRanks + dst_rank]表示token token_idx是否发送到rank dst_rank。
    num_tokens:                          过 要读取的, 表示当前rank要发送的token数量。
    num_worst_tokens:                    要读取的, 如果使用Worst case模式进行预分配，表示预分配的token数量，否则为0。
    hidden_int4:                         要读取的, 表示每个token的hidden size（以int4为单位）。
    num_scales:                          要读取的, 表示每个token的scale数量，等于hidden / 128。每个scale值用float表示。
    num_topk:                            要读取的, 表示每个token激活的专家数量。
    num_experts:                         要读取的, 表示全局所有的专家数量。
    scale_token_stride:                  要读取的, 
    scale_hidden_stride:                 要读取的, 
    rdma_buffer_ptr:                     要读写的, RDMA对称内存缓冲区指针，用于节点间RDMA通信，包含token数据、元数据和同步信号。
    num_max_rdma_chunked_send_tokens:    要读取的, 表示每次RDMA发送的最大token数量（chunk大小），用于限制分批发送token数据。
    num_max_rdma_chunked_recv_tokens:    要读取的, 表示每个 (channel, RDMA rank) 的 RDMA 环形接收缓冲区的 token 容量
    buffer_ptrs:                         要读写的, NUM_MAX_NVL_PEERS, 节点内所有rank的NVLink buffer指针，用于节点内NVL通信。
    num_max_nvl_chunked_send_tokens:     要读取的, 表示每次NVL发送的最大token数量（chunk大小），用于限制分批发送token数据。
    num_max_nvl_chunked_recv_tokens:     要读取的, 表示NVL接收端环形缓冲区的总token容量。
    rank:                                要读取的, 表示当前rank的编号。
    num_ranks:                           要读取的, 表示全局rank的总数。
    */
    /*
    dispatch函数根据SM ID和warp ID为每个warp分配不同的角色：
        kRDMASender,            // RDMA发送器：发送数据到远程节点
        kRDMASenderCoordinator, // RDMA发送协调器：协调RDMA传输
        kRDMAAndNVLForwarder,   // RDMA到NVL转发器：从RDMA缓冲区转发到NVL缓冲区
        kForwarderCoordinator,  // RDMA到NVL转发协调器：协调转发操作
        kNVLReceivers           // NVL接收器：从NVL缓冲区接收数据
    */
    enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };

    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_channels = num_sms / 2, channel_id = sm_id / 2;  // 还是两个sm对应一个channel
    const bool is_forwarder = sm_id % 2 == 0;
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;

    /*
    num_rc_per_pe表示每个 PE 在每个 NIC 设备上的 RC QP 数量
    若 num_rc_per_pe == num_channels，每个 channel 有 1 个 QP。
    若 num_rc_per_pe >= num_sms，每个 SM 可分配至少 1 个 QP。
    test_internode.py中:
        num_qps_per_rank = max(num_sms, ll_num_experts // num_ranks if args.test_ll_compatibility else 0)。
    然后在中buffer.py中: os.environ['NVSHMEM_IBGDA_NUM_RC_PER_PE'] = f'{num_qps_per_rank}'，
    */
    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms);

    // 合理地分配sm中的warp去并行处理不同类型的任务，可以最大限度地提升并行度，打满所有warp。
    const auto role_meta = [=]() -> std::pair<WarpRole, int> {
        // 每个sm的warp数量: (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) = 7 + 1 + 8 = 16
        // 偶数sm_id的warp负责RDMA缓冲区到NVL缓冲区的转发
        if (is_forwarder) {
            /*
            为什么这两种warp要放在同一个sm中？
            因为它们要协同读写共享内存变量 forward_channel_head 和 forward_channel_retired。
            */
            if (warp_id < NUM_MAX_NVL_PEERS) {
                /* 
                kRDMAAndNVLForwarder RDMA到NVL转发器: 前8个warp作为RDMA缓冲区到NVL缓冲区的转发器。
                (warp_id + channel_id) % NUM_MAX_NVL_PEERS 这样写是因为:
                1、同一个channel中的所有warp的channel_id都相同，所以不同的warp负责往当前节点的不同local_rank转发数据。而且是一一对应的。
                2、“+ channel_id”可以swizzle当前rank的不同的偶数sm_id的sm，使得各个sm中的kRDMAAndNVLForwarder warp可以均匀地发往不同的local_rank，
                   不会造成多个sm一下子都往同一个local_rank发数据。
                */
                return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
            } else { 
                /* 
                kForwarderCoordinator RDMA到NVL转发协调器: 协调多个 kRDMAAndNVLForwarder warp 并行处理
                不同远程 RDMA rank 的转发操作时向远程节点报告已消费的 head 位置。
                如果每个 kRDMAAndNVLForwarder warp 都立即更新远程 head，会产生大量 RDMA 原子操作，开销大。
                */
                return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};  // warp_id 一定是8，target_rank 一定是0。
            }
        } else {
            /*
            每个sm有 (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) = 7 + 1 + 8 = 16 个warp
            */
            if (warp_id < kNumDispatchRDMASenderWarps) {  // kRDMASender RDMA发送器: 把 tensor x 写入当前节点的rdma_channel_data.send_buffer
                return {WarpRole::kRDMASender, -1};
            } else if (warp_id == kNumDispatchRDMASenderWarps) { // kRDMASenderCoordinator RDMA发送协调器: 协调RDMA传输
                return {WarpRole::kRDMASenderCoordinator, -1};
            } else {                                             // kNVLReceivers NVL接收器: 从NVL缓冲区接收数据
                /*
                (warp_id - kNumDispatchRDMASenderWarps) 是为了让kNVLReceivers中warp_id想跟channel_id一起负载均衡时
                “负载均衡的逻辑warp_id”是从0开始的。这是写负载均衡时的常规写法。
                
                TODO: 但是前面kRDMASender和kRDMASenderCoordinator好像一起占用了 (kNumDispatchRDMASenderWarps + 1) 个warp，
                      为什么这里不减 1 呢？
                因为kNVLReceivers warp有8个，所以实际上这里减不减kNumDispatchRDMASenderWarps没什么影响。
                */
                return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS};
            }
        }
    }();
    auto warp_role = role_meta.first;
    // target_rank 表示节点内的其他rank
    auto target_rank = role_meta.second;  // Not applicable for RDMA senders
    EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS);

    // Data checks
    EP_DEVICE_ASSERT(num_topk <= 32);

    // RDMA symmetric layout
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto scale_bytes = num_scales * sizeof(float);  // 表示每个token的scale数量，等于hidden / 128。
    /*
    align_up(hidden_int4 * sizeof(int4) + num_scales * sizeof(float) + sizeof(SourceMeta) +
            num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float), sizeof(int4))

    */
    auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk);
    /*
    rdma缓冲区中缓存 (num_channels, kNumRDMARanks)个channel节点对的token数据缓冲区。
    */
    auto rdma_channel_data = SymBuffer<uint8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, 
                                                kNumRDMARanks, channel_id, num_channels);
    /*
    rdma缓冲区中缓存当前节点要发往 (num_channels, kNumRDMARanks)个channel节点对的r2r的channel级别前缀和（NUM_MAX_NVL_PEERS * 2）和
                                                                           r2n的channel级别前缀和（2）
    注意: rdma_channel_meta 是既有发送端缓存也有接收端缓存。因为当前rank既要把自己要发送的token数量告诉别的所有节点和rank，
          也要知道别的所有节点和rank要发送到发送到自己的token数量以便于自己在接收token时知道应该放在缓冲区中的索引。
    */
    auto rdma_channel_meta = SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
    /*
    rdma_channel_head: 存储当前rank的发送缓冲区rdma_channel_data.send_buffer(land_id)中经由channel channel_id发送给
        节点land_id的rank nvl_rank的环形队列的head索引。该值在kForwarderCoordinator warp中由接收端rank更新。
        也就是“消费端才有资格更新head”。
    kRDMASender warp中自旋等待读取rdma_channel_head的值，因此rdma_channel_head存储在当前rank。
    注意: rdma_channel_head 只是记录当前rank的rdma发送缓冲区的环形队列head索引的，不涉及当前rank作为接收端的信息，索引不需要kDecoupled为true。

    */
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    /*
    rdma_channel_tail存储当前rank经由channel channel_id发送到远程节点的rank nvl_rank的对应当前rank的rdma接收缓冲区环形队列的tail索引。
    该值在kRDMASenderCoordinator warp中由发送端rank更新。
    也就是“生产端才有资格更新tail”。
    注意: rdma_channel_tail 只是记录远程节点的rank nvl_rank的rdma接收缓冲区的环形队列tail索引的，不涉及当前rank的rdma发送缓冲区的tail索引，
          该tail索引不需要kDecoupled为true。
    */
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

    /* 
    // NVL buffer layouts
    // NOTES: `rs_wr_buffer_ptr` means "Read for Senders, Write for Receivers", 
    // `ws_rr_buffer_ptr` means "Write for Senders, Read for Receivers"
    
    注意: 认真体会这里的（发送端读，接收端写）和（发送端写，接收端读）的含义和对称性。
         在下面的AsymBuffer中，有且仅有nvl_channel_head是（发送端读，接收端写），其他都是（发送端写，接收端读）。
         接收端才有资格写head指针。
    */
    void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
    int rs_wr_rank = 0, ws_rr_rank = 0;
    if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        /*
        kRDMAAndNVLForwarder: 主要负责把当前rank的rdma_channel_data.recv_buffer中的token数据通过NVLink发送到节点内的其他rank的nvl_channel_x中。
        当当前warp是kRDMAAndNVLForwarder warp时，当前rank作为发送端，要发送token数据给同节点的local_rank target_rank。

        target_rank: (warp_id + channel_id) % NUM_MAX_NVL_PEERS
        */
        rs_wr_buffer_ptr = buffer_ptrs[nvl_rank],
        ws_rr_buffer_ptr = buffer_ptrs[target_rank], // kRDMAAndNVLForwarder warp作为发送端往 ws_rr_buffer_ptr 指向的内存写数据。
        rs_wr_rank = nvl_rank,
        ws_rr_rank = target_rank;
    if (warp_role == WarpRole::kNVLReceivers)
        /*
        kNVLReceivers: 主要负责将当前节点的nvl缓冲区nvl_channel_x中的数据写到recv_x中。
        当当前warp是kNVLReceivers warp时，当前rank作为接收端，要接收来自同节点的local_rank target_rank的rdma缓冲区发送到当前rank的nvl缓冲区中的token数据到当前rank的recv中。

        target_rank: (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS
        */
        rs_wr_buffer_ptr = buffer_ptrs[target_rank],
        ws_rr_buffer_ptr = buffer_ptrs[nvl_rank],
        rs_wr_rank = target_rank,
        ws_rr_rank = nvl_rank;

    // Allocate buffers
    /*
    nvl_channel_x: 最终目的地rank的nvl缓冲区中缓存的来自于同节点的local_rank target_rank的rdma缓冲区的经由channel channel_id发送的token数据。
    
    在kRDMAAndNVLForwarder warp中，就是当前rank的rdma缓冲区要发送到的同节点的local_rank target_rank的nvl缓冲区。
    在kNVLReceivers warp中，就是当前rank的nvl缓冲区中，接收的来自同节点的local_rank target_rank的rdma缓冲区经由channel channel_id发送的token数据的部分。
    
    注意: 这里的AsymBuffer的构造函数的最后一个参数是offset，表示nvl_channel_x指向的是NUM_MAX_NVL_PEERS个rank的第rs_wr_rank个rank。

    注意: 1、advance_also函数传入的是rs_wr_buffer_ptr的别名，即: advance_also(void*& gbl_ptr)，这是说移动的指针变量只是rs_wr_buffer_ptr本身，
            但并不是移动了rs_wr_buffer_ptr在定义的时候的 buffer_ptrs[...]，因为rs_wr_buffer_ptr不是buffer_ptrs[...]的别名。
            变量buffer_ptrs[...]指向的内存地址在整个dispatch函数调用过程中没有改变过。
         2、之所以要对当前rank和target_rank的nvl缓冲区在分配不同内存区域的存储内容时要使用advance_also，
            是因为在cuda并行运行时，同时target_rank也在作为“当前rank”来分配别的同节点的rank的nvl缓冲区的存储内容，
            为了保持在各个rank都执行时缓冲区各区域存储内容的含义的一致性，就需要保持“当前rank和rank target_rank的nvl缓冲区的指针指向位置的一致性”。
    */
    auto nvl_channel_x = AsymBuffer<uint8_t>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                            NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    /*
    注意: nvl_channel_prefix_start 和 nvl_channel_prefix_end 存储在哪个rank的nvl缓冲区中，就说明这个rank是dispatch的最终接收端。

    1、在kRDMAAndNVLForwarder warp中，nvl_channel_prefix_start和nvl_channel_prefix_end存储在rank target_rank 的nvl缓冲区。
        nvl_channel_prefix_start.buffer() + lane_id: 表示节点lane_id的rank nvl_rank    要（经由当前节点的rank nvl_rank）   从channel 0到channel (channel_id - 1)（包含）累计要发送到当前节点的rank target_rank的token数量之和（r2r的channel级别前缀和）。
        nvl_channel_prefix_end.buffer() + lane_id  : 表示节点lane_id的rank nvl_rank    要（经由当前节点的rank nvl_rank）   从channel 0到channel channel_id（包含）      累计要发送到当前节点的rank target_rank的token数量之和（r2r的channel级别前缀和）。

    2、在kNVLReceivers        warp中，nvl_channel_prefix_start和nvl_channel_prefix_end需要轮询读，所以存储在当前rank 的nvl缓冲区。
        nvl_channel_prefix_start.buffer() + lane_id: 表示节点lane_id的rank target_rank 要（经由当前节点的rank target_rank）从channel 0到channel (channel_id - 1)（包含）累计要发送到当前rank的                 token数量之和（r2r的channel级别前缀和）。
        nvl_channel_prefix_end.buffer() + lane_id  : 表示节点lane_id的rank target_rank 要（经由当前节点的rank target_rank）从channel 0到channel channel_id（包含）      累计要发送到当前rank的                 token数量之和（r2r的channel级别前缀和）。

    nvl_channel_prefix_start和nvl_channel_prefix_end在kRDMAAndNVLForwarder warp 中写。即"Write for Senders"。 kRDMAAndNVLForwarder warp是发送端。
    nvl_channel_prefix_start和nvl_channel_prefix_end在kNVLReceivers warp        中读。即"Read for Receivers"。kNVLReceivers warp       是接收端。
    */
    auto nvl_channel_prefix_start = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, 
                                        channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    auto nvl_channel_prefix_end = AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, 
                                        channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);
    /*
    谁要轮询读head就把head存储在谁那里。
    注意: nvl_channel_head指向的内存到底存储在哪个rank，就看轮询读取nvl_channel_head.buffer()的是哪个rank的warp。
    例如: 在kNVLReceivers warp中，为什么当前rank的nvl缓冲区的head索引，需要存储在同节点的local_rank target_rank的nvl缓冲区中？而不是存储在当前rank的nvl缓冲区中？
          这是因为同节点的local_rank target_rank在往当前rank的nvl缓冲区写数据时，是通过轮询读取当前rank的nvl环形队列的head指针的，
          如果轮询读取每次都是到当前rank的nvl缓冲区中来读取，那么每次读取都要经过NVLink通信，导致效率太低。
          而将这个head索引存储在同节点的local_rank target_rank的nvl缓冲区中，那么它轮询读取的时候就读的是它本地的内存，不需要经过NVLink通信。
          另外，这里也不会出现多个rank对这个环形队列的head指针的竞争，因为当前rank的每个环形队列（nvl_channel_x）都是对应同节点的唯一一个local_rank target_rank的。
    在kRDMAAndNVLForwarder warp中，把同节点的local_rank target_rank的nvl缓冲区中的head索引存储在当前rank的nvl缓冲区中也是同理，
    因为当前rank的在往同节点的local_rank target_rank的nvl缓冲区中写数据前，先要轮询读取同节点的local_rank target_rank的nvl缓冲区中的head索引。

    1、在kRDMAAndNVLForwarder warp中，nvl_channel_head存储在当前rank的nvl缓冲区中， 记录当前节点的local_rank target_rank的nvl缓冲区中的环形队列的head索引。
       nvl_channel_head.buffer()表示当前rank要经由 channel channel_id发送到相同节点的rank target_rank的token在相同节点的rank target_rank的nvl缓冲区中的环形队列的head索引。
       这体现了"Read for Senders（当前rank作为发送端）"。
    2、在kNVLReceivers        warp中，nvl_channel_head存储在当前节点的local_rank target_rank的nvl缓冲区中，记录当前rank的nvl缓冲区中的环形队列的head索引。 
       nvl_channel_head.buffer()表示相同节点的rank target_rank要经由 channel channel_id发送到当前rank的token在当前rank的nvl缓冲区中的环形队列的head索引。
       这体现了"Write for Receivers（当前rank作为接收端）"。
    */
    auto nvl_channel_head = AsymBuffer<int>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, 
                                        channel_id, num_channels, ws_rr_rank).advance_also(ws_rr_buffer_ptr);
    /* 
    nvl_channel_tail.buffer(): 
    在kRDMAAndNVLForwarder warp 中写target_rank的nvl缓存，表示同节点的local_rank target_rank的nvl缓冲区中缓存来自于当前rank的rdma缓冲区的经由channel channel_id发送的token的tail索引。
    
    谁要轮询读tail也是把tail存储在谁那里。
    在kNVLReceivers warp        中轮询读当前rank的nvl缓存，表示当前rank的nvl缓冲区中缓存来自于同节点的local_rank target_rank的经由channel channel_id发送的token的tail索引。
    */
    auto nvl_channel_tail = AsymBuffer<int>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, 
                                        channel_id, num_channels, rs_wr_rank).advance_also(rs_wr_buffer_ptr);

    // RDMA sender warp synchronization
    // NOTES: `rdma_send_channel_tail` means the latest released tail
    // NOTES: `rdma_send_channel_window` means the ongoing 32 transactions' status
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
    /*
    barrier.sync 和 bar.sync 功能等价。
    注意: 这个sm内的barrier，是用于同步上面三个共享内存的。
    共享内存的写操作在 kNumDispatchRDMASenderWarps 个 kRDMASender warp中，初始化清零操作在 1 个 kRDMASenderCoordinator warp中执行，
    而 kNumDispatchRDMASenderWarps 个 kRDMASender warp会等到在那唯一一个kRDMASenderCoordinator warp中清零了这三个共享内存才会执行后续对共享内存的写操作。
    */
    auto sync_rdma_sender_smem = []() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // TMA stuffs
    /*
    extern __shared__: 表示动态共享内存。
    constexpr int kNumTMABytesPerWarp = 16384;
    cudaFuncAttributeMaxDynamicSharedMemorySize 动态共享内存大小：kNumTMABytesPerWarp * NUM_MAX_NVL_PEERS
    constexpr int smem_size = kNumTMABytesPerWarp * NUM_MAX_NVL_PEERS;
    */
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];  // 
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
    /*
    forward_channel_head[dst_nvl_rank][src_rdma_rank]: 表示当前rank的rdma接收缓冲区中接收的来自节点src_rdma_rank的local_rank nvl_rank的
                                                       经由channel channel_id发送的token在当前rank的rdma缓冲区中的环形队列的tail索引。
    为什么head指针forward_channel_head放在共享内存中呢？
    答: 因为forward_channel_head记录的是当前rank的rdma接收缓冲区中的指针，而当前sm中的kRDMAAndNVLForwarder warp是当前rank的rdma接收缓冲区的消费者，
        消费里面的数据往同节点的其他local_rank的nvl缓冲区中转发。

    forward_channel_retired[dst_nvl_rank] 表示当前节点的rank dst_nvl_rank是否已经完成接收了来自所有节点的所有rank的token。
    注意: 退休的是warp（dst_nvl_rank）
    */
    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    /*
    barrier.sync 和 bar.sync 功能等价。
    注意: 这个sm内的barrier，是用于同步共享内存forward_channel_head和forward_channel_retired的。
    共享内存的写操作在8个 kRDMAAndNVLForwarder warp中执行，初始化清零操作在 1 个 kForwarderCoordinator warp中执行，
    而 8 个kRDMAAndNVLForwarder warp会等到在那唯一一个kForwarderCoordinator warp中清零了这两个共享内存才会执行后续对共享内存的写操作。
    */
    auto sync_forwarder_smem = []() { asm volatile("barrier.sync 1, %0;" ::"r"((NUM_MAX_NVL_PEERS + 1) * 32)); };

    if (warp_role == WarpRole::kRDMASender) {
        /*
        一句话总结: 7个warp。主要负责把 tensor x 写入当前rank的rdma_channel_data.send_buffer。
        详细实现:
        1、把当前rank的全局rank级别的和全局节点级别的前缀和 写入到 别的节点的和当前rank具有相同local_rank的rank中。
        2、把 x 中的每个token逐个写入到当前rank的rdma发送缓冲区中。每次写入单个token前，需要先等待当前rank的rdma缓冲区中针对远程节点的环形队列中是否空出来一个token空间。
           注意: 是对每个token都执行while自旋等待。
        3、关于 lock ？TODO
        4、更新共享内存中的rdma_send_channel_tail[rdma_idx]，
           生产者才有资格更新tail。当前warp作为将x 发送到rdma_channel_data.send_buffer的生产者。
        */

        // Get tasks
        int token_start_idx, token_end_idx;  // 记录当前rank要发送的num_tokens个token中经由channel channel_id发送的部分token的开始索引和结束索引。
        get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // Send number of tokens in this channel by `-value - 1`
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            /* 
            1、当前rank要发送给其余所有节点的所有rank 的r2r的channel级别前缀和 和 r2n的channel级别前缀和 都要记录到
                接收节点的与当前rank有相同local_rank的rank的rdma_channel_meta.recv_buffer(rdma_rank)缓冲区中。
            2、如果for循环中遍历kNumRDMARanks个节点的相同local_rank的rank的时候发现是当前rank，就意味着这是“当前rank要发送给当前rank”的情况，
                那么就直接写到当前rank的rdma_channel_meta.recv_buffer(rdma_rank)缓冲区中。
                否则写到发送缓冲区中，然后发送到远程节点的有相同local_rank的rank的rdma_channel_meta.recv_buffer(rdma_rank)缓冲区中。
            这样一来，对于节点n上的rank r，节点n上的rank r上会记录所有节点上的local_rank为r的rank对要经由节点n上的rank r发往
                节点n上的的所有rank的token数量的r2r的channel级别前缀和 和 r2n的channel级别前缀和。
            */
            auto dst_ptr = dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : 
                                                        rdma_channel_meta.send_buffer(dst_rdma_rank);
            /*
            gbl_channel_prefix_matrix:  [kNumRanks, num_channels], 
                gbl_channel_prefix_matrix[dst_rank * num_channels + i]表示的是当前rank作为dispatch发送端从channel 0到
                channel i（包含channel i）累计要发送到rank dst_rank的token数量之和（r2r的channel级别前缀和）。
            rdma_channel_prefix_matrix: [kNumRDMARanks, num_channels], 
                rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + i]表示的是当前rank作为dispatch发送端从channel 0到
                channel i（包含channel i）累计要发送到节点dst_rdma_rank的token数量之和（r2n的channel级别前缀和）。
            */
            if (lane_id < NUM_MAX_NVL_PEERS) {
                /*
                channel 0之前的token数还是0。
                DeepEP中存储偏移量用“取反减一”存储是为了在读取的时候避免"数据未写入"和"写入值为0"之间的歧义。`-value - 1`, e.g. 0 -> -1, 1 -> -2 。
                gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1] 表示:
                  当前rank从channel 0到channel (channel_id - 1)（包含）累计要发送到节点dst_rdma_rank的rank lane_id的token数量之和（r2r的channel级别前缀和）。
                */
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + 
                                                            lane_id) * num_channels + channel_id - 1]) - 1;
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + 
                                    lane_id - NUM_MAX_NVL_PEERS) * num_channels + channel_id] - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                dst_ptr[lane_id] = -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + 
                                                            channel_id - 1]) - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
            }
            __syncwarp();

            // 经过上面的__syncwarp()同步warp内的线程，当前rank要发往节点dst_rdma_rank的所有rank的token前缀和都记录到了dst_ptr中。

            // Issue RDMA for non-local ranks
            if (dst_rdma_rank != rdma_rank) {
                /* 
                把当前rank发往其他节点的rank的r2r的channel级别前缀和 和 r2n的channel级别前缀和 都写入到
                其他所有的节点的local_rank为nvl_rank的rank的rdma_channel_meta.recv_buffer(rdma_rank)中。
                */
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                  channel_id,
                                                  lane_id,
                                                  0);
            }
        }
        sync_rdma_sender_smem();  // 等唯一一个kRDMASenderCoordinator warp中清零了三个共享内存才会执行后续对共享内存的写操作。

        // Iterate over tokens and copy into buffer
        int64_t token_idx;
        /*
        每个线程都会遍历下面从 token_start_idx 到 token_end_idx 范围内的所有token，但是每个线程要处理的token是不一样的，
        但是所有线程要处理的所有token在接收端的环形队列中的具体位置是固定的。因此就需要每个线程都找到自己要处理的token在接收端的环形队列中的具体位置，线程之间不能有冲突。
        因此就需要每个线程都维护自己要处理的当前token在接收端的环形队列中的具体位置，这个位置就是 (global_rdma_tail_idx - 1) % num_max_rdma_chunked_recv_tokens。

        global_rdma_tail_idx: 记录 token token_idx 是**第几个** 在当前rank要经由 channel channel_id 发往节点lane_id的rank nvl_rank的rdma缓冲区中的环形队列中的token，
                              “第几个”是从1开始的，也就是说 global_rdma_tail_idx 实际表示的是数量（而不是索引）。
        */
        int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
        /* 
        如果是写往相同节点的其他rank，那么就直接写到当前rank的rdma接收缓冲区中。
        说是叫发送缓冲区，实际上在当前kRDMASender warp中是作为接收tensor x 中的token的rdma接收缓冲区。
        这里称为发送缓冲区，是站在rdma_channel_data用于rdma通信时的角度来说的。
        */
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            // Read RDMA rank existence
            uint64_t is_token_in_rank_uint64 = 0;  // 记录token token_idx是否发往节点lane_id的 8 个local_rank的布尔值。
            if (lane_id < kNumRDMARanks) {
                /* 
                is_token_in_rank: 数据类型是一个字节的bool类型，shape是[num_tokens, kNumRanks]。
                这里转换为8字节的uint64_t类型的指针，是为了一次读取 NUM_MAX_NVL_PEERS = 8 个bool值。
                */
                is_token_in_rank_uint64 =
                    __ldg(reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS));
                // 只要节点lane_id的 8 个local_rank中有一个要接收当前rank要发送的token，那么global_rdma_tail_idx就加1。
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);
            }
            __syncwarp();

            // Skip the token which does not belong to this warp
            /*
            有 kNumDispatchRDMASenderWarps = 7 个 kRDMASender warp，每个warp分散负责 [token_start_idx, token_end_idx) 范围内的token。
            token被分配到的warp进行处理的布局如下：
            warp 0: token token_start_idx + 0, token_start_idx + 7, token_start_idx + 14, token_start_idx + 21, ...
            warp 1: token token_start_idx + 1, token_start_idx + 8, token_start_idx + 15, token_start_idx + 22, ...
            ...
            warp 6: token token_start_idx + 6, token_start_idx + 13, token_start_idx + 20, token_start_idx + 27, ...

            下面的代码是说如果token token_idx没有被分配到当前warp，那么当前warp就跳过对token token_idx的后续处理。
            */
            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            
            /*
            rdma_tail_idx: 表示token token_idx要经由channel channel_id发送到节点lane_id的local_rank nvl_rank的RDMA缓冲区环形队列的tail索引。
            global_rdma_tail_idx 表示数量，不是索引，所以rdma_tail_idx表示索引时就要减1。
            rdma_tail_idx 设置为-1时，是为了表示token token_idx没有发送到节点lane_id。这一点在后面有用到。
            */
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

            // Wait the remote buffer to be released
            auto start_time = clock64();
            /* 
            x2rdma等待之一: 等待rdma send缓冲区有空闲的token空间。

            kForwarderCoordinator warp中会在消费了节点lane_id的local_rank nvl_rank的rdma缓冲区的数据后增大 rdma_channel_head.buffer(lane_id)的值，
            也就是增大了下面的cached_rdma_channel_head。而（rdma_tail_idx - cached_rdma_channel_head）是节点lane_id的local_rank nvl_rank的rdma缓冲区
            中针对(channel, 节点) (channel_id, 当前节点) 的环形队列中还没有被消费的token数量。
            
            自旋退出条件: 只有当这个数量小于num_max_rdma_chunked_recv_tokens的时候，说明节点lane_id的local_rank nvl_rank的环形队列中的token数据被消费到了
                        剩下没有被消费的数量小于num_max_rdma_chunked_recv_tokens。
                        因此，此时节点lane_id的local_rank nvl_rank的rdma缓冲区中 针对当前节点的local_rank nvl_rank的环形队列中 至少有一个token空余空间，
                        而rdma_channel_head.buffer(lane_id)是原子累加的，也就是说即使**有且仅有**一个token空余空间，那也一定是token token_idx要写入的token空余空间，
                        所以只要满足这个条件，就一定可以开始通过ibdga转发token token_idx。于是就可以退出下面的while循环。
            */
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

            // 当前线程执行到这里时，接收端的环形队列中至少有一个token空余空间，可以写入token token_idx。

            // Store RDMA head for combine
            if (lane_id < kNumRDMARanks and not kCachedMode)
                /*
                send_rdma_head[token_idx * kNumRDMARanks + lane_id]: 表示当前rank的token token_idx要发送到节点lane_id的
                    local_rank nvl_rank的RDMA缓冲区环形队列的head索引，用于combine阶段查询token在缓冲区中的位置。
                    由于是在combine阶段消费，所以这里命名为head。
                */
                send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

            // Broadcast tails
            SourceMeta src_meta;
            // topk_ranks[num_topk_ranks]: 表示token token_idx要发往的第num_topk_ranks个专家所在的节点编号。
            int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
            /*
            经过设置后，dst_send_buffers[num_topk_ranks] 指向token token_idx要发往的第num_topk_ranks个专家所在的rank
            在当前rank的rdma发送缓冲区中的对应token位置。
            经过 st_broadcast 的执行，会把 x 中的token token_idx写入到当前rank的rdma发送缓冲区中。
            */
            void* dst_send_buffers[kNumTopkRDMARanks];
            #pragma unroll
            /* 
            注意: 这里之所以要写for循环遍历kNumRDMARanks个节点，是因为每个token可能发往多个kNumRDMARanks节点，而不是只发往一个节点。
            通过__shfl_sync循环遍历token token_idx是否往kNumRDMARanks个节点的local_rank nvl_rank的RDMA缓冲区中发送。
            下面的for循环的目的就是记录: dst_send_buffers[num_topk_ranks]
            */
            for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
                /* 
                rdma_tail_idx: 表示token token_idx要经由channel channel_id发送到节点lane_id的local_rank nvl_rank的RDMA缓冲区环形队列的tail索引。
                               不发送就是 -1，所以这里判断是否>=0。
                */
                if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {  // 如果满足，说明要发往节点 i。
                    // slot_idx: 表示token token_idx要经由channel channel_id发送到节点 i 的local_rank nvl_rank的RDMA缓冲区环形队列的tail索引。
                    slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                    // topk_ranks[num_topk_ranks]: 表示token token_idx要发往的第num_topk_ranks个专家所在的节点是节点 i 。
                    topk_ranks[num_topk_ranks] = i;
                    // 得到token token_idx是否发往节点lane_id的 8 个local_rank的布尔值。
                    auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
                    auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
                    /*
                    注意: 下面的 lane_id 可不是当作节点id来理解的，而是当作token token_idx要发往的“第 lane_id 个节点”来理解的。
                    为什么要这些写？
                    因为token token_idx要发往最多不超过 kNumTopkRDMARanks 个节点，当前的for循环是遍历所有可能的节点，在每轮遍历中如果遍历到的节点 i
                    是需要发往的节点，那么 “token token_idx要发往节点 i ” 这一SourceMeta元信息就只用创建一次，而不是每个线程都创建一次。
                    在并行编程中，如果某个数据只需要创建一次，就一定要从高层次到低层次划分好如何使得这个数据只创建一次。
                    而当前的任务在高层次已经通过:
                    1、sm对应channel；
                    2、channel对应token范围，也就是 sm 也对应token范围；
                    3、部分warp对应sm中的不同 WarpRole；
                    4、具体单个warp对应一批token；
                    5、warp中的单个lane可以对应到“唯一性的逻辑概念”，这是因为代码就是一定在lane上运行的，一定和lane一一对应。
                         这个“逻辑概念”可以是某个任务、某个对象、某个属性或某个状态。
                         比如: 单个lane可以对应: 某个节点、某个要发往的节点、token量化的某个scale、要发往的某个节点中的某个专家。
                    这样就能使得想要保持唯一性的逻辑概念和lane一一对应。
                    因此，凡是cuda程序想要实现唯一性的逻辑概念，就一定要对应到唯一的lane上，lane可以不用完，也可以循环步进。
                    */
                    if (lane_id == num_topk_ranks)
                        /*
                        rdma_rank: 表示token token_idx是从当前节点上发出的。这是用于在dispatch阶段token发送到dispatch接收端之后，
                                   在combine阶段确定这个token要原路返回发往哪个节点。
                        注意: 为什么SourceMeta中不需要记录token token_idx具体是来自哪个local_rank？
                        这里很容易猜到: 
                        假设有4个节点0、1、2、3，每个节点有8个rank。现在要从节点1的local_rank 2发往节点3的local_rank 6的token。
                        之所以SourceMeta中不需要记录token token_idx具体是来自哪个local_rank，是因为combine发送端节点3的local_rank 6
                        知道这个token是来自节点3的local_rank 2的rdma缓冲区，就发送给节点3的local_rank 2的rdma发送缓冲区，然后节点3的local_rank 2
                        就只能通过ibgda往节点1的local_rank 2的rdma接收缓冲区发送这个token。
                        */
                        src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
                    /* 
                    send_buffer: 表示token token_idx要发往节点 land_id 的当前rank的rdma发送缓冲区的地址。
                    broadcast(send_buffer, i): 类型就是uint8_t*。表示token token_idx要发往节点 i 的当前rank的rdma发送缓冲区的地址。
                                               这里作者的类型转换reinterpret_cast<uint8_t*>是冗余的，但语义明确。

                    设置dst_send_buffers指向rdma对称内存缓冲区。
                    broadcast(send_buffer, i): send_buffer传入broadcast函数时，传入的是64位的地址值。因为broadcast的第一个参数的类型是dtype_t&，
                    而地址传给函数的引用类型的形参时，传的就是这个地址值本身，而在cuda和C++中地址都是64位的。
                    dst_send_buffers[num_topk_ranks++]: 表示token token_idx要发往的第num_topk_ranks个专家所在的rank在当前rank的rdma发送缓冲区中的对应token位置。
                    */
                    dst_send_buffers[num_topk_ranks++] =
                        reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_token;
                }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);

            /*
            接下来开始写数据到rdma发送缓冲区中。
            每个token的数据布局是: hidden数据 + scale数据 + SourceMeta数据 + topk_idx数据 + topk_weights数据。即:
            hidden_int4 * sizeof(int4) + num_scales * sizeof(float) + sizeof(SourceMeta) +
                num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float)
            */


            // Copy `x` into symmetric send buffer
            /*
            key: 表示在UNROLLED_WARP_COPY中要传入的 hidden_int4 的第几个 int4 元素。加上存储地址就是这个int4的存储地址。
            value: 表示token token_idx的第key个int4在寄存器中的值的引用。用这个引用就可以不需要在st_broadcast函数中创建函数的形参寄存器，直接使用value值原本所在的寄存器中的值。
            */
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                /* 
                有了下面的for循环，可以使得只调用一次UNROLLED_WARP_COPY就将token token_idx的hidden数据写入到 num_topk_ranks 个缓冲区的对应位置。
                而只用在UNROLLED_WARP_COPY中完整读取一次hidden数据。也就是说在进行num_topk_ranks次 global_memory -> global_memory的数据复制时候，
                依然可以发挥循环展开优势，并且只用读取一次hidden数据。
                */ 
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            // 第四个参数的目的地址无意义了，因为目的地址在st_broadcast中。之前我只纳闷为什么需要传最后的两个读取和写入函数作为参数，现在明白了。
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                // 向后移动 hidden_int4 个int4，就是移动到scale数据的位置。注意: 每个lane都维护了一个dst_send_buffers数组。
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
                // 向后移动 num_scales 个float，就是移动到存储SourceMeta数据的位置。
                dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;

            // Copy source metadata into symmetric send buffer
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                // 向后移动 sizeof(SourceMeta) 个字节，就是移动到topk_idx和topk_weights数据的位置。
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

            // Copy `topk_idx` and `topk_weights` into symmetric send buffer
            #pragma unroll
            /*
            下面这个for循环是说要把每个token的所有路由的专家编号和对应的权重都写入到每个接收端节点中。
            TODO 这里有个问题: 并不是每个节点都需要这个token的所有的路由专家编号和对应的权重的，这样全部写入是否有些冗余呢？
            是这样吗: 没办法，这也是为了内存对其，否则真的没有什么好办法啊。
            */
            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
                /*
                rank_idx: 表示token token_idx要发往的第rank_idx个节点。
                copy_idx: 表示token token_idx路由到的第copy_idx个全局expert。不是专家编号。
                */
                auto rank_idx = i / num_topk, copy_idx = i % num_topk;
                // idx_value: 表示token token_idx路由到的第copy_idx个全局expert的全局expert编号。
                auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
                // weight_value: 表示token token_idx路由到的第copy_idx个全局expert的权重。
                auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
                // 存储专家编号
                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
                // num_topk个专家编号都连续存储在一起，num_topk个权重也连续存储在一起。
                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);
            }
            __syncwarp();

            // Release the transaction in the window
            /*
            下面这个if中的所有代码，在具体业务层面，都是为了更新 rdma_send_channel_tail[lane_id]。
            rdma_send_channel_tail[lane_id]、rdma_send_channel_window[lane_id]、rdma_send_channel_lock[lane_id]:
                这三个变量都是对应当前rank的针对目标节点 lane_id的rdma发送缓冲区的。
            而对rdma_send_channel_window[lane_id]的读写，是为了保证更新rdma_send_channel_tail[lane_id]时可以批量推进tail。
            而对rdma_send_channel_lock[lane_id]的读写，是为了保证更新rdma_send_channel_tail[lane_id]时可以互斥地进行。也就是用 “acquire-release 配对”来实现互斥锁。
            */
            if (is_token_in_rank_uint64 != 0) {  // 如果满足，说明这个token要发往节点lane_id。
                // Acquire lock first
                /*
                获取互斥锁（mutex）的时候，可能要等很久，得等到别的线程释放这个锁才行。7个warp的lane lane_id都可能竞争这个锁。
                */
                acquire_lock(rdma_send_channel_lock + lane_id);
                /*
                rdma_send_channel_tail[lane_id]: 表示针对目标节点 lane_id，当前 channel 下，已经被 kRDMASender warps「释放」掉的最大 tail 下标（不含）。
                */
                auto latest_tail = rdma_send_channel_tail[lane_id];
                auto offset = rdma_tail_idx - latest_tail;
                /*
                这个 32 就是rdma_send_channel_window的每个元素是32位的uint32_t类型的这个32的含义。也就是对应每个节点的滑动窗口大小是32
                一次最多允许「未释放」的 slot 在 tail 前面不超过 32 个，如果超过，一直减少offset，直到 offset < 32。
                这个32既不是一个warp中线程数量，也不是有相同lane_id的lane有多少个，而是滑动窗口的大小。
                */
                while (offset >= 32) {
                    /*
                    这里为什么需要先释放锁，再获取锁？
                    这是为了给别的想要更新rdma_send_channel_tail[lane_id]的warp的lane lane_id 一个机会，
                    让你们写一下，我看看你们新写入的rdma_send_channel_tail[lane_id]是多少。
                    当前lane如果不给别的lane机会，别的lane就无法更新rdma_send_channel_tail[lane_id]，造成死锁。
                    */
                    release_lock(rdma_send_channel_lock + lane_id);
                    acquire_lock(rdma_send_channel_lock + lane_id);
                    latest_tail = rdma_send_channel_tail[lane_id];
                    offset = rdma_tail_idx - latest_tail;
                }

                /* 
                离开while循环时，表示当前线程写入的token token_idx在以latest_tail为起点、
                连续32个逻辑slot（即 latest_tail, latest_tail+1, ..., latest_tail+31）的滑动窗口中了。
                
                注意: while退出时，当前lane还拿着锁，在拿锁到释放锁的这段代码中，整个操作序列具有原子性。
                      当然了，是因为这段代码在业务逻辑层面需要原子性，所以才锁住这段代码。
                这段具有原子性的代码序列包括: 
                读tail，计算offset，读window，计算从bit 0 起连续多少个1，存储推进tail，右移window并存储。
                */

                // Release the transaction slot
                // Add the bit and move the ones if possible
                /*
                注意: 多个warp的相同lane可能同时更新同一个 window（同一目标rank）
                在当前位图rdma_send_channel_window[lane_id]上，把第offset位 置1。
                然后读取到新的位图window，但是window还没有写入共享内存。
                */
                auto window = rdma_send_channel_window[lane_id] | (1u << offset);
                /*
                表示token token_idx写入的是latest_tail 位置的slot，此时才有向前推进tail的必要。
                否则如果写入的是window中别的位置的slot，则会因为latest_tail 位置的slot没有被释放而无法向前推进任何长度的slot，
                所以不需要向前推进tail。
                */
                if (offset == 0) {
                    /*
                    ~window: 表示按位取反。
                    (~window) == 0: 表示window全为1。也就是当前窗口的所有slot都已经被释放。
                    __ffs(x): 返回“x 中最低位 1 是从右数第几位”（第几位是从1开始数）。
                              若对 0 调用 __ffs，CUDA 规定 __ffs(0) 返回 0，所以这里需要特殊处理。
                    (__ffs(~window) - 1): __ffs(x) 返回“x 中最低位 1 是从右数第几位”（第几位是从1开始数）。
                                          所以 __ffs(~window) 是“~window 从 bit 0 起，第一个为 1 的位”的位置（第几位是从1开始数）。
                                          减 1 得到 “window 从 bit 0 起连续 1 的个数”（因为第一个 0 出现在该下标）。
                    例如，当offset == 0时，
                    rdma_send_channel_window[lane_id]:
                            = 0b00000000000000000000000000010110
                     window = 0b00000000000000000000000000010111, 则 
                    ~window = 0b11111111111111111111111111101000, 则 
                    __ffs(~window) = 4, num_empty_slots = 3，也就是从bit 0 起连续3个1。

                    num_empty_slots: 表示从bit 0 起连续多少个1。
                    */
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;
                    /*
                    注意: 当前lane lane_id还在所有的kNumDispatchRDMASenderWarps 个kRDMASender warp中拥有锁，也就是上面while循环中拥有的锁。
                          所以这里可以安全地更新rdma_send_channel_tail[lane_id]。把latest_tail推进num_empty_slots。

                    rdma_send_channel_tail[lane_id]: 在这里写是要应用于kRDMASenderCoordinator warp中读取已经处理完了的可以开启ibgda远程发送的token的tail位置的。
                    release语义: 保证在此存储之前的所有内存操作（主要是写入send buffer的token数据）对其他线程可见后，再进行本次tail的更新。
                                这样其他warp就能看到最新的tail值和对应的token数据。
                    归根结底就是为了保证: 当前程序用st_release_cta更新tail值之后，必定前面代码对send buffer中的token数据写入操作已经完成并且在CTA层面可见了。
                                   使得这里可以安全地先写send buffer，再写tail。
                    */
                    st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);
                    window >>= num_empty_slots;
                    /* 
                    注意: 所谓“移位”，是指“框（硬件电路中的bit位存储单元）”不动，而“框中的内容（也就是bit位的状态）”往左或往右移动。
                         千万不要理解为是“框”在动，那样就把左移右移搞反了。
                    window: 0b00000000000000000000000000010111  右移之后变成:
                            0b00000000000000000000000000000010
                    */
                }
                rdma_send_channel_window[lane_id] = window;

                // Release lock
                release_lock(rdma_send_channel_lock + lane_id);
            }
            __syncwarp();
        }
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        /*
        一句话总结: 只有一个warp执行。把当前rank的rdma_channel_data.send_buffer中的token数据通过IBGDA发送到
                  接收端的rdma_channel_data.recv_buffer。
        详细实现:
        1、清理共享内存。有barrier可以保证这里清理共享内存后kRDMASender warp才可以写入。
        2、获取要往每个节点要发送的token数量。
        3、循环发送token数据到每个节点中的与当前rank相同local_rank的rank。
        4、更新远程rank的rdma缓冲区的tail指针rdma_channel_tail。
           生产者才有资格更新tail（当前warp作为将自己rdma缓冲区中的数据发送到远程rank的rdma缓冲区的生产者）。
        注意: 当前warp跨节点发送时，并不需要用到glb_channel__prefix_matrix。
        */

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
        /*
        num_tokens_to_send: 表示当前rank作为dispatch发送端从channel channel_id要发送到节点 lane_id的即时的、尚未发送的token数量。
        */
        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            /* 
            rdma_channel_prefix_matrix[lane_id * num_channels + channel_id]: 表示当前rank作为dispatch发送端从channel 0到
            channel channel_id（包含）累计要发送到节点lane_id的token数量之和（r2n的channel级别前缀和）。
           */
            num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
            if (channel_id > 0)  // channel 0之前的token数还是0，就不需要下面的减法操作了。
                // 执行下面这行代码之后，num_tokens_to_send 表示当前rank作为dispatch发送端从channel channel_id要发送到节点lane_id的token数量。
                num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
        }

        // Iterate all RDMA ranks
        /*
        last_issued_tail: 表示当前rank作为dispatch发送端上一次通过channel channel_id发送到节点 lane_id的token的最大tail（不含），即"上一次已发出"的进度。
        */
        int last_issued_tail = 0;
        auto start_time = clock64();
        
        /* 
        每次都要执行判断: 只要当前rank至少要往任意一个节点通过channel channel_id发送token数据，就可以进入循环体，否则直接退出循环。
        num_tokens_to_send 的减少是在循环体里面执行。
        */
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
            /*
            之所以要写for循环，是因为使用ibgda传输时使用的nvshmemi_ibgda_put_nbi_warp是warp级的传输，
            需要每个lane都执行到那个调用的位置，否则在nvshmemi_ibgda_put_nbi_warp里面使用__shfl_sync时会造成死锁。
            因此for循环内部一开始用了好几处__shuf_sync，都是为了去对应这次要发送的目的地节点的lane上取值。
            注意: 一个warp中，如果只有部分lane执行了__shfl_sync(0xffffffff, ...)就会造成死锁，除非修改位图0xffffffff，使得只需要部分lane参与。
                 但是一般不会这么做，因为CUDA是机遇SIMT调度的，最小调度单位就是warp。
                 所以在调用__shfl_sync前要避免出现有lane_id参与的条件判断语句使得只有部分lane参与__shfl_sync。
            */
            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++i) {
                /*
                incast: 网络技术术语。在数据中心（IDC）网络中，Incast 指一种“同时多打一”的拥塞现象，即多个主机同时向单一目标发送数据包，
                        导致交换机缓冲区溢出、延迟抖动和丢包。该问题在确定性网络环境中尤为突出，解决方法包括通过随机延迟分散流量，避免并发扇入。
                
                为了缓解 incast 拥塞，通过shuffle操作，对不同 rank 和 channel 进行起始索引洗牌。
                即通过 (i + channel_id + rdma_rank) % kNumRDMARanks 来打乱目标 rank 的访问顺序，避免所有节点同时向同一目标发送数据造成拥塞。
                
                注意: 使用shuffle或者swizzle操作时，有几点要注意:
                1、所有要访问的实体都得在shuffle时均匀访问到。比如这里的 i 是可以遍历到所有的节点，并且是均匀变化的。
                2、真正能发挥shuffle操作优势的(channel_id + rdma_rank)须得尽量是当前gpu层次中具有唯一性的，也就是说这个shuffle偏移量的唯一性才会产生均匀的shuffle。
                3、shuffle后得到的目标实体可以是随机的，并且也不会危害整体的性能，各个目标实体间具有同等重要性。
                4、不同rank、不同chennel，都可能在循环中得到相同的dst_rdma_rank，但是channel_id和rdma_rank是分隔传输不同数据部分的保证。
                5、shuffle后，原本的cuda层次和业务逻辑层次的对应关系可能发生改变，因此后面的代码要注意到这一点。
                   比如，之前是用lane_id表示目标节点，在使用shuffle后，就要用dst_rdma_rank表示目标节点。
                */
                // To mitigate incast congestion, shuffle the starting index of target rank for different ranks and channels
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                /* 
                之前是用lane_id表示目标节点，在使用shuffle后，就要用dst_rdma_rank表示目标节点。
                而当前rank要发往目标节点dst_rdma_rank的token数量需要从当前warp的lane dst_rdma_rank中获取。
                __shfl_sync 往往具有迷惑性。这里读取的是lane dst_rdma_rank的num_tokens_to_send，而不是当前lane lane_id的num_tokens_to_send。
                */
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);
                if (synced_num_tokens_to_send == 0)
                    continue;

                // Read the latest progress
                // NOTES: `rdma_send_channel_tail` does not need to be protected by lock
                /*
                ld_acquire_cta: acquire 读。保证在本次读之后的所有本 CTA 内内存访问，不会被重排到这次读之前，
                                从而能看见 kRDMASender 用 release 写 tail 时的全部更新。
                                acquire语义：保证在此加载之后的所有内存操作，不会重排到本次加载之前。
                                这样读取到tail值后，后续读取send buffer中的token数据时，能看到kRDMASender写入的完整数据。
                归根结底就是为了保证: 当前程序用ld_acquire_cta读取tail值之后，必定已经可以正确读取send buffer中的token数据了。
                                   使得这里可以安全地先读tail，再读send buffer。
                把 int* 转成 const int*，以匹配 ld_acquire_cta(const int* ptr) 的签名，表示这里只读。
                __shfl_sync(0xffffffff, ld_acquire_cta(...), 0): 这样写是为了只让warp中的lane 0 执行 ld_acquire_cta(...) 得到 processed_tail，
                                                                 再通过 __shfl_sync(..., 0)广播给所有lane，其他lane得到的是lane 0的processed_tail。
                                                                 这样可以避免多个lane同时执行 ld_acquire_cta(...)，从而避免竞态。
                
                */
                auto processed_tail = __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);
                /*
                synced_last_issued_tail: 表示当前warp针对目标节点 dst_rdma_rank，已经通过RDMA发出的token的最大tail（不含），即"已发出"的进度。
                而last_issued_tail是每个线程一份（但只有 lane_id == dst_rdma_rank 的线程会在后面更新它）。
                */
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);
                // num_tokens_processed: 表示已就绪且尚未发出的 token 数量（即"可被本轮发出"的token数）。
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                /*
                synced_num_tokens_to_send: 是业务逻辑层次剩余要发送的总的token数量。
                num_max_rdma_chunked_send_tokens: 是单次最多可以发送的token数量。
                num_tokens_processed: 是当前发现的累积了多少可以发送的token。想要发送只有两种情况:
                    1、要么就是累积最后一批的token数量达到了synced_num_tokens_to_send；
                    2、要么就是累积的token数量达到了num_max_rdma_chunked_send_tokens。
                    如果都不是，则继续等待kRDMASender warp的写入，并更新rdma_send_channel_tail + dst_rdma_rank。
                */
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
                    continue;

                // Issue RDMA send
                // 只要是上面我说的两种可以发送的情况之一。
                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
                /*
                问: 如果 num_max_rdma_chunked_recv_tokens 是10，num_max_rdma_chunked_send_tokens 是4，一共有11个token要发送，
                    那么在前两次每次发送了4个token之后，也就是发送了2 * 4 =8个token了，此时synced_last_issued_tail就是8，
                    dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens = 8，
                    num_tokens_processed:表示已就绪且尚未发出的 token 数量（即"可被本轮发出"的token数），假设值是3，
                    所以num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens) = 3，
                    所以dst_slot_idx + num_tokens_to_issue = 8 + 3 = 11，这岂不是大于num_max_rdma_chunked_recv_tokens（值是10），
                    这样岂不是断言判断为false。
                答: 前面有个assert: EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);
                    这个assert保证了不会出现“num_max_rdma_chunked_recv_tokens是10，num_max_rdma_chunked_send_tokens是4”这种情况。
                */
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);
                /*
                现在要把当前rank的rdma缓冲区的针对节点dst_rdma_rank的token数据发送到节点dst_rdma_rank的rank nvl_rank的rdma缓冲区。同节点不用传输。
                */
                if (dst_rdma_rank != rdma_rank) {
                    // 获得环形队列中的真实slot
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;
                    /*
                    1、由于NVSHMEM使用对称内存（symmetric memory），所有节点看到的是同一个全局地址空间。因此：
                       发送端（节点1的rank2）的send_buffer(dst_rdma_rank)和接收端（节点3的rank2）的
                       recv_buffer(src_rdma_rank)在逻辑上指向同一个内存区域；
                    2、元数据中的索引既可以说是接收端recv_buffer的索引，也可以说是发送端send_buffer的索引。

                    dst_ptr: 表示接收端节点dst_rdma_rank的rank nvl_rank的针对当前rank经由channel channel_id发送的token数据在
                             rdma对称内存缓冲区的接收部分的环形队列中的地址。
                    src_ptr: 表示当前rank作为发送端的针对接收端节点dst_rdma_rank的rank nvl_rank经由channel channel_id发送的token数据在
                             rdma对称内存缓冲区的发送部分的环形队列中的地址。
                    */
                    const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + 
                                                                    dst_slot_idx * num_bytes_per_token);
                    const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + 
                                                                    dst_slot_idx * num_bytes_per_token);
                    /* 
                    在internode_ll.cu中，nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, 
                                                                    lane_id, token_idx - offset);
                    warp级别的传输，所以每个lane都要执行到这个位置。
                    */
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,
                                                    src_ptr,
                                                    num_bytes_per_msg,
                                                    translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                    channel_id,
                                                    lane_id,
                                                    0);
                } else {
                    // Lighter fence for local RDMA rank
                    // 保证节点内的线程对内存的可见性。"fence.acq_rel.sys;"是一种轻量级内存屏障：确保所有之前的内存操作对同一个节点内的所有GPU的所有线程可见。
                    memory_fence();
                }
                __syncwarp();

                // Update tails
                // 每个lane都会在for循环中尝试遍历所有的节点去发送，如果不加上下面的条件判断，就会导致每个lane都去更新所有的节点的tail。
                // 有了这个条件判断，就使得每个lane只用负责一个目标节点的tail更新。
                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send -= num_tokens_to_issue;
                    /* 
                    更新节点dst_rdma_rank 的rank nvl_rank的rdma缓冲区中由当前节点的local_rank nvl_rank生产的token的最新的tail索引。
                    这样可以让节点dst_rdma_rank的rank nvl_rank从环形队列中消费到更多的token数据，因为tail索引增大了。
                    环形队列从tail写入，从head消费，消费时要保持head <= tail，而(tail - head)就是环形队列中还没有被消费的token数量。
                    lane_id == dst_rdma_rank == rdma_rank 表示当前节点是目标节点，这样在nvshmemi_ibgda_amo_nonfetch_add中就只需用atomicAdd即可，不需要IBGDA。

                    注意: 在对称内存构成的send和recv双缓冲区中，发送端发送了发送缓冲区中的数据之后，不需要更新发送缓冲区的tail索引，只用更新接收缓冲区的tail索引。
                    */
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
        /*
        一句话总结: 8个warp。主要负责把当前rank的rdma_channel_data.recv_buffer中的token数据通过NVLink发送到节点内的其他rank的nvl_channel_x中。
        详细实现:
        1、rdma2nvl等待之一: 等待rdma缓冲区的前缀和元数据准备好。记录远程rank和节点要发送到当前rank的token数量。
                           把各个远程节点要经由当前rank发送到当前节点的其他local_rank的r2r的channel级别前缀和记录到各个目的地local_rank的nvl缓冲区中。
        2、rdma2nvl等待之二: 等待nvl缓冲区有空闲的token空间。等到 “NVL缓冲区中剩余可用token空间” >= num_max_nvl_chunked_send_tokens。
        3、rdma2nvl等待之三: 等待发送端节点把数据发送到当前rank的rdma缓冲区。只要发现有来自节点src_rdma_rank的token数据，就可以退出等待循环开始转发。

        可以发现，上面的三次等待是从宏观到微观，从大前提到小前提，先等接收端再等发送端。这也是类似这种代码要注意的设计思想。
        然而，基于对称性，完成任务后进行通知的时候，这种“通知”的过程往往就要反过来。

        4、发送来自节点src_rdma_rank的rank nvl_rank的token数据到节点内的其他local_rank的nvl_channel_x中。
        5、更新共享内存中的head指针，即 forward_channel_head[dst_nvl_rank][src_rdma_rank] 记录的当前rank消费的来自
           节点src_rdma_rank的rank nvl_rank经由当前rank要最终发送给当前节点的rank dst_nvl_rank的token数据在当前rank的对应rdma缓冲区中的head索引。
           消费端才有资格更新head（当前warp消费了节点src_rdma_rank的rank nvl_rank的rdma缓冲区中的数据）。
        6、更新nvl缓冲区内存中的nvl_channel_tail。
           生产端才有资格更新tail（当前warp发送token数据给了当前rank相同节点的其他rank的nvl_channel_x缓冲区）。
        */

        // RDMA consumers and NVL producers
        const auto dst_nvl_rank = target_rank;  // target_rank = (warp_id + channel_id) % NUM_MAX_NVL_PEERS

        // Wait counters to arrive
        /*
        num_tokens_to_recv_from_rdma: 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel channel_id发送到当前节点的token数量。
        src_rdma_channel_prefix: 表示 "r2n的channel级别前缀和" 加上 "r2n的rank级别前缀和"。
        */
        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        if (lane_id < kNumRDMARanks) {
            /* 
            rdma2nvl等待之一: 等待别的节点和rank要发送给当前rank的token数量元数据准备好。包括下面两种:
                1、等待节点lane_id的rank dst_nvl_rank要发送给当前rank的token数量。
                2、等待节点lane_id要发送给当前rank的token数量。
            有了这个数据才知道当前rank的rdma缓冲区中在此次调用dispatch函数时会接收多少token。
            */
            while (true) {
                /*
                meta_0: 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel 0到channel (channel_id - 1)（包含）累计要发送到当前节点的rank target_rank的token数量之和（r2r的channel级别前缀和）。
                meta_1: 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel 0到channel channel_id（包含）      累计要发送到当前节点的rank target_rank的token数量之和（r2r的channel级别前缀和）。
                meta_2: 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel 0到channel (channel_id - 1)（包含）累计要发送到当前节点的token数量之和（r2n的channel级别前缀和）。
                meta_3: 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel 0到channel channel_id（包含）      累计要发送到当前节点的token数量之和（r2n的channel级别前缀和）。
                */
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                /*
                所有4个值都为负数表示前缀和元数据都已准备好（发送端已写入并完成传输）。
                编码方式: 发送端写入时使用 -value - 1（见kRDMASender warp中一开始的代码），接收端通过 -meta - 1 还原。这样编码是为了避免0表示的“没有发送”的含义。
                */
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    // Notify NVL ranks
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    /*
                    在kRDMAAndNVLForwarder warp中，nvl_channel_prefix_start 和 nvl_channel_prefix_end 存储在rank target_rank 的nvl缓冲区中。
                    因为对它们的轮询读是发生在rank target_rank中，轮询读在哪个rank发生，就存储在哪个rank的内存中。代码如下:
                    auto nvl_channel_prefix_start = AsymBuffer<int>(buffer_ptrs[target_rank], kNumRDMARanks, NUM_MAX_NVL_PEERS, 
                                                        channel_id, num_channels, nvl_rank).advance_also(buffer_ptrs[nvl_rank]);
                    也就是说nvl_channel_prefix_start是要记录 kNumRDMARanks 个节点的local_rank nvl_rank要经由channel channel_id发送给当前节点的rank target_rank的token数量。
                    
                    下面的两个前缀和可以计算出节点lane_id的rank nvl_rank要经由channel channel_id发送给当前节点的rank target_rank的token数量。
                    nvl_channel_prefix_start.buffer() + lane_id: 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel 0到channel (channel_id - 1)（包含）累计要发送到当前节点的rank target_rank的token数量之和（r2r的channel级别前缀和）。
                    nvl_channel_prefix_end.buffer() + lane_id  : 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel 0到channel channel_id（包含）      累计要发送到当前节点的rank target_rank的token数量之和（r2r的channel级别前缀和）。
                    */
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
                    st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    // Save RDMA channel received token count
                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    // num_tokens_to_recv_from_rdma: 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel channel_id发送到当前节点的token数量。
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    if (not kCachedMode)
                        recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
                    /*
                    recv_rdma_rank_prefix_sum[lane_id - 1]: 表示的是从节点 0的local_rank nvl_rank到节点 (lane_id - 1)的local_rank nvl_rank（包含节点 (lane_id - 1)的local_rank nvl_rank）
                                                            累计发送到当前rank的token数量之和（r2n的rank级别前缀和）。
                    src_rdma_channel_prefix: 表示 "r2n的channel级别前缀和" 加上 "r2n的rank级别前缀和"。
                    */
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
        /* 
        send_nvl_head: 表示当前rank的**所有**rdma缓冲区接收到的第token_idx个token转发到节点内rank dst_nvl_rank的nvl环形缓冲区的单调递增索引，
                       用于combine阶段查询token在nvl缓冲区中的位置。
        注意: send_nvl_head 针对的不是当前rank的某一个rdma接收缓冲区，而是当前rank的**所有**rdma接收缓冲区接收到的所有token。
             针对不同发送端节点的接收的token的排布是按照发送端节点的节点 id 的顺序排布的。
        将指针移动到当前处理的token范围的起始位置。
        
        注意: 当前rank的rdma缓冲区接收的所有token，就是要发送到同节点的所有rank的token。
        注意: 每个环形队列的head指针都是从这个环形队列的索引0开始单调递增的，所以某个节点经由channel channel_id发送到当前rank的rdma接收缓冲区环形队列中的索引是从 0 开始单调递增。
        
        在后面需要设置来自节点lane_id的channel channel_id的范围内的token在当前rank的rdma缓冲区中的head索引。
        src_rdma_head 和 src_rdma_tail 都是指当前rank的rdma接收缓冲区针对节点lane_id的channel channel_id的环形队列中的索引，范围是从 0 开始单调递增。
        src_rdma_head 到 src_rdma_tail 范围内的 token i（i 是从 0 开始单调递增）需要使用 send_nvl_head[i * NUM_MAX_NVL_PEERS]来表示。
        假如: 当前rank的**所有**rdma接收缓冲区一共要接收10个token，也就是deep_ep.cpp中定义send_nvl_head时的 num_rdma_recv_tokens = 10，那么 send_nvl_head的实际总长度是 10 * 8 = 80 个int
            假设节点数是 4，src_rdma_channel_prefix = 6，dst_nvl_rank = 3。那么要设置来自节点 lane_id的local_rank nvl_rank 要发送给当前节点的local_rank dst_nvl_rank的第 i 个token时，
            如何设置这个token在当前节点的local_rank dst_nvl_rank的nvl缓冲区中的逻辑索引head呢？一定要注意 i 是从 0 开始单调递增的。
            假设原本输入到dispatch函数中的send_nvl_head的指针是 orig_send_nvl_head。
            注意: src_rdma_channel_prefix: 表示 "r2n的channel级别前缀和" 加上 "r2n的rank级别前缀和"。
            在执行了下面的send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank; **之后**，那么在后面的“send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;”中:
            send_nvl_head[i * NUM_MAX_NVL_PEERS] = orig_send_nvl_head[src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank + i * NUM_MAX_NVL_PEERS] = 
                orig_send_nvl_head[(src_rdma_channel_prefix + i) * NUM_MAX_NVL_PEERS + dst_nvl_rank]
            于是:
            send_nvl_head[0 * NUM_MAX_NVL_PEERS] = orig_send_nvl_head[(6 + 0) * 8 + 3] = orig_send_nvl_head[51];
            send_nvl_head[1 * NUM_MAX_NVL_PEERS] = orig_send_nvl_head[(6 + 1) * 8 + 3] = orig_send_nvl_head[59];
            send_nvl_head[2 * NUM_MAX_NVL_PEERS] = orig_send_nvl_head[(6 + 2) * 8 + 3] = orig_send_nvl_head[67];
            send_nvl_head[3 * NUM_MAX_NVL_PEERS] = orig_send_nvl_head[(6 + 3) * 8 + 3] = orig_send_nvl_head[75];

            就是说只要send_nvl_head一开始偏移到了8个local_rank中的dst_nvl_rank，那么后面的 i 每次乘以 8 ，都会使得下标指向的含义还是对应dst_nvl_rank。
            因为send_nvl_head的 shape 是 (num_rdma_recv_tokens, NUM_MAX_NVL_PEERS)，是按照默认的dim顺序的维度优先排列内存的。
        
        可见，send_nvl_head是按照“先节点层次，然后channel层次，再然后token层次，最后是local_rank层次”的顺序从小到大排布的。
        */
        send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

        // Wait shared memory to be cleaned
        /*
        等待唯一一个kForwarderCoordinator warp清零共享内存 forward_channel_head 和 forward_channel_retired 。
        */
        sync_forwarder_smem();

        // Forward tokens from RDMA buffer
        // NOTES: always start from the local rank
        /*
        这里shuffle时以sm_id作为索引，是为了当前sm内的所有kRDMAAndNVLForwarder warp从同一个初始src_rdma_rank开始轮询，
        但是不同的sm中的kRDMAAndNVLForwarder warp却从不同的节点开始轮询，这样可以避免不同sm中的kRDMAAndNVLForwarder warp同时从同一个src_rdma_rank相关的地址读取数据，
        造成GPU多个读请求并发从同一个内存读取数据时造成资源竞争。
        虽然8个warp可能从同一个src_rdma_rank读取数据，但它们转发到不同的dst_nvl_rank（通过target_rank = (warp_id + channel_id) % NUM_MAX_NVL_PEERS计算），
        所以每个warp处理的是不同的数据流（不同的目标rank）。在while循环中，src_rdma_rank会通过round-robin方式变化，确保所有src_rdma_rank都能被处理。
        */
        int src_rdma_rank = sm_id % kNumRDMARanks;  // 这个sm_id必定是偶数，因为kRDMAAndNVLForwarder warp的sm_id都是偶数。
        int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
        int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
        // 准备从 RDMA 缓冲区转发数据到 NVL 缓冲区。
        /*
        只要有任意节点的local_rank nvl_rank要发送token到当前节点，就继续转发。 对num_tokens_to_recv_from_rdma的减少在循环内。
        */
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            // Check destination queue emptiness, or wait a buffer to be released
            start_time = clock64();
            /* 
            rdma2nvl等待之二: 等待目的地local_rank dst_nvl_rank的nvl缓冲区有空闲的token空间。
            等待nvl缓冲区有至少num_max_nvl_chunked_send_tokens个空闲的token空间。
            有了这个空间接下来才能把rdma缓冲区中的token数据转发到nvl缓冲区。
            */
            while (true) {
                // num_used_slots: 当前节点的local_rank dst_nvl_rank的nvl缓冲区中针对当前rank的rdma缓冲区的channel channel_id的环形队列当前已使用的 token slot数。
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                /*
                num_max_nvl_chunked_recv_tokens: 每个local_rank针对（channel, 本地rank）的nvl接收缓冲区的总token数容量
                num_max_nvl_chunked_recv_tokens - num_used_slots: nvl缓冲区中剩余可用空间（token 数）
                num_max_nvl_chunked_send_tokens: 每次从 RDMA 缓冲区转发到 NVL 缓冲区的最大 token 数（chunk 大小）
                当 NVL 缓冲区剩余空间 ≥ 一次rdma到nvl转发所需的 token 数时，退出等待循环，接下来开始转发。
                也就是local_rank dst_nvl_rank的nvl缓冲区已经有足够的空间可以容纳一次从当前rank的rdma缓冲区转发到nvl缓冲区的token数据。
                */
                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens)
                    break;
                /*
                nvl_channel_head.buffer(): 表示当前rank要经由channel channel_id发送到相同节点的rank dst_nvl_rank的token在
                                           相同节点的rank dst_nvl_rank的nvl缓冲区中的环形队列的head索引。
                
                只有lane 0去读取nvl_channel_head.buffer()，然后广播给所有的lane，防止所有的lane都去读取造成对内存的竞争。
                注意: 当前kRDMAAndNVLForwarder warp作为其他local_rank的nvl缓冲区的生产者，生产者才会轮询等待读head。而且这个head存储在当前rank的nvl缓冲区中。
                注意: 轮询等待读head，使得可以在kNVLReceivers warp中更新nvl_channel_head.buffer()时使用开销小的relaxed语义（只保证写入本身的原子性，不提供额外的同步保证）。

                使用volatile表明每次读取都从内存读取，不使用缓存值。
                读取head不需要使用acquire-release 模式。因为读取head时没有立即读取到最新的head也无所谓，只会导致发送端晚一点时间知道 
                “接收端已经把老head到新head之间的数据消费完了” 这件事（注意，是接收端消费发送端的缓冲区的数据）。
                那么发送端就晚一点时间发送数据到老head到新head之间的空间，仅此而已，不会造成数据错乱的影响。
                但是tail不行，如果没有acquire-release，可能导致接收端先看到 tail 更新，但读到的是旧数据（实际需要的新数据并没有真正写入）
                */
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
            /* 
            rdma2nvl等待之三: 等待发送端节点把数据发送到当前rank的rdma缓冲区。
            */
            while (true) {
                // 轮询下一个源 RDMA rank（round-robin）
                /*
                假设有4个节点0、1、2、3，每个节点有8个rank，这一共32个rank的全局rank id是0、1、2、...、31。而各个节点的内部的8个rank的nvl_rank都是0、1、2、...、7。
                现在要从节点1的rank2 （local_rank是2）发往当前节点3的rank6的token A，这是dispatch阶段，由下面4个dispatch步骤组成：
                dispatch步骤1: 将节点1的rank2的token A数据复制到节点1的rank2的rdma缓冲区中；
                dispatch步骤2: 将节点1的rank2的rdma缓冲区send_buffer中的token A数据发送到当前节点3的rank2的rdma缓冲区recv_buffer中，对称内存ibgda发送；
                dispatch步骤3: 由当前节点3的rank2从它的rdma缓冲区recv_buffer中把这些数据转发（forward）到当前节点3的rank6的nvl缓冲区中；
                dispatch步骤4: 当前节点3的rank6从它的nvl缓冲区中读取对应的经由当前节点3的rank2转发的来自节点1的rank2的token数据。

                在步骤3中，节点3的rank2有4个对应channel channel_id且分别对应4个节点的rdma缓冲区，这4个rdma缓冲区的数据都将写到节点3的rank6的
                对应channel channel_id且对应节点3的rank2的nvl缓冲区中，那么就会出现4个rdma缓冲区**并行**往同一个nvl缓冲区中写数据的情况。
                为了解决这个问题，作者设置每个sm值对应一个channel，这样不同的sm处理的数据所在的rdma缓冲区和nvl缓冲区就隔离开了。
                另外，在同一个sm中，作者用src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;轮流处理节点3的rank2的4个对应channel channel_id
                且分别对应4个节点的rdma缓冲区的数据，所以避免出现4个rdma缓冲区并行往同一个nvl缓冲区中写数据的情况。
                
                sm只对应到一个channel: 不同 sm 负责不同 channel，避免不同 channel 的 rdma/nvl 混在一起。
                同一sm内 round-robin src_rdma_rank: 保证同一时刻只有一个 src_rdma_rank 在被消费并写入各个 dst_nvl_rank 的 nvl 环形队列，
                从而不会出现 4 个 rdma 缓冲区并行往同一个 nvl 环形队列写。
                后面的 src_rdma_head 和 src_rdma_tail 的由来也是出于这个原因。

                注意: 这样还有个好处: 在同一个nvl缓冲区中，来自同一个源节点的经由相同channel发送的token数据是紧邻存储的。
                */
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                /*
                num_tokens_to_recv_from_rdma: 表示节点lane_id的rank nvl_rank 要（经由当前rank）从channel channel_id发送到当前节点的token数量。
                而__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank): 
                    表示节点src_rdma_rank的rank nvl_rank 要（经由当前rank）从channel channel_id发送到当前节点的token数量。
                */
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    /*
                    lane_id == src_rdma_rank: 表示一个warp中，各个lane负责读取对应节点在当前rank的rdma缓冲区中的tail指针即可。
                        否则就会导致所有的lane都去读取所有的节点的，造成不必要的内存竞争读取。
                        当已经有了lane_id和src_rdma_rank的对应关系之后，各个lane想要读取别的节点的tail指针，只需通过__shfl_sync来读取即可。
                        当需要一个warp中的所有lane一起处理同一个节点的数据时，也是只需用__shfl_sync来广播即可。

                    cached_rdma_channel_head == cached_rdma_channel_tail: 表示当前rank临时消费完了来自节点src_rdma_rank发送到当前rank的rdma缓冲区中的token数据。
                    
                    满足上面两个条件，然后当前lane就得尝试读一下tail，看看节点src_rdma_rank是否有发送新的数据到当前rank的对应的rdma缓冲区。
                    */
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        /* 
                        rdma_channel_tail.buffer(src_rdma_rank): 表示当前rank的rdma接收缓冲区中由节点src_rdma_rank的local_rank nvl_rank
                            经由channel channel_id发送的token目前在环形队列的tail索引。
                            当前rank的rdma接收缓冲区，生产者是远程节点，消费者是当前warp（消费用以发往当前节点的其他local_rank）
                        还是那个原则: "哪个rank要轮询读某个指针，就把这个指针放在这个rank的内存上"。不要发生跨GPU甚至跨节点轮询读指针的情况。
                        还是那个原则: 涉及tail的，都用acquire-release语义，以保证代码有序、内存操作有序和内存可见性。
                        注意: 当前kRDMAAndNVLForwarder warp作为当前rank的rdma接收端缓冲区的消费者，消费者才会轮询等待读tail。而且这个tail存储在当前rank的rdma缓冲区中。
                        */
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
                    
                    // 只要发现有来自节点src_rdma_rank的token数据，就可以退出循环开始转发。
                    // 注意: 这里是只要有一个新的来自节点src_rdma_rank的local_rank nvl_rank的token数据，就可以退出循环开始转发。
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

            /* 
            从lane src_rdma_rank中广播读取当前rank接收的来自节点src_rdma_rank的token数据在当前rank的rdma缓冲区中的head和tail索引。
            当前rank的rdma接收缓冲区，生产者是远程节点，消费者是当前warp（消费用以发往当前节点的其他local_rank）。

            注意: 由 __shfl_sync 可知，每个warp中的所有lane都读取了相同的src_rdma_head和src_rdma_tail。
                 虽然8个kRDMAAndNVLForwarder warp可能从同一个src_rdma_rank读取数据，但它们转发到不同的dst_nvl_rank。
                 在for循环中，每个warp只会处理那些需要转发到自己的dst_nvl_rank的token（通过is_in_dst_nvl_rank判断）。
            */
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

            // Iterate over every token from the RDMA buffer
            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
                // 对环形队列容量求余，得到物理的slot索引。
                auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
                // src_rdma_head 和 src_rdma_tail 是当前rank的rdma接收缓冲区中的环形队列索引。
                auto shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;
                /*
                使用 ld.global.nc.L1::no_allocate 绕过 L1，直接从 L2 或全局内存读取，避免读取到 L1 中的脏数据。但可能稍微降低性能。
                每个token的数据布局是: hidden数据 + scale数据 + SourceMeta数据 + topk_idx数据 + topk_weights数据。即:
                    hidden_int4 * sizeof(int4) + num_scales * sizeof(float) + sizeof(SourceMeta) +
                    num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float)
                */ 
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
                /* 
                每个SourceMeta对象有且仅有两个字段: 
                src_rdma_rank:              int类型。                     表示token token_idx是从哪个节点上发出的。
                is_token_in_nvl_rank_bits:  int类型，32比特位图，实际只用8位。表示token token_idx是否要发送到的目标节点上的 8 个local_rank的布尔值。

                基于SourceMeta位图判断这个token i是否要发送到当前节点的local_rank dst_nvl_rank。
                */
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
                // 注意下面的if ！！！ TODO。保证了send_nvl_head[i * NUM_MAX_NVL_PEERS]是记录来自节点lane_id发送到当前rank的rdma缓冲区的。
                if (lane_id == src_rdma_rank) {
                    // 这里的 send_nvl_head 为什么命名为head？因为send_nvl_head是要在之后的combine阶段消费读取的。而消费都是从环形队列的head消费。
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                    /* 
                    rdma_nvl_token_idx: 记录当前rank的rdma接收缓冲区经由channel channel_id发送到当前节点的local_rank dst_nvl_rank的token数量。
                                        也就是token i在当前节点的local_rank dst_nvl_rank的nvl缓冲区中的环形队列的逻辑索引（从0开始单调递增）。
                    */
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;
                    if (not kCachedMode)
                        send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;
                }
                if (not is_in_dst_nvl_rank)  // 只处理需要接收的token
                    continue;

                // Get an empty slot
                /* 
                (cached_nvl_channel_tail++) 返回的是自增1之前的值（梦回大一）。tail指针记录的都是实际写入了的slot的后一个空的slot。
                dst_slot_idx: 表示token i在当前节点的local_rank dst_nvl_rank的nvl缓冲区中的环形队列的物理索引。
                */
                int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;
                // nvl_channel_x.buffer(): 表示和当前rank同节点的local_rank target_rank的nvl缓冲区接收的来自当前rank的rdma缓冲区经由channel channel_id发送的token的nvl缓冲区。
                // dst_shifted: token i要存到nvl_channel_x的具体地址。
                auto dst_shifted = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

                // Copy data
                if (elect_one_sync()) {
                    /*
                    把token i的整个数据从当前rank的rdma缓冲区中加载到tma共享内存缓冲区（tma缓冲区也只能在共享内存中）。
                    evict_first = false: 表示这个token i的数据在tma共享内存缓冲区中被消费（计算）完后，不会被优先从L2缓存中逐出。
                     */
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, num_bytes_per_token, false);
                    // Arrive_Count“加一”，Transaction_Count增加num_bytes_per_token。
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, num_bytes_per_token);
                }
                __syncwarp();
                // 等待“两个条件”都满足时，这个mbarrier才可以通过，才认为tma_load_1d任务完成。
                mbarrier_wait(tma_mbarrier, tma_phase);
                if (elect_one_sync())
                    // 把token i的整个数据从tma共享内存缓冲区中存储到当前节点的local_rank dst_nvl_rank的nvl缓冲区。
                    tma_store_1d(tma_buffer, dst_shifted, num_bytes_per_token);
                __syncwarp();

                // In case of insufficient NVL buffers, early stopping
                if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens)
                    /* 
                    注意for循环的执行顺序和判断条件，这里设置src_rdma_tail = i + 1之后，在下一轮循环之前会执行 i++，然后判断for循环条件，不满足i < src_rdma_tail，于是退出for循环。
                    为什么这里可以early stopping?
                    答: rdma2nvl等待之二: 等待nvl缓冲区有至少 num_max_nvl_chunked_send_tokens 个空闲的token空间。
                        每个kRDMAAndNVLForwarder warp与当前节点的各个rank是一一对应的。当前节点的local_rank dst_nvl_rank的nvl缓冲区从当前rank的rdma缓冲区中
                        经由channel channel_id发送的token当且仅当当前warp来转发。所以num_tokens_sent就表示的是这个nvl缓冲区中环形队列的接收数量。
                    */
                    src_rdma_tail = i + 1;

                // Wait TMA to be finished
                tma_store_wait<0>();
                __syncwarp();
            }

            // Sync head index
            if (lane_id == src_rdma_rank)
                /* 
                forward_channel_head[dst_nvl_rank][src_rdma_rank]: 表示当前rank的rdma接收缓冲区中接收的来自节点src_rdma_rank的local_rank nvl_rank的
                                                经由channel channel_id发送的token在当前rank的rdma**接收**缓冲区环形队列的head索引。
                
                注意: 每个kRDMAAndNVLForwarder warp和一个dst_nvl_rank一一对应。而这里的if (lane_id == src_rdma_rank)保证了每个lane和
                      一个节点src_rdma_rank一一对应。因此，在这里有且仅有一个线程给 forward_channel_head[dst_nvl_rank][src_rdma_rank] 写数据。
                
                重要: 双缓冲区的消费者: 
                     消费者在消费了自己的接收缓冲区中的数据之后，消费者会记录自己的接收缓冲区中的数据被后续的消费者消费到了的head（
                        当多个后续的消费者消费时还要记录多个后续的消费者消费各自消费到的head和根据这些head计算得到的min_head
                     ）。也就是说，只有当“消费者的后续消费者消费了消费者的接收缓冲区中的数据”之后，消费者才能据此更新生产者的发送缓冲区的head。
                而forward_channel_head就是记录多个后续的消费者消费各自消费到的head和这些head的min_head的。
                */
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

            // Move tail index
            __syncwarp();
            if (elect_one_sync())
                /*
                NVLink通信。
                nvl_channel_tail.buffer(): 表示当前rank的rdma接收缓冲区的经由channel channel_id发送的token的环形队列的tail索引。
                    该值存储在同节点的local_rank target_rank的nvl缓冲区中，因为同节点的local_rank target_rank在kNVLReceivers warp中作为消费者要自旋等待读取tail。
                同样地，生产端才有资格更新tail。当前warp往同节点的local_rank target_rank的nvl缓冲区中生产数据。
                注意: kNVLReceivers warp 作为目的地local_rank target_rank的nvl缓冲区的消费者，消费者才会轮询等待读tail，
                      于是这个tail存储在目的地local_rank target_rank的nvl缓冲区中。
                */
                st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
        }

        // Retired
        // 退出了上面最外层的while循环，说明现在已经没有任何发送端节点的local_rank nvl_rank要发送token到当前rank了。
        __syncwarp();
        if (elect_one_sync())
            // 表示当前rank已经不再需要把自己的rdma缓冲区中的数据经由channel channel_id往当前节点的local_rank dst_nvl_rank的nvl缓冲区中发送token数据了。
            // 注意: 退休的是warp（dst_nvl_rank）
            forward_channel_retired[dst_nvl_rank] = true;
    } else if (warp_role == WarpRole::kForwarderCoordinator) {
        /*
        一句话总结: 只有一个warp执行。主要负责更新源rank的rdma发送缓冲区中对应当前rank的head索引。
        详细实现:
        1、清理共享内存forward_channel_head和forward_channel_retired。
        2、更新源rank的rdma发送缓冲区中对应当前rank的head索引。注意批量更新和最小值保证。
           消费端才有资格更新head。
        */

        // Extra warps for forwarder coordinator should exit directly
        /*
        只有target_rank 0的kForwarderCoordinator warp会执行下面的操作
        warp_id = 8 → target_rank = 0
        */
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
            forward_channel_retired[lane_id] = false;  // 注意: 退休的是warp（dst_nvl_rank）
        
        // 只有一个warp执行。就只有warp_id = 8 的kForwarderCoordinator warp执行这个同步。
        sync_forwarder_smem();

        // kForwarderCoordinator warp在每个sm中只有一个，这个warp中的每个线程只负责一个远程节点target_rdma。
        int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
        // 轮询，是因为每次循环中forward_channel_head和forward_channel_retired的写操作是分散在8个kRDMAAndNVLForwarder warp中执行。
        while (true) {
            // Find minimum head
            int min_head = std::numeric_limits<int>::max();
            #pragma unroll
            /*
            __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
            __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];

            这里就体现了为什么要用 forward_channel_retired 和 forward_channel_head 这两个共享内存来实现对
                “当前rank的channel channel_id的接收的来自各个发送端节点的local_rank nvl_rank的rdma发送缓冲区被消费到的head索引”
            的更新。
            
            作者的实现:
                作者是把forward_channel_head定义成了二维数组，即记录每个kRDMAAndNVLForwarder warp（对应一个local_rank）中的每个lane（对应一个token源节点）
                    消费到了的当前rank的channel channel_id的环形队列中的索引。
                然后在唯一一个kForwarderCoordinator warp中去找到forward_channel_head[i][target_rdma]的最小的head，最小的head之前的token已经被转发到当前rank的同节点的其它local_rank的nvl缓冲区了。
                因此，上一个最小head（last_head）到当前这个最小的head（min_head）之间的token已经被完全彻底地落入到目的地rank的nvl缓冲区了，
                然后才去更新源节点的local_rank nvl_rank的rdma发送缓冲区中的head索引，将其向前推进 (min_head - last_head)。
            更简单但是涉及同步的实现方式: ❌
                把forward_channel_head定义为: __shared__ volatile int forward_channel_head[kNumRDMARanks];
                也就是说当前sm（channel）中每个kRDMAAndNVLForwarder warp只需要记录一个发送端节点的local_rank nvl_rank的rdma发送缓冲区的被当前sm消费到的head索引。
                但是这样需要同步，因为8个kRDMAAndNVLForwarder warp中的相同lane_id（对应相同src_rdma_rank）的线程需要竞争写 forward_channel_head[src_rdma_rank]。
                注意：在当前实现中，由于每个warp对应不同的dst_nvl_rank，且每个warp中只有lane_id == src_rdma_rank的线程会写，
                所以对于固定的(dst_nvl_rank, src_rdma_rank)组合，实际上只有一个线程会写，不存在竞争。
                这个同步可以放在kRDMAAndNVLForwarder warp的循环处理从src_rdma_head 到 src_rdma_tail之间的token的时候，在for循环退出之后执行对这8个warp的同步。
                因为退出for循环时，说明相同lane_id的lane都对节点src_rdma_rank发到当前rank的rdma缓冲区中的从 src_rdma_head 到 src_rdma_tail之间的token都处理完了。
                并且这些token肯定都已经发送到当前节点的需要最终接收的local_rank的nvl缓冲区了。既然到达了目的地rank的nvl缓冲区，那么就可以更新发送端节点的local_rank nvl_rank的rdma发送缓冲区中的head索引了。
            但是这种简单方式有个重要缺点:
                需要对这 8 个kRDMAAndNVLForwarder warp同步。8 个kRDMAAndNVLForwarder warp中的相同lane_id的lane只需要派出一个去写forward_channel_head[src_rdma_rank]即可。
                kRDMAAndNVLForwarder warp中有early stopping机制，也就是说不通的warp退出那个for循环的时间点不同，有的早，有的晚。同步代价大。
            注意: lane_id 对应src_rdma_rank(节点)，而 warp_id 对应dst_nvl_rank(local_rank)。early stopping机制是针对每个warp的。
                作者这种方式**好就好在**每一次每个warp退出for循环的时候，都会在kForwarderCoordinator warp的 while (true)循环中**有机会**推进min_head，
                使得kForwarderCoordinator warp可以更及时地推进发送端节点的local_rank nvl_rank的rdma发送缓冲区中的head索引。

            注意: forward_channel_head 和 rdma_send_channel_tail 都是针对多线程并行处理队列的滑动窗口问题:
                    - rdma_send_channel_tail: 多个kRDMASender warp并行生产，使用固定大小32的滑动窗口，需要互斥锁保护tail和window的更新
                    - forward_channel_head: 多个kRDMAAndNVLForwarder warp并行消费，每个warp处理不同的dst_nvl_rank，
                      对于固定的(dst_nvl_rank, src_rdma_rank)组合只有一个线程写，因此不需要锁。
                      kForwarderCoordinator通过找最小head来确保所有warp都处理完的token才能被释放。
                    这样的head和tail需要记录在共享内存中，因为需要sm内的线程协同读写。
            */
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                if (not forward_channel_retired[i])  // 注意: 退休的是warp（dst_nvl_rank）
                    min_head = min(min_head, forward_channel_head[i][target_rdma]);
            /*
            min_head == std::numeric_limits<int>::max()为true，说明当前lane上发现所有的forward_channel_retired[i]都为true，
            即:   表示当前rank已经不再需要把自己的rdma缓冲区中的数据经由channel channel_id往当前节点的任何local_rank的nvl缓冲区中发送token数据了。
            亦即: 表示当前sm的从rdma缓冲区转发数据到节点内的其它rank的任务彻底完成了，可以退出了。
            */
            if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
                // 此时说明当前sm已经完成了把当前rank的rdma缓冲区中的数据经由channel channel_id往当前节点的所有local_rank的nvl缓冲区中发送token数据的所有任务了。
                break;

            // Update remote head
            /*
            批量更新: 累积到阈值（num_max_rdma_chunked_send_tokens）才更新一次。
            最小值保证: 使用所有 kRDMAAndNVLForwarder warp 中的最小 head，确保远程节点不会过早释放仍在使用的缓冲区。
            注意: 这样的目的是将多次小更新合并为一次批量更新可以减少通信。
            */
            if (min_head != std::numeric_limits<int>::max() and 
                min_head >= last_head + num_max_rdma_chunked_send_tokens and
                lane_id < kNumRDMARanks) {
                /* 
                更新节点lane_id的rank nvl_rank的rdma发送缓冲区中被当前rank消费到的head索引，就是head索引推进(min_head - last_head)，
                该head索引存储在节点lane_id的rank nvl_rank的rdma发送缓冲区的rdma_channel_head.buffer(rdma_rank)中。
                使得节点lane_id的rank nvl_rank可以清空它的针对当前rank的rdma发送缓冲区中的已经被消费了的token数据空间，
                然后继续往环形队列中写入新的token数据，以便生产（传输）给当前节点的nvl_rank。

                channel_id + num_channels 表示当前PE选择的qp id，一个PE会用到num_rc_per_pe * num_devices_initialized个qp，
                最后根据这里传入的qp id选择qp时实际上是做了取模运算的，即(channel_id + num_channels) % (num_rc_per_pe * num_devices_initialized)，
                kRDMASenderCoordinator warp的最后调用nvshmemi_ibgda_amo_nonfetch_add时使用的qp id是 channel_id，
                而这里使用的qp的id是(channel_id + num_channels)可以和它尽量避开，实现负载均衡。

                lane_id == rdma_rank 表示当前节点是目标节点，这样在nvshmemi_ibgda_amo_nonfetch_add中就只需用atomicAdd即可，不需要IBGDA。

                重要:
                在双缓冲区中：
                生产者:
                    生产者把自己的发送缓冲区中的数据写到消费者的接收缓冲区之后，生产者就会推进消费者的接收缓冲区的tail。
                    生产者并不需要管自己的发送缓冲区的tail，只有往发送缓冲区写数据的生产者才会推进这个发送缓冲区的tail。
                消费者:
                    消费者在消费了自己的接收缓冲区中的数据之后，消费者会记录自己的接收缓冲区中的数据被后续的消费者消费到了的head（
                        当多个后续的消费者消费时还要记录多个后续的消费者消费各自消费到的head和根据这些head计算得到的min_head
                    ）,也就是说，只有当“消费者的后续消费者消费了消费者的接收缓冲区中的数据”之后，
                    消费者才能据此更新生产者的发送缓冲区的head。
                */
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                min_head - last_head,
                                                translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank),
                                                channel_id + num_channels,
                                                lane_id == rdma_rank);
                last_head = min_head;
            }

            // Nanosleep and let other warps work
            /*
            让当前 warp 休眠指定的纳秒数。NUM_WAIT_NANOSECONDS = 500纳秒（0.5微秒）
            
            为什么需要休眠？
            虽然kForwarderCoordinator和kRDMAAndNVLForwarder是不同的warp，但它们都在同一个SM上，共享SM的执行资源：
            - 执行单元（ALU）：数量有限，同时执行的warp数量受限于可用执行单元
            - 调度器槽位：虽然SM可以驻留很多warp，但实际同时执行的warp数量有限
            - 指令发射带宽：有限的指令发射能力
            
            如果没有__nanosleep，kForwarderCoordinator的while循环会变成忙等待（busy waiting），
            一直占用执行单元做无意义的检查，即使forward_channel_head还没有更新。
            这会影响同SM上的kRDMAAndNVLForwarder warp执行实际的数据转发工作。
            
            通过__nanosleep，kForwarderCoordinator主动让出执行资源，让调度器可以将执行时间分配给其他warp。
            500纳秒足够短，能及时响应更新；足够长，让其他warp有机会执行实际工作。
            */
            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
    } else {  // kNVLReceivers
        /*
        一句话总结: 主要负责将当前节点的nvl缓冲区nvl_channel_x中的数据写到输出tensor recv_x中。
        详细实现:
        1、写recv_gbl_channel_prefix_matrix: 记录token数据在接收端输出数组recv_x中的给全局rank (lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank)
           的channel channel_id准备的空间的起始索引。它的意义是在combine的时候能将传给combine的token数据正确地写入到cimbine的输出tensor的对应位置中。
        2、将当前rank的nvl缓冲区nvl_channel_x中的token相关数据写到dispatch函数的输出中: hidden、topk_idx、topk_weights、meta。
        3、更新nvl_channel_head.buffer(): 表示当前rank的nvl缓冲区中缓存来自于同节点的local_rank target_rank的
           经由 channel channel_id发送的token的环形队列的head索引。
           消费端才有资格更新head。
        */

        // NVL consumers
        // Retrieve rank offset from barrier results (each lane's register stores an RDMA rank)
        /* 
        target_rank: (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS 
        target_rank 是给当前rank发送自己的rdma接收缓冲区的数据的local_rank。
        */
        int src_nvl_rank = target_rank, total_offset = 0;
        // 当前 rank 托管的专家的全局编号范围是: [local_expert_begin, local_expert_end]
        const int local_expert_begin = rank * (num_experts / num_ranks);
        const int local_expert_end = local_expert_begin + (num_experts / num_ranks);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            // total_offset 表示从全局的rank 0到节点lane_id的rank (src_nvl_rank - 1)累计要发送给当前rank的token数量之和（r2r的全局rank级别前缀和，channel无关）。
            total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

        // Receive channel offsets
        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
        auto start_time = clock64();
        /*
        nvl2recv等待之一: 等待r2r的channel级别前缀和元数据准备好。
        等待节点lane_id的rank src_nvl_rank要发送给最终目的地rank的token数量的r2r的channel级别前缀和。
        */
        while (lane_id < kNumRDMARanks) {
            /*
            在kNVLReceivers warp中，nvl_channel_prefix_start 和 nvl_channel_prefix_end 需要轮询读，所以存储在当前rank 的nvl缓冲区中。
            auto nvl_channel_prefix_start = AsymBuffer<int>(buffer_ptrs[nvl_rank], kNumRDMARanks, NUM_MAX_NVL_PEERS, 
                                                channel_id, num_channels, target_rank).advance_also(buffer_ptrs[target_rank]);
            
            nvl_channel_prefix_start 和 nvl_channel_prefix_end 的写入是在kRDMAAndNVLForwarder warp中执行的。
            start_offset 表示节点lane_id的rank src_nvl_rank 要（经由当前节点的rank src_nvl_rank）从channel 0到channel (channel_id - 1)（包含）累计要发送到当前rank的token数量之和（r2r的channel级别前缀和）。
            end_offset   表示节点lane_id的rank src_nvl_rank 要（经由当前节点的rank src_nvl_rank）从channel 0到channel channel_id（包含）      累计要发送到当前rank的token数量之和（r2r的channel级别前缀和）。
            */
            start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
            end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
            // 下面的if为true，则说明节点lane_id的rank src_nvl_rank 要发送给当前rank的token数量元数据已准备好。
            if (start_offset < 0 and end_offset < 0) {
                start_offset = -start_offset - 1, end_offset = -end_offset - 1;
                /* 
                r2r的全局rank级别前缀和，加上channel (channel_id - 1)的r2r的channel级别前缀和，得到:
                从全局的rank 0到节点lane_id的rank src_nvl_rank的从channel 0到channel (channel_id - 1)（包含）累计要发送给最终目的地rank（当前rank）的token数量之和。
                */
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

        /*
        上面的while循环能退出，说明现在的start_offset和end_offset已经不是“取负减一”编码了，而是正常的数值。
        二者相减就是: 节点lane_id的rank src_nvl_rank 要经由channel channel_id发送给当前rank的token数量。
        这里是把kNumRDMARanks个节点的所有local_rank src_nvl_rank要经由channel channel_id发送给当前rank的token数量加起来，得到:
        num_tokens_to_recv: 当前rank要从同节点的local_rank target_rank的rdma接收缓冲区经由channel channel_id接收的token总数量。
        */
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

        // Save for combine usage
        if (lane_id < kNumRDMARanks and not kCachedMode)
            // 记录token数据在接收端输出数组recv_x中的给全局rank (lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank)的channel channel_id准备的空间的起始索引。
            // recv_gbl_channel_prefix_matrix 的意义是在combine的时候能将传给combine的token数据正确地写入到cimbine的输出tensor的对应位置中。
            recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
        __syncwarp();

        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) {  // 对num_tokens_to_recv的减少在while循环的内部
            // Check channel status by lane 0
            start_time = clock64();
            while (true) {
                // Ready to copy
                if (cached_channel_head_idx != cached_channel_tail_idx)  // 环形队列的head != tail，说明有新数据。而相等就是没有数据。
                    break;
                /*
                nvl_channel_tail.buffer() 表示同节点的local_rank src_nvl_rank的rdma接收缓冲区经由channel channel_id发送到当前rank的nvl缓冲区环形队列的token的tail索引。
                注意: 虽然nvl_channel_tail.buffer()是存的别的rank上的环形队列的tail，但是nvl_channel_tail.buffer()本身却是存储在当前rank的nvl缓冲区中的。
                     这是因为当前rank要对其进行自旋等待读取。
                */
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
                // 环形队列的逻辑head对队列容量求余变成物理head，然后逻辑head推进一位。
                int token_idx_in_buffer = (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;
                // nvl_channel_x: 当前rank的nvl缓冲区中缓存的来自于同节点的local_rank src_nvl_rank的rdma接收缓冲区的经由channel channel_id发送的token数据。
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;
                /*
                每个token的数据布局是: hidden数据 + scale数据 + SourceMeta数据 + topk_idx数据 + topk_weights数据。即:
                    hidden_int4 * sizeof(int4) + num_scales * sizeof(float) + sizeof(SourceMeta) +
                    num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float)

                每个SourceMeta对象有且仅有两个字段: 
                src_rdma_rank:              int类型。                     表示token token_idx是从哪个节点上发出的。
                is_token_in_nvl_rank_bits:  int类型，32比特位图，实际只用8位。表示token token_idx是否要发送到的目标节点上的 8 个local_rank的布尔值。

                基于SourceMeta位图判断这个token i是否要发送到当前节点的local_rank dst_nvl_rank。
                */
                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                /* 
                recv_token_idx: 表示从全局的rank 0到节点meta.src_rdma_rank的rank src_nvl_rank的从channel 0到channel (channel_id - 1)（包含）累计要
                                发送给最终目的地rank（当前rank）的token数量之和，加上当前已经写入了的token数。
                                这个“和”也就是dispatch输出tensor recv_x中，要写入当前token的位置。
                */
                int64_t recv_token_idx = __shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank);
                // (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;
                /* 
                在设置target_rank的时候，就已经设置好了每个warp对应同节点的一个local_rank。
                每个warp在处理同节点的local_rank src_nvl_rank的rdma接收缓冲区的数据时，会面临来自各个源节点的local_rank src_nvl_rank的token。
                而每个lane对应一个源节点，为了使得同warp的其他lane能通过__shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank)得到源节点meta.src_rdma_rank
                的local_rank src_nvl_rank发送到当前rank的recv_x中的位置。
                注意: 能使用"每个lane维护一个源节点的token 偏移量计数器“ 是因为:
                      recv_x 的token的排列顺序是: 节点优先，节点内的local_rank次优先，channel更次优先。
                      total_offset 此时已经被限定在了节点lane_id的local_rank src_nvl_rank经由channel channel_id发送给当前rank的token的全局偏移量范围之内。
                      因此，这里同一个lane负责记录的对应的源节点发送过来的token数据在recv_x中是连续存储的。
                      这也是可以在前面使用recv_gbl_channel_prefix_matrix记录全局r2r的channel级别前缀和的原因。
                */ 
                if (lane_id == meta.src_rdma_rank) {
                    // 当前lane负责的源节点meta.src_rdma_rank的local_rank src_nvl_rank经由channel channel_id发送给当前rank的token数量加1。
                    // 这也是下一个token要写入到recv_x中的位置。
                    total_offset += 1;
                }

                // auto scale_bytes = num_scales * sizeof(float);  // 表示每个token的scale数量，等于hidden / 128。
                // TMA 通常要求地址和大小按 16 字节对齐，才能高效批量传输。
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
                    /*
                    recv_x: 只存储hidden数据。recv_x是Tensor的内存，shape是[num_recv_tokens, hidden]，类型为 int4*。
                    recv_x_scales: 只存储scale。scale存在独立的Tensor的内存中。
                    两者是分离的数组，不是交错存储。
                    num_bytes_per_token 是缓冲区中的布局的token大小。
                    */
                    tma_store_1d(tma_buffer, recv_x + recv_token_idx * hidden_int4, hidden_bytes, false);
                    if (scale_aligned)
                        tma_store_1d(tma_buffer + hidden_bytes, recv_x_scales + recv_token_idx * num_scales, scale_bytes, false);
                }
                __syncwarp();
                shifted += hidden_bytes;

                // Copy scales
                // TODO: make it as templated
                if (not scale_aligned) {  // 如果scale没有按16字节对齐，则需要单独复制scale数据，此时不经过共享内存，这一点scale数据相对于hidden数据来说非常少。
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
                    /* 
                    idx_value 存储为rank内托管的局部专家ID，不在当前rank上的专家ID就是-1。
                    */
                    idx_value = (idx_value >= local_expert_begin and idx_value < local_expert_end) ? idx_value - local_expert_begin : -1;
                    // 不在当前rank上的专家的权重设置为0，这要和 combine 核函数中的 combine_token 的最后求权重和的代码对应起来理解。
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
                /*
                当前warp作为消费者消费当前rank上的nvl缓冲区的环形队列中的token。
                nvl_channel_head.buffer(): 存储在同节点的local_rank target_rank的nvl缓冲区中，记录当前rank的nvl缓冲区中的环形队列的head索引。 
                                           表示同节点的local_rank target_rank要经由 channel channel_id发送到当前rank的token在当前rank的nvl缓冲区中的环形队列的head索引。
                                           同节点的local_rank target_rank需要知道这个信息，是用于target_rank判断当它的rdma接收缓冲区往当前rank的nvl环形队列缓冲区写token数据时的空闲空间的判断。
                同样地，消费端才有资格更新head。
                注意: 这里有个问题: 为什么当前rank的缓冲区的head索引，需要存储在同节点的local_rank target_rank的nvl缓冲区中？而不是存储在当前rank的nvl缓冲区中？
                     这是因为同节点的local_rank target_rank在往当前rank的nvl缓冲区写数据时，是通过轮询读取当前rank的nvl环形队列的head指针的，
                     如果轮询读取每次都是到当前rank的nvl缓冲区中来读取，那么每次读取都要经过NVLink通信，导致效率太低。
                     而将这个head索引存储在同节点的local_rank target_rank的nvl缓冲区中，那么它轮询读取的时候就读的是它本地的nvl缓冲区，不需要经过NVLink通信。
                     另外，这里也不会出现多个rank对这个环形队列的head指针的竞争，因为当前rank的每环形队列（nvl_channel_x）都是对应同节点的唯一一个local_rank target_rank的。
                注意: 由于整个dispatch函数中，只有kNVLReceivers warp会更新nvl_channel_head.buffer()，而在kRDMAAndNVLForwarder warp中使用轮询读取nvl_channel_head.buffer()，
                     所以这里使用relaxed语义可以避免不必要的同步开销。一般对于head指针都是这样操作。
                */
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }
    }

    // Clean unused `recv_topk_idx` as -1
    /* 
    如果使用模式2：Worst case模式。使用最坏情况的值（预分配），那么 recv_topk_idx 的shape是 [num_worst_tokens, num_topk]。
    num_worst_tokens = num_tokens * num_ranks: 即假设所有 rank 的所有 token 都发送到当前 rank，这是理论上的上界。
    */
    if (num_worst_tokens > 0) {
        if (is_forwarder)
            return;
        // get the actual number of num_recv_tokens on the current rank
        int num_recv_tokens = recv_gbl_rank_prefix_sum[num_ranks - 1];
        // some ForwarderCoordinator threads exit early, so we only use non-forwarder in clean-up
        // channel_id * num_threads is the offset of the current non-forwarder sms
        const auto clean_start = num_recv_tokens * num_topk + channel_id * num_threads;  // 当前sm要清空的的token的开始位置
        const auto clean_end = num_worst_tokens * num_topk;
        const auto clean_stride = num_channels * num_threads;  // grid的线程数
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
    // cudaFuncAttributeMaxDynamicSharedMemorySize 动态共享内存大小：kNumTMABytesPerWarp * NUM_MAX_NVL_PEERS
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


}  // namespace internode

}  // namespace deep_ep
