#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"

namespace deep_ep {

namespace layout {

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void get_dispatch_layout(const topk_idx_t* topk_idx,
                                    int* num_tokens_per_rank,
                                    int* num_tokens_per_rdma_rank,
                                    int* num_tokens_per_expert,
                                    bool* is_token_in_rank,
                                    int num_tokens,
                                    int num_topk,
                                    int num_ranks,
                                    int num_experts) {
    /*
    topk_idx:                   [num_tokens, num_topk],  topk_idx_t  要读取的  [i, j]表示第i个token的第j个expert的索引
    num_tokens_per_rank:        [num_ranks],             int         要写入的  表示当前rank要发往每个rank的 token 数量
    num_tokens_per_rdma_rank:   [num_rdma_ranks],        int         要写入的  表示当前rank要发往每个 RDMA rank 的 token 数量
    num_tokens_per_expert:      [num_experts],           int         要写入的  表示每个 expert 的 token 数量
    is_token_in_rank:           [num_tokens, num_ranks], bool        要写入的  [i, j]表示第i个token是否发往第j个rank
    num_tokens:                 1,                       int         要读取的  表示当前rank要发送的token数量
    num_topk:                   1,                       int         要读取的  表示每个token的激活专家数量
    num_ranks:                  1,                       int         要读取的  表示整个训练集群中的rank数量
    num_experts:                1,                       int         要读取的  表示当前 MoE 层的专家的数量
    */
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);

    // Count expert statistics
    // num_tokens_per_expert_per_thread[thread_id][i]表示第thread_id个线程负责统计的topk_idx中第（expert_begin_idx + i）个expert的token数量。
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
    // 前sm_id个SM每个SM负责统计kNumExpertsPerSM个专家的token数，所以一共是sm_id * kNumExpertsPerSM个专家的token数。
    int expert_begin_idx = sm_id * kNumExpertsPerSM, expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
    // 代码最后有return，也就是说当前核函数分配到的多个SM中，前((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM)个SM负责统计当前rank要发往各个rank的token数量。
    if (expert_begin_idx < expert_end_idx) {
        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumExpertsPerSM; ++i)   // 初始化每个线程负责统计的专家的token数量为0。
            num_tokens_per_expert_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {  // block循环步进
            auto shifted_topk_idx = topk_idx + i * num_topk;
            #pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++j) {  // 遍历token i在topk_idx中的一行数据，即token i的num_topk个激活专家的索引。
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    // (expert_idx - expert_begin_idx) 是在当前block中的专家的局部专家索引。
                    ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
            }
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
        /*
        // 这里是我写的条件判断，还是没有LyricZhao写的robust。因为可能最后一个负责统计专家的SM统计的专家数没对齐kNumExpertsPerSM。
        if(thread_idx < kNumExpertsPerSM){
            int sum = 0;
            #pragma unroll
            for(int i=0; i<kNumThreads; i++){
                sum += num_tokens_per_expert_per_thread[i][thread_idx];
            }
            num_tokens_per_expert[expert_begin_idx + thread_idx] = sum;
        }
        */
        if (expert_begin_idx + thread_id < expert_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
        }
        // 你居然直接搞个return，之前你都是用if else来控制流程的，现在直接用return来控制流程，我有点不适应。
        return;
    }

    if (num_tokens_per_rdma_rank != nullptr)
        EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);

    // Count rank statistics
    constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
    // num_tokens_per_rank_per_thread[thread_id][i]表示第thread_id个线程负责统计的第（rank_begin_idx + i）个rank的token数量。
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
    auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
    // 获取当前SM负责统计的rank的范围: [rank_begin_idx, rank_end_idx)
    int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM, rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
    int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS, rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;
    if (rank_begin_idx < rank_end_idx) {
        const auto num_expert_per_rank = num_experts / num_ranks;
        auto expert_begin = rank_begin_idx * num_expert_per_rank;
        auto expert_end = rank_end_idx * num_expert_per_rank;

        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {  // block循环步进
            auto shifted_topk_idx = topk_idx + i * num_topk;  // 获取token i在topk_idx中的一行数据。
            int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
            #pragma unroll
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {  // 遍历token i在topk_idx中的一行数据，即token i的num_topk个激活专家的索引。
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin <= expert_idx and expert_idx < expert_end) {
                    // Count single rank
                    // expert_idx / num_expert_per_rank 表示专家expert_idx在整个专家列表中的全局rank 索引。
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;  // 得到专家expert_idx所在的rank在当前SM处理的rank中的局部rank 索引。
                    // 因为要统计当前rank要发往各个rank的token数量，这里先记录往专家expert_idx所在的rank要发送的token数量。
                    is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
                }
            }

            // 获取token i在is_token_in_rank中的行数据，这行数据一共num_ranks个元素。
            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
            #pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
                // is_in_rank 由 topk_idx 得到，is_token_in_rank 由 is_in_rank 得到。
                // 记录token i是否发往第 (rank_begin_idx + j) 个rank。
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                // 记录当前SM中，每个线程统计的当前rank要发送给rank (rank_begin_idx + j)的token数量。
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
            }

            #pragma unroll
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
        // 这里，每个thread_id对应一个rank，具体来说就是以rank_begin_idx为rank id起点的偏移量。
        if (rank_begin_idx + thread_id < rank_end_idx) {
            int sum = 0;
            #pragma unroll
            // 聚合当前SM中，每个线程统计的当前rank要发送给rank (rank_begin_idx + thread_id)的token数量。
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];
            // 记录当前rank要发送给rank (rank_begin_idx + thread_id)的token数量。
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
        }

        if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
        }
    }
}

void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         cudaStream_t stream) {
    // kNumExpertsPerSM: 每个 SM 负责统计 4 个 expert 的 token 数量。 作用: 将 expert 按组分配给不同的 SM，每个 SM 统计分配给它的 expert 的 token 数量
    // kNumRanksPerSM:   每个 SM 负责统计 8 个 rank 的 token 数量。   作用: 将 rank 按组分配给不同的 SM，每个 SM 统计分配给它的 rank 的 token 数量
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 4, kNumRanksPerSM = 8;
    // 负责统计expert的token数量的SM 和 负责统计rank的token数量的SM 之和就是num_sms
    int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
    EP_STATIC_ASSERT(kNumRanksPerSM % NUM_MAX_NVL_PEERS == 0, "Invalid number of ranks per SM");

    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg,
                  (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
                  topk_idx,
                  num_tokens_per_rank,
                  num_tokens_per_rdma_rank,
                  num_tokens_per_expert,
                  is_token_in_rank,
                  num_tokens,
                  num_topk,
                  num_ranks,
                  num_experts);
}

}  // namespace layout

}  // namespace deep_ep
