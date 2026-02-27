#include <cstring>
#include <vector>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#ifndef DISABLE_NVSHMEM
#include "ibgda_device.cuh"
#include "nvshmem.h"
#endif

namespace deep_ep {

namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                  \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 32, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace intranode

namespace internode {

#ifndef DISABLE_NVSHMEM
nvshmem_team_t cpu_rdma_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    /*
    nvshmemx_init_attr 会：
    建立所有 PE 之间的通信基础设施
    创建对称堆（symmetric heap）的基础结构
    为每个 PE 分配相同大小的堆空间
    */
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(cpu_rdma_team == NVSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        /*
        int nvshmem_team_split_strided(
            nvshmem_team_t parent_team,     // 父team（NVSHMEM_TEAM_WORLD）。NVSHMEM_TEAM_WORLD：所有 PE 的全局 team
            int start,                      // 起始索引。rank % NUM_MAX_NVL_PEERS：起始索引（节点内 GPU 索引）
            int stride,                     // 步长。NUM_MAX_NVL_PEERS：步长（8）
            int size,                       // team大小。num_ranks / NUM_MAX_NVL_PEERS：team 大小（节点数）
            const nvshmem_team_config_t* config,  // 默认配置即可
            long config_mask,                   // 默认0即可
            nvshmem_team_t* new_team             // 输出的新team。&cpu_rdma_team
        );
        */
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                                  rank % NUM_MAX_NVL_PEERS,
                                                  NUM_MAX_NVL_PEERS,
                                                  num_ranks / NUM_MAX_NVL_PEERS,
                                                  &cpu_rdma_team_config,
                                                  0,
                                                  &cpu_rdma_team) == 0);
        EP_HOST_ASSERT(cpu_rdma_team != NVSHMEM_TEAM_INVALID);
    }

    nvshmem_barrier_all();
    return nvshmem_my_pe();
}

void* alloc(size_t size, size_t alignment) {
    /*
    NVSHMEM 的 nvshmem_align 函数分配 RDMA 注册内存, 使其支持 RDMA 操作。
    NVSHMEM 在分配时通过协商/全局状态（并通常在分配后做 barrier）保证“对称分配语义”。因此发送端可以用同样的 offset 指向接收端的对应槽位。
    NVSHMEM 的设计语义就是“symmetric heap”——每个 PE 在其本地都有一块同样大小、同样布局的堆区域。
    
    注意: NVSHMEM 利用这一点：发送方用一个“逻辑地址”或 offset 来描述目标位置，接收方用其本地 heap_base + offset 访问数据。
    */
    return nvshmem_align(alignment, size);
}

void free(void* ptr) {
    nvshmem_free(ptr);
}

void barrier() {
    nvshmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
}
#endif

}  // namespace internode

}  // namespace deep_ep
