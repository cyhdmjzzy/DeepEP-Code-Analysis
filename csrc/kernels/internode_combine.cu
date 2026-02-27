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

template <bool kLowLatencyMode, int kNumTMABytesPerWarp>
// const int kNumTMABytesPerWarp = 8192，即8KB。每个warp分配8192字节的TMA缓冲区（必须位于共享内存中）。
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
    /*
    rdma_clean_offset:          1,                                  要读取的。表示RDMA缓冲区需要清零的偏移量（以int为单位）。
    rdma_num_int_clean:         1,                                  要读取的。表示RDMA缓冲区需要清零的int数量。
    nvl_clean_offset:           1,                                  要读取的。表示NVLink缓冲区需要清零的偏移量（以int为单位）。
    nvl_num_int_clean:          1,                                  要读取的。表示NVLink缓冲区需要清零的int数量。
    combined_rdma_head:       [num_combined_tokens, kNumRDMARanks], 要读写的。combined_rdma_head[token_idx * kNumRDMARanks + dst_rdma_rank]表示在dispatch阶段当前rank的token token_idx发送到节点dst_rdma_rank的local_rank nvl_rank的RDMA接收缓冲区环形队列的索引（就是dispatch阶段的 send_rdma_head ）。-1表示该token未发送到该节点，会被转换为特殊值供combine阶段判断使用。
    num_combined_tokens:        1,                                  要读取的。表示当前rank在combine阶段要处理的token数量（即dispatch阶段接收到的token数量）。
        在dispatch阶段: 表示当前rank收到的来自所有节点的local_rank nvl_rank发送到当前rank的token数量之和。 
                       即在notify_dispatch中记录到 moe_recv_rdma_counter_mapped 中的值，
                       亦即send_nvl_head的shape（[num_rdma_recv_tokens, NUM_MAX_NVL_PEERS]）中的num_rdma_recv_tokens。
        在combine阶段:  表示当前rank作为中转rank要从rdma发送缓冲区发送的所有token数量之和。ibgda发送前需要先combine，使得真正需要发送的token数量是num_combined_tokens。
        注意: 无论在dispatch阶段还是combine阶段, 当前rank作为中转rank接收或发送的 “关于所有节点的local_rank nvl_rank” 的这num_combined_tokens个token，
              都是按照“先节点层次，然后channel层次，再然后token层次，最后是local_rank层次”的顺序从小到大排布的。这一点可以参见dispatch函数的kRDMAAndNVLForwarder warp中对send_nvl_head的解释。
    
    num_channels:               1,                                  要读取的。表示combine通信使用的channel数量。
    rdma_channel_prefix_matrix: [kNumRDMARanks, num_channels],      要读取的。由 dispatch 阶段写入的 recv_rdma_channel_prefix_matrix 传入。
        rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id]
        在dispatch阶段: 表示节点dst_rdma_rank的rank nvl_rank 要（经由当前rank作为中转rank）从channel 0到channel warp_id（包含）累计要发送到当前节点的token数量之和（r2n的channel级别前缀和）。
        在combine阶段:  表示当前rank作为中转rank从channel 0到channel warp_id（包含）累计要发送到节点dst_rdma_rank的rank nvl_rank的token数量之和（n2r的channel级别前缀和）。
        注意: dispatch阶段和combine阶段这两个值是一样的。combine阶段发送的是已经在 kNVLAndRDMAForwarder warp内合并好的token数据。
    
    rdma_rank_prefix_sum:       [kNumRDMARanks],                    要读取的。由 dispatch 阶段写入的 recv_rdma_rank_prefix_sum 传入。
        rdma_rank_prefix_sum[dst_rdma_rank - 1]
        在dispatch阶段: 表示从节点0的local_rank nvl_rank到节点(dst_rdma_rank - 1)（包含）的local_rank nvl_rank累计发送到当前rank的token数量之和（r2n的rank级别前缀和）。
        在combine阶段:  表示当前rank作为中转rank要发送的从节点0的local_rank nvl_rank到节点(dst_rdma_rank - 1)（包含）的local_rank nvl_rank累计的token数量之和（n2r的rank级别前缀和）。
    
    combined_nvl_head:    [num_combined_tokens, NUM_MAX_NVL_PEERS], 要读写的。由 dispatch 阶段写入的 send_nvl_head 传入。
        combined_nvl_head[token_idx * NUM_MAX_NVL_PEERS + dst_nvl_rank]
        在dispatch阶段: 表示当前rank作为中转节点从所有节点的local_rank nvl_rank接收到的所有token中的第token_idx个token要发送到当前节点的local_rank dst_nvl_rank的NVL缓冲区环形队列的逻辑索引。
        在combine阶段:  表示当前rank作为中转节点要从当前节点的local_rank dst_nvl_rank **可能**接收的第token_idx个token在当前节点的local_rank dst_nvl_rank的NVL缓冲区环形队列中的逻辑索引。
        这个**可能**是说token token_idx可能在dispatch阶段根本就没有发送到当前节点的local_rank dst_nvl_rank中，那么这个值就是-1。
        
    rdma_buffer_ptr:            void*                               要读写的。RDMA对称缓冲区指针，用于节点间RDMA通信。
    buffer_ptrs:                NUM_MAX_NVL_PEERS, void**           要读写的。节点内所有rank的NVLink buffer指针数组。
    barrier_signal_ptrs:        [kNumRanks, kNumRanks],             要读写的。节点内的所有rank同步所需的barrier信号指针数组。
    rank:                       1,                                  要读取的。表示当前rank在集群内的唯一编号（全局rank编号）。
    num_ranks:                  1,                                  要读取的。表示整个训练集群中的rank总数量。
    is_cached_dispatch:         1,                                  要读取的。表示是否使用cached dispatch模式。如果为true，则跳过combined_rdma_head和combined_nvl_head的处理。
    rdma_team:                  nvshmem_team_t                      要读取的。通过传入特定的nvshmem_team_t，可以控制哪些GPU参与RDMA（远程直接内存访问）通信的同步，确保只有相关的GPU进程协同工作，提高通信效率。
    */
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);
    auto num_threads = static_cast<int>(blockDim.x);
    auto num_warps = num_threads / 32;
    auto warp_id = thread_id / 32;
    auto lane_id = get_lane_id();

    auto nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    auto rdma_rank = rank / NUM_MAX_NVL_PEERS;

    // SM 0: 负责清零RDMA和NVL缓冲区，确保缓冲区干净供combine阶段使用
    if (sm_id == 0) {
        // 等待所有RDMA QP完成未完成的传输操作，确保RDMA缓冲区中的数据已完全写入
        /*
        num_rc_per_pe: int类型，每个rank的RC QP数量。
        num_devices_initialized: 当前 PE 选择并成功初始化的 NIC 设备数量。
        qps_per_rdma_rank: 表示的是当前PE每次往某个节点发送数据的qp数量。
        */
        auto qps_per_rdma_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;
        for (int i = thread_id; i < qps_per_rdma_rank * (num_rdma_ranks - 1); i += num_threads) {
            /*
            使用模运算的循环偏移（Circular Shift）使得dst_rdma_rank的值在[0, num_rdma_ranks-1]之间循环，
            但是恰好避开rdma_rank，从而实现对所有目标节点的QP进行等待。 
            (i / qps_per_rdma_rank + 1) ∈ [1, num_rdma_ranks - 1]，是变量。
            rdma_rank ∈ [0, num_rdma_ranks - 1]，是常量。
            (i / qps_per_rdma_rank + rdma_rank + 1) ∈ [rdma_rank + 1, rdma_rank + num_rdma_ranks - 1]，是变量。
            将这个范围分为两部分: [rdma_rank + 1, num_rdma_ranks] 和 [num_rdma_ranks + 1, rdma_rank + num_rdma_ranks - 1]。
            第一部分对num_rdma_ranks取模，得到的是{rdma_rank + 1, rdma_rank + 2, ..., 0} 中的一个整数。
            第二部分对num_rdma_ranks取模，得到的是{1, 2, ..., rdma_rank - 1} 中的一个整数。
            因此，这个变量对num_rdma_ranks取模，得到的是[0, num_rdma_ranks-1]中除了rdma_rank的任何一个整数。
            */
            auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % num_rdma_ranks;
            // 在每个目标节点的QP中选择具体的QP编号
            auto qp_id = i % qps_per_rdma_rank;
            // 等待指定目标 PE 的指定 QP 上所有已提交的 RDMA 操作完成。用于确保在清理缓冲区或进行同步操作前，所有未完成的 RDMA 写操作已完成。
            nvshmemi_ibgda_quiet(translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), qp_id);
        }
        __syncthreads();

        // Barrier for RDMA: 同步RDMA team内的所有PE，确保所有RDMA操作已完成
        if (thread_id == 32)
            /*
            低延迟模式：使用nvshmem_sync(rdma_team)只同步指定 team 内的GPU，也就是不同节点上的具有相同nvl_rank的GPU作为一个team。
            普通模式：使用nvshmem_sync_all()同步所有节点。也就是所有节点的所有GPU作为一个team。
            */
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);

        /* 
        Barrier for NVL: 同步节点内所有rank，确保节点内通信已完成
        第二个模板参数为true表示执行内存屏障：确保所有之前的内存操作对同一个节点内的所有GPU的所有线程可见。
        ”fence.acq_rel.sys“ 中的 sys 表示 system scope。
        system scope 影响同一节点内的所有GPU的内存系统，跨越GPU边界，不跨越节点边界。
        */
        barrier_block<NUM_MAX_NVL_PEERS, true>(barrier_signal_ptrs, nvl_rank);

        // Clean RDMA buffer: 清零RDMA缓冲区中需要清理的区域
        auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
        #pragma unroll
        /*
        rdma_clean_offset 和 rdma_num_int_clean 的计算参见 get_rdma_clean_meta 函数。
        rdma_num_int_clean 表示: 需要清零的 RDMA 元数据区域的 int 数量。
        rdma_num_int_clean = (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels
        */
        for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
            rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

        // Clean NVL buffer: 清零NVLink缓冲区中需要清理的区域
        auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
        #pragma unroll
        /*
        nvl_clean_offset 和 nvl_num_int_clean 的计算参见 get_nvl_clean_meta 函数。
        nvl_num_int_clean 表示: 需要清零的 nvl 元数据区域的 int 数量。
        nvl_num_int_clean = NUM_MAX_NVL_PEERS * (2 * num_rdma_ranks + 2) * num_channels,
        */
        for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
            nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;
        __syncthreads();

        // Barrier again: 再次同步，确保清零操作对所有rank可见
        if (thread_id == 32) // 保证内存可见性，只需要有一个线程发起即可。因为这不是屏障同步。
            // 仅确保先前发起的**内存存储操作（指本地PE对自己内存的常规存储操作）**完成并可见（本地PE自己可见）
            nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
        // 执行节点内的屏障同步，并且实现节点内跨rank之间的内存可见。但是不跨越节点可见。
        barrier_block<NUM_MAX_NVL_PEERS>(barrier_signal_ptrs, nvl_rank);
    } else if (sm_id == 1) {
        // SM 1: 处理 combined_rdma_head，将-1转换为特殊值供combine阶段使用
        if (is_cached_dispatch)
            return;

        // 保证每个channel有一个warp来处理token，这也是get_channel_task_range中用warp_id表示channel_id的前提。
        EP_DEVICE_ASSERT(num_warps >= num_channels);
        EP_DEVICE_ASSERT(num_rdma_ranks <= 32);

        /* 
        倒序处理token，以便正确传播last_head值。
        注意: 这里的if条件中，在当前sm中，只用了num_channels个warp，每个warp只用了num_rdma_ranks个lane。
              每个warp对应一个channel，每个lane对应一个节点。
        */
        if (lane_id < num_rdma_ranks and warp_id < num_channels) {
            int token_start_idx, token_end_idx;
            /* 
            获取当前channel（warp_id）要处理的token范围。
            注意: 同一个channel中的token才需要下面这样的处理，因为为了尽可能早地、尽可能多地推进rdma_channel_head中的head，也是在同一个channel中进行的。
            */
            get_channel_task_range(num_combined_tokens, num_channels, warp_id, token_start_idx, token_end_idx);

            /* 
            `1 << 25` 是一个启发式的大数值 33554432，用于初始化last_head。
            如果一开始遍历时，token_idx (token_end_idx-1) 就是未发送的，或者接下来连续多个都是未发送的，那么就设置为
            这个很大的值（“取反减一”存负数）。在读取到这个head值时，说明消费者把它前面的逻辑slot的数据都已经消费完了，
            都可以在生产者写数据时作为空闲的slot写入。
            */
            int last_head = 1 << 25;
            // 倒序遍历token，从后往前处理
            for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                // 读取token token_idx发送到节点lane_id的local_rank nvl_rank的RDMA接收缓冲区环形队列的索引
                auto current_head = __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id);
                if (current_head < 0) {
                    /* 
                    如果为-1（表示未发送），将其转换为特殊值 -last_head - 1
                    combine阶段看到负值就知道"不需要等待新写入（不会有数据，因为dispatch阶段就没有发送），直接当作尚无负载处理"。
                    并且在需要读取这个值时可以直接将这个值“取反减一”作为可以推进到的head值，这样一下子就可以推进多个空的slot。
                    */
                    combined_rdma_head[token_idx * num_rdma_ranks + lane_id] = -last_head - 1;
                } else {
                    /* 
                    如果为非负数（表示已发送），使用last_head记录最近一次出现的已发送token的head值，供接下来所有比token_idx小的
                    并且未发送的token用作可以推进到的head值（但是这种情况要“取反减一”，以示这是未发送的token）。
                    */
                    last_head = current_head;
                }
            }
        }
    } else {
        /* 
        SM 2～(num_channels * 2 - 1): 一共 (num_channels * 2 - 2) 个sm，处理 combined_nvl_head ，将-1转换为特殊值供combine阶段使用。
        combined_nvl_head: [num_combined_tokens, NUM_MAX_NVL_PEERS]
        */
        if (is_cached_dispatch)
            return;

        EP_DEVICE_ASSERT(num_warps >= num_channels);
        EP_DEVICE_ASSERT(rdma_channel_prefix_matrix != nullptr and rdma_rank_prefix_sum != nullptr);
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Too many NVL peers");

        /*
        注意: 这里的if条件中，在当前sm中，只用了num_channels个warp，
              每个warp对应一个channel，每个sm在for循环中负责处理一批节点。
        */
        if (warp_id < num_channels) {
            // TMA批量处理配置：计算每个batch可以处理的token数量
            /*
            kNumTMABytesPerWarp = 8192，即8KB。每个warp分配8192字节的TMA缓冲区。
            减去sizeof(uint64_t)是为了给每个warp的每个stage都设置一个uint64_t的mbarrier。
            tma_batch_size: 表示每个batch可以传输的字节数。
            */
            constexpr int tma_batch_size = kNumTMABytesPerWarp - sizeof(uint64_t);
            /*
            每个token需要传输 NUM_MAX_NVL_PEERS 个head索引。每个head索引是用一个int存储的。
            num_bytes_per_token: 表示每个token需要传输的字节数，即 NUM_MAX_NVL_PEERS * sizeof(int) = num_bytes_per_token = 32。
            */
            constexpr int num_bytes_per_token = sizeof(int) * NUM_MAX_NVL_PEERS; 
            // num_tokens_per_batch: 表示每次TMA传输可以传输多少份token的数据。
            constexpr int num_tokens_per_batch = tma_batch_size / num_bytes_per_token;
            // num_bytes_per_token 必须被16整除，因为TMA加载和存储操作需要16字节对齐。
            EP_STATIC_ASSERT(num_bytes_per_token % 16 == 0, "num_bytes_per_token should be divisible by 16");

            /*
            TMA stuffs: 初始化TMA共享内存缓冲区和barrier。
            注意: 每个warp（channel）有一个字节的TMA共享内存缓冲区。
            smem_tma_buffer: 总大小是smem_size = kNumTMABytesPerWarp * num_warps ，同一个sm中的所有 warp 共用一块共享内存，按 warp_id 分段。
                             num_channels 个warp均分smem_tma_buffer这块共享内存，每个warp分配到的字节数是 kNumTMABytesPerWarp 。
                             如果 num_warps > num_channels，那么 （num_warps - num_channels）个kNumTMABytesPerWarp字节的空间就不会被用到
            */
            extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];  // 注意: smem_tma_buffer 是对应一个 sm（blcok）的。
            // 偏移到当前 warp 的 TMA 子 buffer 的起始地址。
            auto tma_buffer = smem_tma_buffer + warp_id * kNumTMABytesPerWarp;
            // 偏移到当前 warp 的 mbarrier 的起始地址。也就是当前warp分配到的kNumTMABytesPerWarp个字节的空间的最后sizeof(uint64_t)个字节的空间。
            auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + tma_batch_size);
            // 初始化TMA phase。
            uint32_t tma_phase = 0;
            if (elect_one_sync()) {
                mbarrier_init(tma_mbarrier, 1);
                fence_barrier_init();
            }
            __syncwarp();

            /* 
            遍历当前SM负责的所有节点（sm_id - 2是要处理的起始节点，步长为num_channels * 2 - 2）。减2是因为前两个sm有各自特殊的任务。
            注意: 当前任务中，一共有 (num_channels * 2 - 2) 个 sm 并行处理，因而for循环中的步长就是 (num_channels * 2 - 2)，
                  也就是每轮循环都会处理 (num_channels * 2 - 2)个节点的数据。
            */
            for (int dst_rdma_rank = sm_id - 2; dst_rdma_rank < num_rdma_ranks; dst_rdma_rank += num_channels * 2 - 2) {
                /* 
                计算r2n的channel级别前缀和。
                rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id - 1]: 表示当前rank作为中转rank从channel 0到
                    channel (warp_id - 1)（包含）累计要发送到节点dst_rdma_rank的rank nvl_rank的token数量之和（n2r的channel级别前缀和）。
                
                token_start_idx 和 token_end_idx 此时都是channel级别的前缀和偏移量。
                */
                int token_start_idx = warp_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id - 1];
                int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id];
                // 加上RDMA rank级别的前缀和偏移
                /*
                rdma_rank_prefix_sum[dst_rdma_rank - 1]: 表示当前rank作为中转rank要发送到的从节点0的local_rank nvl_rank到
                    节点(dst_rdma_rank - 1)（包含）的local_rank nvl_rank累计的token数量之和（n2r的rank级别前缀和）。
                
                shift 是rank级别的前缀和偏移量。
                */
                int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                /*
                token_start_idx 和 token_end_idx 都加上rank级别的前缀和偏移量（shift）之后，就变成了当前rank要经由channel warp_id发送到
                    节点dst_rdma_rank的rank nvl_rank的token在combined_nvl_head的num_combined_tokens个token的token的逻辑索引范围。
                注意: 每一对 (token_start_idx, token_end_idx) 都对应一对 (目的地节点, channel)，也对应一对 (目的地节点, warp)，也对应一对(sm, warp)。
                      外层for循环每次执行时，每个sm负责处理一个目的地节点的head。
                */
                token_start_idx += shift;
                token_end_idx += shift;

                /* 
                NOTES: `1 << 25` 是一个启发式的大数值，用于初始化last_head。
                注意: last_head 的定义是在下面的两重for循环的外面。每次处理一对 (目的地节点, channel) 的head数据才需要一个last_head。
                */
                int last_head = 1 << 25;
                /*
                下面的for循环是当前 (sm, warp) 负责的一对 (目的地节点, channel) 的head数据。

                倒序遍历要往(目的地节点, channel) (目的地节点dst_rdma_rank, channel warp_id) 发送的token，从后往前处理。
                注意: 下面这个for循环遍历的token范围是 [token_start_idx, token_end_idx)，这个范围是当前warp专门负责处理的token范围，
                      不与其他任何sm的任何warp产生token范围的冲突。
                */
                for (int batch_end_idx = token_end_idx; batch_end_idx > token_start_idx; batch_end_idx -= num_tokens_per_batch) {
                    // 从后往前遍历，找 start_idx。保证每次用TMA传输的head数据所对应的token数不超过num_tokens_per_batch。
                    auto batch_start_idx = max(token_start_idx, batch_end_idx - num_tokens_per_batch);

                    // TMA加载：从全局内存批量加载combined_nvl_head数据到共享内存
                    if (elect_one_sync()) {
                        /* 
                        这里传输的不是token数据，而是 (batch_end_idx - batch_start_idx) * NUM_MAX_NVL_PEERS 个head索引。每个head索引是用一个int存储的。
                        因此，这里传输数据的字节数是: (batch_end_idx - batch_start_idx) * NUM_MAX_NVL_PEERS * sizeof(int) = 
                                                  (batch_end_idx - batch_start_idx) * num_bytes_per_token 。
                        
                        tma_buffer: 表示当前 warp 的 TMA 子 buffer 的起始地址。 
                        */
                        tma_load_1d(tma_buffer,
                                    combined_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS,
                                    tma_mbarrier,
                                    (batch_end_idx - batch_start_idx) * num_bytes_per_token);
                        // mbarrier_init(tma_mbarrier, 1); 所以此次TMA传输只需要执行一次mbarrier_arrive_and_expect_tx即可。
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, (batch_end_idx - batch_start_idx) * num_bytes_per_token);
                    }
                    // 等到满足TMA tma_load_1d完成的“那两个条件”。
                    mbarrier_wait(tma_mbarrier, tma_phase);
                    __syncwarp();

                    // 倒序处理batch内的token，将-1转换为特殊值
                    for (int token_idx = batch_end_idx - 1; token_idx >= batch_start_idx; --token_idx) {
                        /*
                        注意: combined_nvl_head的层次是“先节点层次，然后channel层次，再然后token层次，最后是local_rank层次”。
                        kNVLAndRDMAForwarder warp在使用combined_nvl_head的时候读取同一个local_rank的head才有意义，跨不同的local_rank的head没有意义。
                        每个 warp 只有 NUM_MAX_NVL_PEERS 个lane参与。
                        */
                        if (lane_id < NUM_MAX_NVL_PEERS) {
                            /*
                            读取当前token发送到当前NVL rank（lane_id）的head索引。
                            使用TMA就是为了在下面读写head时是对共享内存tma_buffer读写，这样比在全局内存中进行读写要快得多。
                            */
                            auto current_head = reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id];
                            if (current_head < 0) {
                                /*
                                如果为-1（表示未发送），将其转换为特殊值 -last_head - 1。
                                reinterpret_cast<int*>(tma_buffer): 把 tma_buffer 转为 int*，视为 int 数组。
                                [(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id]: 使用下标运算符 []。
                                a[i] 等价于 *(a + i)，[] 已经包含一次**解引用**，因此这里是在对数组的某个元素赋值，不需要在reinterpret_cast前面再写 * 。
                                */
                                reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id] = -last_head - 1;
                            } else {
                                // 如果为非负数（表示已发送），更新last_head为当前head值
                                last_head = current_head;
                            }
                        }
                    }
                    tma_store_fence();
                    __syncwarp();

                    // TMA存储：将在共享内存中读写后的数据写回全局内存的原本位置。
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
    // 32 * num_channels: SM 1 和 SM 2+ 都需要每个block都至少有num_channels个warp，而每个warp32个线程。
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
        if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {  // 只处理当前token发送到第i个rank的情况
            // 获取当前token在第 i 个rank上的缓冲区环形队列中的物理索引。
            slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
            topk_ranks[num_topk_ranks++] = i;
        }
    EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);
    EP_STATIC_ASSERT(not(kUseTMA and kMaybeWithBias), "TMA cannot be used by receiver warps");
    EP_STATIC_ASSERT(kNumStages == 2, "Only support 2 stages now");

    // Reduce data
    if constexpr (kUseTMA) {
        /*
        smem_ptr: 当前warp的共享内存起始地址。

        warp warp_id 的共享内存 (9248 字节)
            ├── Stage 0 ── kNumTMABufferBytesPerStage = 4624 字节
            │   ├── Load Buffer 0 ── kNumTMALoadBytes = 512 字节 (local_rank = 0)
            │   ├── Load Buffer 1 ── 512 字节 (local_rank = 1)
            │   ├── ...
            │   ├── Load Buffer 7 ── 512 字节 (local_rank = 7)
            │   ├── Store Buffer ──── 512 字节 (用于TMA store)
            │   └── Mbarrier ──────── 16 字节 (内存屏障)
            └── Stage 1 ── 4624 字节
                ├── Load Buffer 0 ── 512 字节 (local_rank = 0)
                ├── Load Buffer 1 ── 512 字节 (local_rank = 1)
                ├── ...
                ├── Load Buffer 7 ── 512 字节 (local_rank = 7)
                ├── Store Buffer ──── 512 字节 (用于TMA store)
                └── Mbarrier ──────── 16 字节 (内存屏障)
        */
        constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
        EP_DEVICE_ASSERT(hidden_int4 % 32 == 0);

        auto tma_load_buffer = [=](const int& i, const int& j) -> int4* {
            // 定位到 stage i 的 给local_rank j 提供的TMA共享内存缓冲区 Load Buffer j 的起始地址。
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + j * kNumTMALoadBytes);
        };
        auto tma_store_buffer = [=](const int& i) -> int4* {
            // 定位到 stage i 的 Store Buffer 的起始地址。
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + NUM_MAX_NVL_PEERS * kNumTMALoadBytes);
        };
        auto tma_mbarrier = [=](const int& i) -> uint64_t* {
            // 定位到 stage i 的 Mbarrier 的起始地址。
            return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + (NUM_MAX_NVL_PEERS + 1) * kNumTMALoadBytes);
        };

        // Prefetch
        if (lane_id < num_topk_ranks)
            tma_load_1d(tma_load_buffer(0, lane_id), 
                        get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], 0), 
                        tma_mbarrier(0), kNumTMALoadBytes);
        /*
        注意: 对于每一个warp的每一个stage的mbarrier，在mbarrier_init(tma_mbarrier(lane_id), 32) 中
              初始化 Expected_Arrive_Count = 32，表示期望有32个TMA到达事件。
        */
        mbarrier_arrive_and_expect_tx(tma_mbarrier(0), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
        __syncwarp();

        /*
        一个shifted 对应一个 kNumTMALoadBytes = sizeof(int4) * 32 = 512字节的数据。
        注意: 这里 shifted 是从0开始，步长是32。也就是说 shifted 不对应任何CUDA层次。这在 tma_store_1d 中有体现，
        */
        for (int shifted = 0, iter = 0; shifted < hidden_int4; shifted += 32, iter += 1) {
            const int stage_idx = iter % kNumStages;
            const int next_stage_idx = (iter + 1) % kNumStages;

            // Prefetch next stage
            if (shifted + 32 < hidden_int4) {
                if (lane_id < num_topk_ranks)
                    /* 
                    注意: 这里执行的是stage next_stage_idx的tma_load_1d任务。
                    */
                    tma_load_1d(tma_load_buffer(next_stage_idx, lane_id),
                                get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], shifted + 32),
                                tma_mbarrier(next_stage_idx), kNumTMALoadBytes);
                mbarrier_arrive_and_expect_tx(tma_mbarrier(next_stage_idx), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
                __syncwarp();
            }

            /* 
            阻塞当前线程，直到tma_load_1d传输完成。所谓完成，就是满足“那两个条件”。
            注意: 这里等待的是stage stage_idx，不是stage next_stage_idx。
            */
            mbarrier_wait(tma_mbarrier(stage_idx), tma_phase[stage_idx]);
            // 当前lane对应num_topk_ranks个rank的int4之和。float类型进行累加，最后将结果转为低精度的dtype_t类型放到后面的out_int4中。
            float values[kDtypePerInt4] = {0};
            #pragma unroll
            /*
            num_topk_ranks份共享内存的数据求和后变成 一份数据，然后把这一份数据写到寄存器中。
            */
            for (int j = 0; j < num_topk_ranks; ++j) {
                /*
                这里出现的 lane_id 也许让人疑惑。这是因为: 
                    tma_load_1d 每次传输了 kNumTMALoadBytes = sizeof(int4) * 32 = 512字节的数据。
                这样就实现了一个warp的32个lane刚刚好把某个stage的对应某个local_rank的TMA共享内存缓冲区中的kNumTMALoadBytes字节的数据都读取完。
                */
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(tma_load_buffer(stage_idx, j) + lane_id);
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    // hidden 求和。加权（各专家的输出hidden乘以这个专家在gater时输出的wegiht）的部分在各个rank的专家输出时已经在PyTotch代码中做过了。
                    values[k] += static_cast<float>(recv_value_dtypes[k]);
            }

            // Wait shared memory to be released
            /*
            等待当前线程提交的 tma_store_1d 任务仅剩余 (kNumStages - 1) 个或更少的未完成组。
            也就是说，当前线程提交的 tma_store_1d 任务只要有一个完成了，就可以继续提交下面的 tma_store_1d 任务了。
            */
            tma_store_wait<kNumStages - 1>();

            // Copy into shared and issue TMA
            /*
            注意: Store Buffer的大小也是 kNumTMALoadBytes = sizeof(int4) * 32 = 512字节。也正好是一个warp的32个lane 对应完。
            */
            auto out_dtypes = reinterpret_cast<dtype_t*>(tma_store_buffer(stage_idx) + lane_id);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                /*
                寄存器中的一份数据 写到共享内存中。把寄存器中存储的float数组values转化为低精度的dtype_t类型的数组out_dtypes。
                */
                out_dtypes[j] = static_cast<dtype_t>(values[j]);
            /*
            asm volatile("fence.proxy.async.shared::cta;");  // 官方文档最权威: 等待共享内存写入操作对 TMA 引擎可见。
            更精确地说是: 当前执行线程可见“共享内存写入操作对 TMA 引擎可见”这一信息所在的内存视图。
            */
            tma_store_fence();
            __syncwarp();

            if (elect_one_sync())
                // 每个warp只用一个lane把 8 个local_rank的合并后的int4数据从共享内存写到combined_row的对应位置。
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
            // 每个rank最多只会传一份token过来。并不会因为某个rank上有多个专家被命中就对每份专家输出的token都传过来。
            for (int j = 0; j < num_topk_ranks; ++j)
                recv_value_int4[j] = ld_nc_global(get_addr_fn(topk_ranks[j], slot_indices[j], i));

            // Clean
            // Reduce bias
            // 以float类型这种高精度的数据类型进行累加，最后将结果转为低精度的dtype_t类型放到后面的out_int4中。
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
                // 一个int4类型包含 kDtypePerInt4 个hidden中有实际含义的值。
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    /* 
                    直接把 num_topk_ranks 个 rank中传进来的hidden数据相加。
                    注意: 传入 combine 的 x 的每个 token，是该 token 在该 rank 上命中的多个专家的输出，在 rank 内部已经按这些专家
                          在 topk_weights 中的权重加权求和 的结果。
                          rank 间不再在 combine 里做一次按权重的加权，combine 只做对多个rank的（已加权）输出做加法。
                    */
                    values[k] += static_cast<float>(recv_value_dtypes[k]);
            }

            // Cast back to `dtype_t` and write
            int4 out_int4;
            auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                // 转换为低精度的存储，可以降低内存占用。
                out_dtypes[j] = static_cast<dtype_t>(values[j]);
            st_na_global(combined_row + i, out_int4);  // 存储token合并后的第 i 个int4
        }
    }

    // Reduce `topk_weights`
    /*
    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
        return ld_nc_global(reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token +
                                                     hidden_bytes + sizeof(SourceMeta)) + topk_idx);
    };
    */
    if (lane_id < num_topk) {
        float value = 0;
        #pragma unroll
        /*
        对于同一个逻辑token，也就是dispatch阶段发送过来的token，这 num_topk_ranks 份权重数据不一样。
        具体可见dispatch函数中的 "weight_value = idx_value >= 0 ? weight_value : 0.0f;"。
        也就是传入combine函数的topk_weights数据中，不在当前rank上的专家的权重设置为0。
        */
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
          int kNumForwarders = kNumRDMARanks * kNumWarpsPerForwarder,
          int kNumRDMAReceivers = kNumForwarders - NUM_MAX_NVL_PEERS>
/*
在Deepseek-V3的文档中，kNumRDMARanks=4，而 kNumCombineForwarderWarps=24，所以:
    kNumWarpsPerForwarder=24/4=6，kNumForwarders=4*6=24。
*/
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
    /*
    combined_x:                          要写入的, [num_combined_tokens, hidden], combine 的输出，当前 rank 接收并聚合后的 token 组成的 tensor，供后续层使用。
    combined_topk_weights:               要写入的, [num_combined_tokens, num_topk], 与 combined_x 对应的 top-k 权重。
    is_combined_token_in_rank:           要读取的, [num_combined_tokens, num_ranks], is_combined_token_in_rank[token_idx * num_ranks + dst_rank] 表示 combine 后的第 token_idx 个 token 是否由 rank dst_rank 参与聚合（即该 token 是否发送到过 dst_rank）。
    x:                                   要读取的, [num_tokens, hidden], 当前 rank 的 expert 输出，即 dispatch 阶段当前 rank 作为发送端分发的 num_tokens 个 token 经 expert 计算后的 hidden 数据。
    topk_weights:                        要读取的, [num_tokens, num_topk], 与 x 对应的 top-k 权重。
    bias_0:                              要读取的, [num_combined_tokens, hidden] 或 nullptr, 可选 bias。
    bias_1:                              要读取的, [num_combined_tokens, hidden] 或 nullptr, 可选 bias。
    combined_rdma_head:                  要读取的, [num_combined_tokens, kNumRDMARanks], 由 dispatch 阶段的 send_rdma_head 传入。combined_rdma_head[token_idx * kNumRDMARanks + dst_rdma_rank] 表示当前 rank 的 token token_idx 发送到节点 dst_rdma_rank 的 RDMA 环形队列的 head 索引，-1 表示未发送到该节点，在 cached_notify 中会转换为特殊值供本内核使用。
    combined_nvl_head:                   要读取的, [num_combined_tokens, NUM_MAX_NVL_PEERS], 由 dispatch 阶段的 send_nvl_head 传入。combined_nvl_head[token_idx * NUM_MAX_NVL_PEERS + dst_nvl_rank] 表示当前 rank 的 token token_idx 发送到节点内 rank dst_nvl_rank 的 NVL 环形队列的 head 索引，-1 表示未发送到该 rank，在 cached_notify 中会转换为特殊值供本内核使用。
    src_meta:                            要读取的, [num_combined_tokens, get_source_meta_bytes()], 接收到的 token 的源元数据，包含发送端 rank 与节点内路由信息，用于 combine 时定位与聚合。
    rdma_channel_prefix_matrix:          要读取的, [kNumRDMARanks, num_channels], 由 dispatch 阶段写入的 recv_rdma_channel_prefix_matrix 传入。
        rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id]
        在dispatch阶段: 表示节点dst_rdma_rank的rank nvl_rank 要（经由当前rank作为中转rank）从channel 0到channel channel_id（包含）累计要发送到当前节点的token数量之和（r2n的channel级别前缀和）。
        在combine阶段:  表示当前rank作为中转rank从channel 0到channel channel_id（包含）累计要发送到节点dst_rdma_rank的rank nvl_rank的token数量之和（n2r的channel级别前缀和）。
        注意: dispatch阶段和combine阶段这两个值是一样的。combine阶段发送的是已经在 kNVLAndRDMAForwarder warp内合并好的token数据。
    rdma_rank_prefix_sum:                要读取的, [kNumRDMARanks], 由 dispatch 阶段写入的 recv_rdma_rank_prefix_sum 传入。rdma_rank_prefix_sum[i] 表示从节点 0 到节点 i（含）累计发送到当前 rank 的 token 数量之和（r2n 的 rank 级别前缀和）。
    gbl_channel_prefix_matrix:           要读取的, [num_ranks, num_channels], 由 dispatch 阶段的 recv_gbl_channel_prefix_matrix 传入。gbl_channel_prefix_matrix[(src_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] 表示从全局 rank 0 到该 rank 的、到 channel channel_id（含）累计要发送给当前 rank 的 token 数量，即 token 在 combine 输出中对应段的起始索引。
    num_tokens:                          要读取的, 当前 rank 在 dispatch 阶段作为发送端时的 token 数量（即 x 的 token 维大小）。
    num_combined_tokens:                 要读取的, 当前 rank 在 combine 阶段要聚合的 token 数量（即 dispatch 阶段接收到的 token 数量，combined_x 的 token 维大小）。
    hidden:                              要读取的, 每个 token 的 hidden 维度（以标量元素计）。
    num_topk:                            要读取的, 每个 token 的 top-k 专家数。
    rdma_buffer_ptr:                     要读写的, RDMA 对称缓冲区指针，用于节点间 RDMA 通信。
    num_max_rdma_chunked_send_tokens:    要读取的, 每次 RDMA 发送的最大 token 数（chunk 大小）。
    num_max_rdma_chunked_recv_tokens:    要读取的, 每个 (channel, 节点) 的 RDMA 环形接收缓冲区的 token 容量。
    buffer_ptrs:                         要读写的, NUM_MAX_NVL_PEERS 个 void*，节点内各 rank 的 NVLink buffer 指针。
    num_max_nvl_chunked_send_tokens:     要读取的, 每次 NVL 发送的最大 token 数（chunk 大小）。
    num_max_nvl_chunked_recv_tokens:     要读取的, NVL 接收端环形缓冲区的总 token 容量。
    rank:                                要读取的, 当前 rank 的全局编号。
    num_ranks:                           要读取的, 全局 rank 总数。
    */
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

    /*
    注意: combine作为dispatch的回传，已经不需要传递scales和专家编号的信息了。所以第二个参数和第三个参数都是0。
    */
    const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);

    // NOTES: we decouple a channel into 2 SMs
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        auto warp_id = thread_id / 32;
        if (not is_forwarder_sm) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                auto shuffled_warp_id = warp_id;
                shuffled_warp_id = (shuffled_warp_id + channel_id) % NUM_MAX_NVL_PEERS;
                return {WarpRole::kNVLSender, shuffled_warp_id};   // 8 个 kNVLSender warp。
            } else if (warp_id < kNumForwarders) {   // kNumForwarders=24，因此 kRDMAReceiver warp 有24-8=16个。
                return {WarpRole::kRDMAReceiver, warp_id - NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kCoordinator, 0};  // not is_forwarder_sm 的 kCoordinator warp 有1个。
            }
        } else {
            if (warp_id < kNumForwarders) {  // kNumForwarders=24，因此 kNVLAndRDMAForwarder warp 有24个。
                /*
                “+ channel_id”可以swizzle当前rank的不同的is_forwarder_sm 的sm（每个is_forwarder_sm sm对应一个channel），
                使得各个sm中的kNVLAndRDMAForwarder warp可以均匀地发往不同的远程节点， 不会造成多个sm一下子都往同一个远程节点发数据。
                参见kNVLAndRDMAForwarder warp中的:
                    const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;
                    const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
                */
                auto shuffled_warp_id = (warp_id + channel_id) % kNumForwarders;
                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
            } else {
                return {WarpRole::kCoordinator, 0};  // is_forwarder_sm kCoordinator warp 有1个。
            }
        }
        /* 
        感觉上面强行把是转发和非转发的sm中的kCoordinator取一样的名字很不好，而且kCoordinator warp的代码中也是用is_forwarder_sm区分的，
        这样命名又把不同的代码逻辑搞到一起显得很乱。还不如像dispatch函数中那样，搞 5 种不同的 warp 角色。
        */
    }();
    auto warp_role = role_meta.first;
    auto warp_id = role_meta.second;

    EP_DEVICE_ASSERT(num_warps == kNumForwarders + 1);
    /*
    注意: 相同节点的local_rank的nvl缓冲区接收当前rank的 x 中的token时，会将nvl缓冲区的环形队列容量均分给各个对应的combine最终接收端节点。
    */
    auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;

    if (warp_role == WarpRole::kNVLSender) {    // 8 个 kNVLSender warp。
        /*
        一句话总结: 8个warp。每个warp负责把 tensor x 中的token写入当前节点的对应的中转local_rank的nvl缓冲区中。
        详细实现:
        1、
        */

        // NVL producers
        const auto dst_nvl_rank = warp_id;

        // NVL layouts
        // NOTES: to avoid deadlocks, we use separate NVL buffers for different RDMA sources
        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];

        /* 
        nvl_channel_x: 当前节点的中转local_rank dst_nvl_rank的nvl缓冲区中针对当前channel、当前rank以及所有目的地接收端节点的环形队列。
        这里针对每个节点内rank的大小为num_max_nvl_chunked_recv_tokens的缓冲区会被均分为kNumRDMARanks个部分，每个部分对应一个目的地接收端节点。
        */
        auto nvl_channel_x = AsymBuffer<uint8_t>(dst_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_buffer_ptr);
        /*
        注意: 这个nvl_channel_head与之前的代码很不同的一点就是第二个参数是kNumRDMARanks，而不是 1。
              这是把每个nvl缓冲区的实际物理空间均分为 kNumRDMARanks 个部分，每个部分对应一个目的地接收端节点。
        因为在当前节点的local_rank dst_nvl_rank的nvl缓冲区中针对当前channel、当前rank的环形队列会将实际物理空间均分给各个对应的combine最终接收端节点，
        也就是均匀划分为 kNumRDMARanks 个部分。
        ld_volatile_global(nvl_channel_head.buffer() + lane_id): 
        当前节点的local_rank dst_nvl_rank的nvl缓冲区中针对当前channel、当前rank、目的地接收端节点lane_id的缓冲区指针。

        nvl_channel_head 因为要在当前rank中自旋轮询读取head指针，所以需要存储在当前rank的nvl缓冲区中。
        */
        auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, 
                                    channel_id, num_channels, dst_nvl_rank).advance_also(dst_buffer_ptr);
        /*
        nvl_channel_tail 因为需要在kNVLAndRDMAForwarder warp中自旋等待读取tail指针，而kNVLAndRDMAForwarder warp的当前节点
            是kNVLSender warp中的当前节点的中转local_rank dst_nvl_rank（warp_id），
            因此，nvl_channel_tail需要存储在当前节点的中转local_rank dst_nvl_rank的nvl缓冲区中。
        还是那个原则: 需要在哪个rank自旋等待读取指针，就存储在哪个rank的缓冲区中。
        */
        auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, 
                                    channel_id, num_channels, nvl_rank).advance_also(local_buffer_ptr);

        // TMA stuffs
        /*
        smem_tma_buffer: 声明的一块动态分配的共享内存，按 1024 字节对齐，类型是 uint8_t[]。
                         不同部分给不同的 kNVLSender warp 使用，因为 dst_nvl_rank = warp_id。
        这里之所以只用给每个当前节点的local_rank分配只能缓存一个token数据的共享内存，是因为每个 dst_nvl_rank（warp_id）是对自己负责的local_rank的
            对应 (目的地接收端节点, 目的地接收端节点的各个token) 的两重 for 循环中一个token一个token地处理的。
            因此每个 dst_nvl_rank（warp_id）在同一个时刻每次只能处理一个token。
        TODO: 这里是不是可以优化？比如用 MultiStage TMA 来加速节点内的数据传递。
        */
        extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
        /* 
        constexpr int kNumTMABytesPerSenderWarp = 16384;  // 16KB
        constexpr int kNumTMABytesPerForwarderWarp = 9248;  // 9KB
        */
        auto tma_buffer = smem_tma_buffer + dst_nvl_rank * kNumTMABytesPerSenderWarp;
        auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
        uint32_t tma_phase = 0;
        if (elect_one_sync()) {
            // mbarrier_init(tma_mbarrier, 1)中的arrive_count = 1实际上设置的是Expected_Arrive_Count。
            mbarrier_init(tma_mbarrier, 1);
            fence_barrier_init();
            /*
            kNumTMABytesPerSenderWarp 在实现上取 16384，必须 ≥ 上述 num_bytes_per_token + 8，没有更复杂的“公式”，只是选了一个足够大的常数。
            TODO: 我感觉这里有点浪费共享内存。因为 num_bytes_per_token 表示 hidden + SourceMeta + topk_weights （无 scale、无 topk_idx），
                  再按 16 字节对齐。这是可以明确计算出大小的。
            注意: combine作为dispatch的回传，已经不需要传递scales和专家编号的信息了。
            */
            EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerSenderWarp);
        }
        __syncwarp();

        // Get tasks for each RDMA lane
        int token_start_idx = 0, token_end_idx = 0;
        if (lane_id < kNumRDMARanks) {
            /*
            combine 里的gbl_channel_prefix_matrix就是dispatch中的 recv_gbl_channel_prefix_matrix。
            gbl_channel_prefix_matrix[prefix_idx] 表示**dispatch阶段**来自源节点lane_id的local_rank dst_nvl_rank经由channel channel_id
                发到本节点的local_rank dst_nvl_rank的 token 在当前rank的接收缓冲区（或者 combine 的输入 x/recv_x）中的逻辑区间。这是接收端的前缀和。
            注意: 在同一个nvl缓冲区中，来自同一个源节点的经由相同channel和相同“中转local_rank”发送的token数据是紧邻存储的。
            */
            int prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;
            /*
            这里的token_start_idx 到 token_end_idx(不包含) 表示的是当前rank的 x 中的token区间。
            下面的循环中，token_start_idx增加，但是token_end_idx不变。直到token_start_idx >= token_end_idx，
            才说明当前lane消费完了所负责的token范围，可以退出循环。
            */
            token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
            token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1];
        }
        __syncwarp();

        // NOTES: here the cached value of each lane is only responsible for a single RDMA buffer
        /* 
        cached_channel_head_idx 和 cached_channel_tail_idx分别是下面这个缓冲区的head指针和tail指针:
            当前节点的中转local_rank dst_nvl_rank的nvl缓冲区中针对当前channel、当前rank、目的地接收端节点lane_id的环形队列的head指针和tail指针。
            这个环形队列的物理大小是 num_max_nvl_chunked_recv_tokens_per_rdma。

        每个sm对应一个channel。每个warp负责同节点的一个中转local_rank。每个lane对应一个目的地接收端节点。
        */
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

        // Iterate over all tokens and send by chunks
        /*
        这里shuffle时以channel_id作为索引，是为了当前sm内的所有kNVLSender warp从同一个初始src_rdma_rank开始轮询，
        但是不同的sm中的kNVLSender warp却从不同的节点开始轮询，这样可以避免不同sm中的kNVLSender warp同时从同一个src_rdma_rank相关的地址读取数据，
        造成GPU多个读请求并发从同一个内存读取数据时造成资源竞争。
        */
        int current_rdma_idx = channel_id % kNumRDMARanks;
        while (true) {
            // Exit if possible
            /*
            如果当前rank的 x 中的在区间 [token_start_idx, token_end_idx) 的token都已经被写入到对应local_rank的nvl缓冲区中，则退出循环。
            下面的循环中，token_start_idx增加，但是token_end_idx不变。
            */
            if (__all_sync(0xffffffff, token_start_idx >= token_end_idx))
                break;

            // Decide the next RDMA buffer to send
            bool is_lane_ready = false;
            auto start_time = clock64();
            // 自旋等待，等的就是cached_channel_head_i向前推进到了一个满足条件的位置。
            /* 
            x2nvl等待之一: 等待当前节点的中转local_rank dst_nvl_rank的nvl缓冲区中针对当前channel、当前rank、目的地接收端节点lane_id 的部分有充足的空余空间。
            */
            while (true) {
                int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                // 当前lane要对应的local_rank dst_nvl_rank（warp_id）的nvl缓冲区中针对节点lane_id部分有充足的空余空间，则当前lane准备好了可以开始发送。
                is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and
                                num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
                
                // 8 个 kNVLSender warp，每个warp负责同节点的一个中转local_rank，每个lane负责这个warp对应的中转local_rank对应的目的地节点。
                // 任意一个lane准备好了就可以退出下面的while循环，开始发送数据。
                if (__any_sync(0xffffffff, is_lane_ready))  // SIMT调度，哪怕还剩下一个lane没有处理完，整个warp都得等着。
                    break;

                // Retry
                if (lane_id < kNumRDMARanks and token_start_idx < token_end_idx)
                    /* 
                    cached_channel_head_idx: 当前节点的中转local_rank dst_nvl_rank的nvl缓冲区中针对当前channel、当前rank、目的地接收端节点lane_id的缓冲区指针。
                    因为 kCoordinator warp 中会推进nvl_channel_head的值，这里是在场尝试读取最新值。
                    */
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
                /* 
                如果lane current_rdma_idx没有准备好 当前节点的中转local_rank dst_nvl_rank的nvl缓冲区中针对当前channel、当前rank
                和目的地接收端节点current_rdma_idx 的部分的空余空间，则去执行往下一个目的地接收端节点的缓冲区发送token数据的逻辑。
                */
                if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;

                // Sync token start index
                /* 
                token_idx: 当前rank的 x 中针对 “当前节点的中转local_rank dst_nvl_rank（warp_id）、当前channel、当前rank
                和目的地接收端节点current_rdma_idx” 的部分token的索引。是属于 x 的所有token中的索引。
                */
                auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));

                // num_tokens_in_chunk: 接下来for循环中要发送到当前节点的中转local_rank dst_nvl_rank的nvl缓冲区中的token数。
                int num_tokens_in_chunk =
                    __shfl_sync(0xffffffff, min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx);

                // Send by chunk
                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                    // Get an empty slot
                    int dst_slot_idx = 0;
                    /* 
                    注意: 同一个warp的lane中，只需要lane current_rdma_idx去计算针对目的地接收端节点的环形队列的物理索引，
                          然后用 __shfl_sync 广播到当前warp中的所有lane。
                    */
                    if (lane_id == current_rdma_idx) {
                        // dst_slot_idx: 下面这行代码执行后表示: 针对目的地接收端节点的环形队列的物理索引。
                        dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                        // dst_slot_idx: 下面这行代码执行后表示: 在同节点的中转local_rank dst_nvl_rank的nvl缓冲区中的物理索引。
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
                    }
                    // 依然保持同一个warp中的所有lane接下来都负责处理同一个目的地接收端节点的token数据。
                    // dst_slot_idx 是物理索引，不是逻辑索引，不是物理地址。
                    // 上面的 if 语句中涉及单个lane的判断，但是 if 之后没有调用__syncwarp()，是因为 __shfl_sync 也会起到同步整个warp的所有线程的作用。
                    dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

                    // Load data
                    /*
                    shifted_x_buffers: 当前节点的中转local_rank dst_nvl_rank的nvl缓冲区中针对当前channel、当前rank、
                                       目的地接收端节点current_rdma_idx的部分的环形队列的物理地址。
                    */
                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;
                    /*
                    shifted_x: 当前rank的 x 中针对token token_idx的数据的物理地址。每个int4占用4 * 4 = 16个字节。
                               tensor中存储的是int4类型的数据，所以对 x 中的元素寻址是以int4为单位的，这里不要以字节为单位。
                    当前rank直接从 x 中把token token_idx的数据复制到共享内存tma_buffer中。
                    */ 
                    auto shifted_x = x + token_idx * hidden_int4;
                    /* 
                    tma_store_wait<0>() 用了.read修饰符，只需要等到当前线程的bulk_group中的tma_store_1d必须完成对共享内存的读取即可，
                    不用等待把共享内存中的数据完全都复制到全局内存之后才等待结束。
                    */
                    tma_store_wait<0>();
                    if (elect_one_sync()) {  // 发起tma_load任务只需要一个lane发起，而且这里选择的永远都是lane 0。
                        tma_load_1d(tma_buffer, shifted_x, tma_mbarrier, hidden_bytes);
                        // Arrive_Count“加 1”，Transaction_Count增加num_bytes_per_token。
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);
                    }
                    __syncwarp();  // 上面只选取了一个lane发起tma_load_1d任务，这里同步整个warp的所有线程。
                    // 等待“两个条件”都满足时，这个mbarrier才可以通过，才认为tma_load_1d任务完成
                    mbarrier_wait(tma_mbarrier, tma_phase);

                    // Load source meta
                    if (lane_id == num_topk)
                        /*
                        src_meta的类型是: SourceMeta* 类型。
                        注意: combine作为dispatch的回传，已经不需要传递scales和专家编号的信息了。
                        reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes) 表示: 将 tma_buffer + hidden_bytes 的地址转换为 SourceMeta* 类型。
                        最左边的 * 是解引用操作符，从指针中取值。
                        总体的意思就是: 将指针(tma_buffer + hidden_bytes)指向的地址存储的值设置为 ld_nc_global(src_meta + token_idx)。
                        */
                        *reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes) = ld_nc_global(src_meta + token_idx);

                    // Load `topk_weights`
                    if (lane_id < num_topk)
                        // 注意: 把当前rank的 x 中的token发往同节点的local_rank的nvl缓冲区，是不需要传输 scale 和 topk_idx 的，因为已经不需要了
                        *reinterpret_cast<float*>(tma_buffer + hidden_bytes + sizeof(SourceMeta) + lane_id * sizeof(float)) =
                            ld_nc_global(topk_weights + token_idx * num_topk + lane_id);

                    // Issue TMA store
                    // asm volatile("fence.proxy.async.shared::cta;");  // 官方文档最权威: 等待共享内存写入操作对 TMA 引擎可见。
                    tma_store_fence();
                    __syncwarp();
                    if (elect_one_sync())
                        tma_store_1d(tma_buffer, shifted_x_buffers, num_bytes_per_token, false);
                }
                // lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;

                // 每个lane只用维护 x 中自己需要维护的针对 “目的地接收端节点current_rdma_idx（lane_id）” 的发送了的最新token在 x 中的序号。
                if(lane_id == current_rdma_idx){
                    token_start_idx = static_cast<int>(token_idx);
                }
            }

            // Move queue tail
            tma_store_wait<0>();
            __syncwarp();
            /*
            每个lane各司其职: 上面的两重for循环遍历发送了对所有目的地接收端节点的token数据，而当前warp的当前lane只用负责当前节点的
                            中转local_rank dst_nvl_rank（warp_id）的nvl缓冲区中针对当前channel、当前rank和目的地接收端节点lane_id的部分的
                            环形队列的tail指针的更新。
            还是那个原则: 生产端才有资格更新tail。
            */
            if (lane_id < kNumRDMARanks and is_lane_ready)
                /*
                cached_channel_tail_idx 对应的环形队列的物理大小是 num_max_nvl_chunked_recv_tokens_per_rdma。
                nvl_channel_tail 因为需要在kNVLAndRDMAForwarder warp中自旋等待读取tail指针，而kNVLAndRDMAForwarder warp的当前节点
                    是kNVLSender warp中的当前节点的中转local_rank dst_nvl_rank（warp_id），
                    因此，nvl_channel_tail需要存储在当前节点的中转local_rank dst_nvl_rank的nvl缓冲区中。
                还是那个原则: 需要在哪个rank自旋等待读取指针，就存储在哪个rank的缓冲区中。
                还是那个原则: 读写tail，如果没有acquire-release，可能导致接收端先看到 tail 更新，但读到的是旧数据（实际需要的新数据并没有真正写入）。
                */
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
        }
    } else {
        // Combiners and coordinators
        // RDMA symmetric layout
        auto rdma_channel_data = SymBuffer<int8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

        // NVL layouts
        void* local_nvl_buffer = buffer_ptrs[nvl_rank];
        void* nvl_buffers[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
            nvl_buffers[i] = buffer_ptrs[i];
        auto nvl_channel_x = AsymBuffer<uint8_t>(local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, 
                                NUM_MAX_NVL_PEERS, channel_id, num_channels).advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        /* 
        给每个local_rank分配kNumRDMARanks个head指针。
        nvl_channel_head: 表示当前rank的nvl缓冲区中经由当前channel转发的针对所有local_rank和所有节点的部分的环形队列的head索引。
                          参见: auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;
        注意: nvl_channel_head每个元素都对应一个local_rank, 而每个local_rank对每个channel都会记录 (NUM_MAX_NVL_PEERS * kNumRDMARanks) 个head指针。
        注意: 这里之所以不写为:
            auto nvl_channel_head = AsymBuffer<int>(nvl_buffers[nvl_rank], kNumRDMARanks, 
                                        NUM_MAX_NVL_PEERS, channel_id, num_channels).advance_also(local_nvl_buffer);
            是因为这样会使得 nvl_buffers 中的8个指针没有对齐，也就是相对于buffer_ptrs的8个指针的偏移量不一致，
            这样会导致在后面使用nvl_buffers[i]时出现意想不到的错误，比如接下来的nvl_channel_tail就应该在各个local_rank的nvl缓冲区中的head指针之后。
            advance_also(local_nvl_buffer)也是为了保持相对于buffer_ptrs的初始指针的偏移量一致。
        */
        auto nvl_channel_head = AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks, 
                                    NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_nvl_buffer);
        /* 
        nvl_channel_tail: 表示当前rank的nvl缓冲区中经由当前channel转发的针对所有local_rank的所有节点的部分的环形队列的tail索引。
        nvl_channel_tail 会存储当前rank的nvl缓冲区中经由当前channel转发的 (NUM_MAX_NVL_PEERS * kNumRDMARanks) 个环形队列的tail索引。
        */
        auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels)
                                    .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

        // Combiner warp synchronization
        /*
        
        */
        __shared__ volatile int forwarder_nvl_head[kNumForwarders][NUM_MAX_NVL_PEERS];
        __shared__ volatile bool forwarder_retired[kNumForwarders];
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
        // 加 1 都是因为要等待唯一一个kCoordinator warp。
        auto sync_forwarder_smem = [=]() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumForwarders + 1) * 32)); };
        auto sync_rdma_receiver_smem = [=]() { asm volatile("barrier.sync 1, %0;" ::"r"((kNumRDMAReceivers + 1) * 32)); };

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {  // kNVLAndRDMAForwarder warp 有24个。
            // Receive from NVL ranks and forward to RDMA ranks
            // NOTES: this part is using "large warps" for each RDMA ranks
            /*
            kNumWarpsPerForwarder = kNumCombineForwarderWarps / kNumRDMARanks = 24 / 4 = 6。也就是每 6 个 kNVLAndRDMAForwarder warp 负责 1 个节点。
            */
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
            /*
            针对每个远程节点的rdma缓冲区都有 num_max_rdma_chunked_recv_tokens 个slot。
            */
            auto send_buffer = dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            // 同步kNumWarpsPerForwarder（=6）个kNVLAndRDMAForwarder warp。
            auto sync_large_warp = [=]() {
                if (kNumWarpsPerForwarder == 1) {
                    __syncwarp();
                } else {
                    /*
                    %0 表示 barrier_id，%1 表示 thread_count。
                    + 2: 是因为前面的sync_forwarder_smem 和 sync_rdma_receiver_smem分别占用了一个barrier_id（0和1）。
                    */
                    asm volatile("bar.sync %0, %1;" ::"r"(dst_rdma_rank + 2), "r"(kNumWarpsPerForwarder * 32));
                }
            };
            // 每个SM有且仅有16个硬件barrier（Barrier Units）。
            EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough");

            // TMA stuffs
            constexpr int kNumStages = 2;
            /*
            每次TMA加载32个int4的数据，512字节。
            注意: 这里必须是32，因为一个warp对应32个lane。
            */
            constexpr int kNumTMALoadBytes = sizeof(int4) * 32;  
            /*
            kNumTMABufferBytesPerStage = 16 * 32 * (8 + 1) + 16 = 4608 + 16 = 4624。每个stage需要4624字节的共享内存。
            这里加 1 是为了存储Store Buffer ──── 512 字节 (用于TMA store)的内存空间。
            16: 是Mbarrier ──────── 16 字节 (内存屏障)的内存空间。
            */
            constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
            // kNumTMABufferBytesPerStage * kNumStages = 4624 * 2 = 9248。
            // constexpr int kNumTMABytesPerForwarderWarp = 9248;  // 9KB
            EP_STATIC_ASSERT(kNumTMABufferBytesPerStage * kNumStages <= kNumTMABytesPerForwarderWarp, "TMA buffer is not larger enough");

            // smem_buffer 的大小至少有 kNumTMABytesPerForwarderWarp * kNumCombineForwarderWarps = 9248 * 24 = 221952 个字节，约216KB;
            extern __shared__ __align__(1024) uint8_t smem_buffer[];
            /* 
            smem_buffer (动态共享内存)
            ├── Warp 0 (Forwarder) ── kNumStages * kNumTMABufferBytesPerStage = 2 * 4624 = 9248 字节
            ├── Warp 1 (Forwarder) ── 9248 字节
            ├── ...
            ├── Warp 23 (Forwarder) ─ 9248 字节
            └── 总大小: 24 * 9248 = 221952 字节 (约216KB)

            warp warp_id 的共享内存 (9248 字节)
            ├── Stage 0 ── kNumTMABufferBytesPerStage = 4624 字节
            │   ├── Load Buffer 0 ── kNumTMALoadBytes = 512 字节 (local_rank = 0)
            │   ├── Load Buffer 1 ── 512 字节 (local_rank = 1)
            │   ├── ...
            │   ├── Load Buffer 7 ── 512 字节 (local_rank = 7)
            │   ├── Store Buffer ──── 512 字节 (用于TMA store)
            │   └── Mbarrier ──────── 16 字节 (内存屏障)
            └── Stage 1 ── 4624 字节
                ├── Load Buffer 0 ── 512 字节 (local_rank = 0)
                ├── Load Buffer 1 ── 512 字节 (local_rank = 1)
                ├── ...
                ├── Load Buffer 7 ── 512 字节 (local_rank = 7)
                ├── Store Buffer ──── 512 字节 (用于TMA store)
                └── Mbarrier ──────── 16 字节 (内存屏障)
            
            这里有可能对每个warp空出 (kNumTMABytesPerForwarderWarp - kNumTMABufferBytesPerStage * kNumStages) 字节的共享内存。

            smem_ptr: 表示当前warp的TMA共享内存起始地址。
            */
            auto smem_ptr = smem_buffer + warp_id * kNumStages * kNumTMABufferBytesPerStage;
            // tma_mbarrier(lane_id) 表示当前warp的第 lane_id 个stage的mbarrier的地址。
            auto tma_mbarrier = [=](const int& i) {
                return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1));
            };

            /*
            tma_phase[kNumStages] = {0}; 表示当前warp的每个stage的相位都初始化为0。
            */
            uint32_t tma_phase[kNumStages] = {0};
            if (lane_id < kNumStages) {
                /*
                对于每一个warp的每一个stage的mbarrier，期望32个TMA到达事件，也就是  mbarrier_arrive_and_expect_tx(tma_mbarrier(lane_id), ...) 要执行32次。
                */
                mbarrier_init(tma_mbarrier(lane_id), 32);
                /*
                fence.mbarrier_init.release.cluster: release前不到后。 cluster作用域，确保cluster内所有SM都能看到一致的初始化状态。
                因为使用mbarrier的tma_load_1d是将全局内存的数据写入shared::cluster，而不是shared::cta。
                发布内存屏障，确保之前的mbarrier.init操作对所有线程可见（集群内的其他SM）。 
                */
                fence_barrier_init();
            }
            __syncwarp();

            // Advance to the corresponding NVL buffer
            /*
            这里可能看起来有点奇怪。明明是: 
                当前rank的nvl缓冲区被划分为num_channels部分，每个channel里有针对同节点的local_rank的
                NUM_MAX_NVL_PEERS个nvl缓冲区部分，然后这每个nvl缓冲区部分又被划分为kNumRDMARanks个子部分。
            但是这里为什么直接偏移到节点，并没有偏移到某个local_rank？不是应该先偏移到某个local_rank，再偏移到这个local_rank下的节点dst_rdma_rank的缓冲区吗？
            答: 偏移到local_rank是在后面的 nvl_channel_x.buffer(src_nvl_rank) 中实现的，也就是说先偏移了“内层”，再偏移“才层”。
                这也提醒我在操作内存地址时的灵活性，并不一定要按照内存布局层次来计算到地址。类似的还有dispatch核函数中的send_nvl_head。
            */
            nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token);

            /*
            其实下面这一行代码可以不要！因为kNVLAndRDMAForwarder warp 的后续代码逻辑里面并没有用到 nvl_channel_head。
            
            插句题外话（可以不看）: 这里为什么有这行代码，我猜测要么是为了保持和nvl_channel_tail的偏移量的一致性，要么是因为之前在定义nvl_channel_head的时候
                第一个参数使用的是buffer_ptrs，这就导致kNVLAndRDMAForwarder warp和kCoordinator warp定义的nvl_channel_head都是在同一个
                指针数组buffer_ptrs中进行操作，包括执行offset和读写值。按照这种思路，作者还在接下来调用sync_forwarder_smem()实现在
                kNVLAndRDMAForwarder warp和kCoordinator warp中的同步，也就是kNVLAndRDMAForwarder warp中的
                nvl_channel_head.advance(dst_rdma_rank)在kCoordinator warp中起作用。
                但是后来发现，在kCoordinator warp中想更新nvl_channel_head还得使用forwarder_nvl_head对每个节点对应的head进行更改，于是作者就改变思路，
                让nvl_channel_head中的指针变量（定义时的第一个参数）在每个线程独立的寄存器中都存储一份，这样不同线程之间操作对指针的偏移就不会冲突，
                在kCoordinator warp中以创建nvl_channel_head时的基址进行写入。于是下面这行代码就忘记删除了。
            
            其实下面这一行代码可以不要！因为 kNVLAndRDMAForwarder warp 的后续代码逻辑里面并没有用到 nvl_channel_head。
            */
            nvl_channel_head.advance(dst_rdma_rank);

            /*
            nvl_channel_tail 会存储当前rank的nvl缓冲区中经由当前channel转发的 (NUM_MAX_NVL_PEERS * kNumRDMARanks) 个环形队列的tail索引。
            下面的advance使得nvl_channel_tail偏移到针对节点dst_rdma_rank的nvl缓冲区中的tail指针的位置。
            至于偏移到对应哪个同节点的local_rank的tail指针，则是在 nvl_channel_tail.buffer(lane_id) 中实现的。
            */
            nvl_channel_tail.advance(dst_rdma_rank);

            // Clean shared memory and sync
            EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
            lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
            // 和formarder类型的Coordinator warp的最开始同步，因为现在已经清零了forwarder_nvl_head和forwarder_retired。
            sync_forwarder_smem();

            // Get count and cached head
            int cached_nvl_channel_tail_idx = 0;
            /*
            rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id]: 表示当前rank作为中转rank从channel 0到
                channel channel_id（包含）累计要发送到节点dst_rdma_rank的rank nvl_rank的token数量之和（n2r的channel级别前缀和）。
            */
            int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
            /*
            num_tokens_prefix: 表示当前rank作为中转rank从channel 0到channel (channel_id - 1)（包含）累计要发送到节点dst_rdma_rank的
                local_rank nvl_rank的token数量之和（n2r的channel级别前缀和）。
            */
            int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
            /* 
            执行完下面的代码，num_tokens_to_combine 就表示当前rank作为中转rank经由channel channel_id要发送到节点dst_rdma_rank的local_rank nvl_rank的token数量。
            注意: dispatch阶段在两个不同节点的相同local_rank编号的rank之间发送了多少个token，那么在combine阶段回传时也要发送相同数量的token。回传前先combine。
            */
            num_tokens_to_combine -= num_tokens_prefix;
            /*
            rdma_rank_prefix_sum[i]: 表示dispatch阶段从节点0的local_rank nvl_rank到节点i的local_rank nvl_rank（包含）累计发送到当前rank的token数量之和（r2n的rank级别前缀和），
            num_tokens_prefix 表示 "r2n的channel级别前缀和" 加上 "r2n级别前缀和"。
            */
            num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
            /*
            combined_nvl_head的层次是“先节点层次，然后channel层次，再然后token层次，最后是local_rank层次”。
            下面的代码是让combined_nvl_head偏移到对应 (目标节点, channel) (目标节点dst_rdma_rank, channel channel_id) 的head指针的位置。
            后面会在 ld_nc_global(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id) 中偏移到token层次的token token_idx的第lane_id个local_rank的head指针的位置。
            */
            combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS;

            // Iterate over all tokens and combine by chunks
            /*
            这个for循环针对的是要发往节点dst_rdma_rank的local_rank nvl_rank的rdma接收缓冲区的所有token。
            由 kNumWarpsPerForwarder 个warp负责并行处理这些token。
            注意: num_max_rdma_chunked_send_tokens 是限制并行ibgda发送时的的最大发送token数量。
                 但是实际发送时，并行ibgda发送的token数量是 num_chunked_tokens 。
            */
            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                // Check destination queue emptiness, or wait a buffer to be released
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                // num_chunked_tokens: 表示当前rank的rdma缓冲区要经由channel channel_id发送到节点dst_rdma_rank的local_rank nvl_rank的rdma接收缓冲区的token数量。
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                auto start_time = clock64();
                /*
                每 kNumWarpsPerForwarder = 6 个 kNVLAndRDMAForwarder warp 负责 1 个节点。
                这里只需要这6个warp的其中一个线程来等待目的地rank的rdma接收缓冲区有足够的空余空间。

                nvl2rdma等待之一: 等待目的地rank的rdma接收缓冲区有足够的空余空间。
                */
                while (sub_warp_id == 0 and lane_id == 0) {
                    // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
                    // Here, `token_start_idx` is the actual tail
                    /*
                    注意: token_start_idx看作是tail索引，之所以命名中带有start，是因为写入数据从token_start_idx开始写，也就是从tail开始写。
                    rdma_channel_head.buffer(dst_rdma_rank): 表示当前rank的rdma缓冲区中经由channel channel_id发送的针对节点dst_rdma_rank的环形队列的head的指针。
                    rdma_channel_head 的更新在kCoordinator warp中实现。
                    */
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    /*
                    如若满足条件，则说明目的地节点dst_rdma_rank的local_rank nvl_rank经由当前channel接收的当前rank的rdma接收缓冲区的环形队列中
                    至少有 num_chunked_tokens 个token空余空间，可以开始往里面发送了。
                    */
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
                sync_large_warp();  // 同步6个kNVLAndRDMAForwarder warp。

                // Combine and write to the RDMA buffer
                /*
                每 kNumWarpsPerForwarder = 6 个 kNVLAndRDMAForwarder warp 负责 1 个节点（节点dst_rdma_rank）。
                每个 sub_warp 负责处理 1 个token。这里是“kNumWarpsPerForwarder个warp组成的warp_group循环步进”。
                */
                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
                    // Read expected head
                    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS) {
                        /*
                        前面combined_nvl_head已经偏移到对应 (目标节点, channel) (目标节点dst_rdma_rank, channel channel_id) 的head指针的位置。
                        现在偏移到token层次的token token_idx的第lane_id个local_rank的head指针的位置，读取这个指针指向的head索引。
                        */
                        expected_head = ld_nc_global(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                        /* 
                        下面的每6个连续的 warp_id 对应一个节点，后面在kCoordinator warp中对forwarder_nvl_head读取时，会针对每一个节点的nvl_channel_head更新为
                            对应这个节点的6个sub warp（kNVLAndRDMAForwarder warp）中的最小 head索引。
                        如果expected_head >= 0: 表示该 token 在针对local_rank lane_id的接收缓冲区中有，必须等到 tail 超过 expected_head 才能读取；
                        如果expected_head < 0:  表示该 token 在针对local_rank lane_id的接收缓冲区中没有，cached_nvl_channel_tail_idx <= expected_head 
                                                恒为false（tail 非负），下面的while循环不进入，无需等待。
                        */
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head);
                    }

                    // Wait lanes to be ready
                    start_time = clock64();
                    /*
                    自旋等待，直到当前rank的nvl缓冲区缓存的来自同节点的local_rank lane_id生产的要发送到节点dst_rdma_rank的token数据在环形队列
                    中的tail索引到达或者已经超过了token token_idx对应的索引。即使超过很多，这里也只处理token token_idx。
                    注意: expected_head 和 cached_nvl_channel_tail_idx 都是针对nvl缓冲区环形队列（包括dispatch阶段目的地rank的 和 combine阶段中转rank的）中的索引，
                          而token_idx是rdma缓冲区中的索引（包括dispatch阶段发送端rank和中转rank的 和 combine阶段中转rank和最终目的地rank）。
                    */
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
                    /*
                    token_idx: 是rdma缓冲区中的索引（包括dispatch阶段发送端rank和中转rank的 和 combine阶段中转rank和最终目的地rank）。
                    num_max_rdma_chunked_recv_tokens: 表示rdma缓冲区中针对一个节点的环形队列的物理slot大小。
                    rdma_slot_idx: 从逻辑slot计算得到的物理slot。
                    */
                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_token;
                    auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                        /* 
                        注意: 这里的 nvl_channel_x 已经在前面先偏移到了对应节点dst_rdma_rank的nvl缓冲区中索引位置（nvl_channel_x.advance(...)），
                              所以下面这行代码只需要再偏移到同节点的local_rank src_nvl_rank即可。
                        下面 combine_token 中的 num_max_nvl_chunked_recv_tokens_per_rdma 也印证了这一点。
                        */
                        return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) + hidden_int4_idx;
                    };
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
                        /*
                        获取当前rank中的针对同节点的local_rank src_nvl_rank和目的地节点dst_rdma_rank的nvl缓冲区中的子环形队列的token token_idx的第topk_idx个专家的权重。
                        */
                        return ld_nc_global(reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token +
                                                                     hidden_bytes + sizeof(SourceMeta)) + topk_idx);
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
                    /*
                    上面的combine_token，已经把当前rank的nvl缓冲区中的来自local_rank lane_id的warp_id对应的节点dst_rdma_rank的
                    token token_idx从子环形队列中消费了，现在可以把这个head索引往前推进一位了。
                    */
                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
                }
                /*
                同步对应同一个节点dst_rdma_rank的6个warp，因为后面要发送token数据。
                */
                sync_large_warp();

                // Issue RDMA send
                if (sub_warp_id == kNumWarpsPerForwarder - 1) {  // 只用针对节点dst_rdma_rank的6个warp中的最后一个sub warp来发送token数据。
                    if (dst_rdma_rank != rdma_rank) {   // 如果相等，早就写到当前rank（也是目的地rank）的对应位置的rdma接收缓冲区了。
                        // num_max_rdma_chunked_recv_tokens: 表示rdma缓冲区中针对一个节点的环形队列的物理slot大小。
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        // num_chunked_tokens = token_end_idx - token_start_idx;
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
                        /*
                        保证节点内的线程对内存的可见性。"fence.acq_rel.sys;"是一种轻量级内存屏障：确保所有之前的内存操作对同一个节点内的所有GPU的所有线程可见。
                        业务层面，这是为了保证当前节点内的所有local_rank都可见在combine_token中往 
                        send_buffer = dma_channel_data.recv_buffer(dst_rdma_rank) 写入的数据，这个写入不需要用ibgda。
                        */
                        memory_fence();
                    }

                    // Write new RDMA tail
                    __syncwarp();
                    /*
                    针对节点dst_rdma_rank的6个warp中，只用一个线程去推进translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank))的
                    rdma接收缓冲区中接收当前rank的数据的环形队列的tail索引。
                    */
                    if (elect_one_sync()) {
                        /*
                        注意: 在对称内存构成的send和recv双缓冲区中，发送端发送了发送缓冲区中的数据之后，不需要更新发送缓冲区的tail索引，只用更新接收缓冲区的tail索引。

                        这里写 tail 索引没有用release，那是如何保证接收端读取到新的tail时，数据已经写入到接收缓冲区的呢？
                        答: nvshmemi_ibgda_put_nbi_warp 和下面的 nvshmemi_ibgda_amo_nonfetch_add 都调用了 ibgda_submit_requests 函数。
                            在 ibgda_submit_requests 里面调用了 __threadfence(); __threadfence()有两个作用:
                            1、强制刷新当前线程的写缓冲区，确保所有待写入的数据都真正写入到全局内存；
                            2、建立了happens-before关系，屏障之前的所有内存写入，在屏障之后对其他线程和设备可见。
                        */
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
            /*
            num_tokens_to_combine: 当前rank作为中转rank经由channel channel_id要发送到节点dst_rdma_rank的local_rank nvl_rank的token数量。
            当前warp在 num_tokens_to_combine 个token中负责处理的token都处理完了，可以退休了。
            */
            if (elect_one_sync())
                forwarder_retired[warp_id] = true;
        } else if (warp_role == WarpRole::kRDMAReceiver) {  // kRDMAReceiver warp 有16个。这16个warp并行处理同一个channel中的所有token，而不同的lane对应不同的节点。
            // Receive from RDMA ranks and write to the output tensor
            // Clean shared memory and sync
            EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
            /*
            注意，这里的warp_id已经是从0开始的了，因为在分配 kRDMAReceiver 的warp时，已经把second设置为: warp_id - NUM_MAX_NVL_PEER。
            */
            lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            // 每个warp中只需要一个线程来设置该warp对应的的rdma_receiver_retired[warp_id]为false。
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_smem();  // 用于kCoordinator warp同步等到了rdma_receiver_rdma_head和rdma_receiver_retired已经清空。

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            int cached_channel_tail_idx = 0;
            /*
            每次每个warp处理一个token。kNumRDMAReceivers 个warp并行处理token。
            */
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
                // Read expected head
                EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
                /*
                __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
                在共享内存rdma_receiver_rdma_head中记录可以直接用的都是正数的head索引。而rdma_receiver_rdma_head用于kCoordinator warp中在min reduce中找到最小的head。
                rdma_receiver_rdma_head[warp_id][lane_id]: 表示第warp_id个kRDMAReceiver warp接收当前rank中的来自节点lane_id的local_rank nvl_rank的rdma接收缓冲区的已经消费到的head索引。
                */
                int expected_head = -1;
                if (lane_id < kNumRDMARanks) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);
                    /* 
                    如果expected_head >= 0: 表示该 token 在针对节点lane_id的接收缓冲区中有，必须等到 tail 超过 expected_head 才能读取。哪怕只有一个token也可以读取；
                    如果expected_head < 0:  表示该 token 在针对节点lane_id的接收缓冲区中没有，cached_channel_tail_idx <= expected_head 恒为false（tail 非负），循环不进入，无需等待。
                    执行下面的代码之后，rdma_receiver_rdma_head中存储的就是正数的slot值。
                    */
                    (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1)
                                        : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
                }

                // Wait lanes to be ready
                auto start_time = clock64();
                /*
                自旋等待rdma_channel_tail.buffer(lane_id)超过expected_head。
                还是那个原则: "哪个rank要轮询读某个指针，就把这个指针放在这个rank的内存上"。不要发生跨GPU甚至跨节点轮询读指针的情况。
                还是那个原则: 涉及tail的，都用acquire-release语义，以保证代码有序、内存操作有序和内存可见性。
                */
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
                    /*
                    获取当前rank经由当前channel channel_id接收的来自节点src_rdma_rank的local_rank nvl_rank的rdma接收缓冲区中
                    第slot_idx个token的第hidden_int4_idx个hidden的int4数据地址。
                    */
                    return reinterpret_cast<int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token) +
                        hidden_int4_idx;
                };

                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
                    /*
                    num_bytes_per_token 表示 hidden + SourceMeta + topk_weights （无 scale、无 topk_idx） 占用的字节数。
                    
                    获取当前rank经由当前channel channel_id接收的来自节点src_rdma_rank的local_rank nvl_rank的rdma接收缓冲区中
                    第slot_idx个token的第topk_idx个权重。
                    */
                    return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) +
                                            slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) + topk_idx);
                };
                uint32_t dummy_tma_phases[2];
                combine_token<kNumRDMARanks, true, dtype_t, kNumTopkRDMARanks, false, 2>(
                    expected_head >= 0,
                    expected_head,
                    lane_id,
                    hidden_int4,
                    num_topk,
                    combined_x + token_idx * hidden_int4,  // 逻辑token token_idx合并后存储到combined_x的地址。
                    combined_topk_weights + token_idx * num_topk,  // 逻辑token token_idx合并后存储到combined_topk_weights的地址。
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
            /*
            [token_start_idx, token_end_idx): 表示当前warp要处理的所有token的索引范围。
            当前warp处理完了[token_start_idx, token_end_idx)范围内的所有token，可以退休了。
            */
            if (elect_one_sync())
                rdma_receiver_retired[warp_id] = true;
        } else {  // kCoordinator warp
            // Coordinator
            // Sync shared memory status
            is_forwarder_sm ? sync_forwarder_smem() : sync_rdma_receiver_smem();
            const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

            int last_rdma_head = 0;
            int last_nvl_head[kNumRDMARanks] = {0};  // 针对每个节点记录当前rank的nvl缓冲区中被消费到的最小head索引。
            int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
            // 这里命名为 “dst_XXX”，指的是dispatch阶段token发送到的最终目的地rank，实际上在combine核函数中，这里命名为 “src_XXX” 比较合适，指的是combine阶段最开始发送token的rank。
            int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;
            EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps");
            while (true) {
                // Retired
                /*
                如果都 retired 了，则退出循环。
                注意: 虽然__all_sync针对的是当前warp中的所有lane，但是每个lane对应的是每个kRDMAReceiver warp。因此，只要某一个warp执行
                      这个__all_sync返回true，则所有的kRDMAReceiver warp都会返回true，于是所有当前 sm 中的kRDMAReceiver warp都会退出循环。
                下面的is_forwarder_sm的kCoordinator warp（针对kNVLAndRDMAForwarder warp）也是一样的原理。
                */
                if (not is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
                    break;
                if (is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumForwarders or forwarder_retired[lane_id]))
                    break;

                // Find minimum head for RDMA ranks
                if (not is_forwarder_sm) {
                    /*
                    min_head: 表示所有未 retired 的 kRDMAReceiver warp 中的最小消费进度，即所有 kRDMAReceiver warp 至少都消费到了 slot min_head。
                    */
                    int min_head = std::numeric_limits<int>::max();
                    #pragma unroll
                    // 不用担心这里一直循环读rdma_receiver_rdma_head消耗资源， 因为有__nanosleep(NUM_WAIT_NANOSECONDS);
                    for (int i = 0; i < kNumRDMAReceivers; ++i)
                        if (not rdma_receiver_retired[i])
                            /* 
                            rdma_receiver_rdma_head[i][dst_rdma_rank]: 表示第i个kRDMAReceiver warp消费当前rank的来自节点
                                dst_rdma_rank的local_rank nvl_rank的rdma接收缓冲区的head索引。
                            */
                            min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
                    /*
                    批量更新: 累积到阈值（num_max_rdma_chunked_send_tokens）才更新一次。
                    最小值保证: 使用所有 kRDMAReceiver warp 中的最小 head，确保远程节点不会过早错误地释放尚未被消费的缓冲区。
                    注意: 这样的目的是将多次小更新合并为一次批量更新可以减少通信。
                    */
                    if (min_head != std::numeric_limits<int>::max() and 
                        min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and
                        lane_id < kNumRDMARanks) {
                        /* 
                        将节点dst_rdma_rank的local_rank nvl_rank的rdma发送缓冲区中被当前rank消费到的head索引，推进(min_head - last_rdma_head)。
                        */
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
                    // 针对每一个节点的nvl_channel_head更新为对应这个节点的6个sub warp（kNVLAndRDMAForwarder warp）中的最小 head索引。
                    for (int i = 0; i < kNumRDMARanks; ++i) {
                        int min_head = std::numeric_limits<int>::max();
                        #pragma unroll
                        for (int j = 0; j < num_warps_per_rdma_rank; ++j)  // 这里的 j 相当于kNVLAndRDMAForwarder warp中的sub_warp_id。
                            if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                                /*
                                __shared__ volatile int forwarder_nvl_head[kNumForwarders][NUM_MAX_NVL_PEERS];
                                num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

                                */
                                min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);
                        if (min_head != std::numeric_limits<int>::max() and 
                            min_head > last_nvl_head[i] and 
                            lane_id < NUM_MAX_NVL_PEERS)
                            /*
                            nvl_channel_head.buffer_by(dst_nvl_rank) + i: 表示当前rank的nvl缓冲区中接收local_rank dst_nvl_rank发来的
                                针对节点 i 的token在当前rank的nvl缓冲区环形队列中的head指针。 这个指针本身存储在local_rank dst_nvl_rank的nvl缓冲区中。
                            
                            下面的store时为了: 使得kNVLSender warp中的local_rank dst_nvl_rank在往当前rank的nvl缓冲区中发送token数据时，
                                             在自旋等待nvl_channel_head时等到了更新，于是可以继续往当前rank的nvl缓冲区中发送token数据。
                            注意: 在kNVLAndRDMAForwarder warp中的nvl_channel_head.advance(dst_rdma_rank) 对当前kCoordinator warp中的nvl_channel_head完全无意义。
                            注意: nvl_channel_head.buffer_by(dst_nvl_rank) 调用的是存储在local_rank dst_nvl_rank的nvl缓冲区中的nvl_channel_head部分。
                                  这也满足“需要在哪个rank自旋等待读取指针，就存储在哪个rank的缓冲区中”的原则。
                            */
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
