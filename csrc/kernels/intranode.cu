#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace deep_ep {

namespace intranode {

// 启动 1 + num_ranks 个 block（gridDim.x = 1 + num_ranks）
// 每个 block 有 128 个线程（blockDim.x = 128）
template <int kNumRanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank,
                                int* moe_recv_counter_mapped,
                                const int* num_tokens_per_expert,
                                int* moe_recv_expert_counter_mapped,
                                int num_experts,
                                int num_tokens,
                                int num_channels,
                                const bool* is_token_in_rank,
                                int* channel_prefix_matrix,
                                int* rank_prefix_matrix_copy,
                                int num_memset_int,
                                int expert_alignment,
                                void** buffer_ptrs,
                                int** barrier_signal_ptrs,
                                int rank) {
    /*
    num_tokens_per_rank:            kNumRanks,                          要读取的。表示当前的rank rank发送到节点内各个rank的token数量。
    moe_recv_counter_mapped:        1,                                  要写入的。表示所有rank发送到rank rank的token数量之和。
    num_tokens_per_expert:          [kNumRanks, num_experts_per_rank],  要读取的。num_tokens_per_expert[i * num_experts_per_rank + j] 表示rank rank要发送到rank i内的expert j的token数量。
    moe_recv_expert_counter_mapped: num_experts_per_rank,               要写入的。moe_recv_expert_counter_mapped[i] 表示rank rank内的expert i收到的token数量。
    num_experts:                    1,                                  要读取的。表示节点内所有rank的专家数量之和。
    num_tokens:                     1,                                  要读取的。表示当前rank要发送的token数量。
    num_channels:                   1,                                  要读取的。表示接下来进行dispatch通信使用的channel数量。
    is_token_in_rank:               [num_tokens, kNumRanks],            要读取的。表示当前rank要发送的token是否需要发送到节点内其他rank。
    channel_prefix_matrix:          [kNumRanks, num_channels],          要写入的。channel_prefix_matrix[dst_rank * num_channels + i]表示的是dispatch发送端rank rank从channel 0到channel i（包含channel i）累计要发送到rank dst_rank的token数量之和（channel级别前缀和）。
    rank_prefix_matrix_copy:        [kNumRanks, kNumRanks],             要写入的。rank_prefix_matrix_copy[i * kNumRanks + j]表示的是从rank 0到rank i累计发送到rank j的token数量之和。
    num_memset_int:                 1,                                  要读取的。num_channels * num_ranks * 4，每个channel需要清零的元数据项数即start、end、head、tail。。
    expert_alignment:               1,                                  要读取的。rank rank内的expert thread_id收到的token数量以expert_alignment为对齐单位。
    buffer_ptrs:                    kNumRanks,                          要写入的。节点内所有rank的NVLink buffer。
    barrier_signal_ptrs:            [kNumRanks, kNumRanks],             要读写的。节点内的所有rank同步所需的barrier信号。
    rank:                           1,                                  要读取的。表示当前rank的编号。
    */
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32;
    
    /*
    注意：为什么下面的代码中if中的代码那么多，而且任务那么多，却只需要sm_id == 0的一个block执行，而else中的代码那么少，任务那么少，却需要多个sm来执行？
    回答：因为下面的代码所需要处理的任务中，只涉及“一维线性”的任务，也就是说，一个sm中的多个线程足以匹配上这些任务中的每个“任务元素”。
         而else当中之所以需要多个sm，是因为它的任务中涉及“二维平面”的任务，也就是多个rank的多个channel的任务，所以多个sm和每个sm里的多个线程刚好配对上这每个“任务元素”，因此需要多个sm。
    */
    if (sm_id == 0) {
        // Barrier first
        // 同步节点内的跨 rank 的 barrier 等待所有 rank 到达，确保后续写入前各 rank 已就绪。
        // 不用设置内存屏障，因为此时只是等待所有rank到达同步点，为后续的跨rank数据写入做准备，不需要确保之前的内存操作可见性。
        // sm_id == 0 的block中的128个线程都在执行这个barrier_block
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        int *per_rank_buffer, *per_expert_buffer;
        if (thread_id < kNumRanks) {
            /* 获取当前线程负责的rank的NVLink buffer。per_rank_buffer的大小是 kNumRanks * kNumRanks，
            per_rank_buffer[i * kNumRanks + j]表示的就是从rank i发送到rank j的token数量。
            这份数据在这个核函数运行完之后在所有rank上都是一样的。

            buffer_ptrs[thread_id] 就涉及到跨rank的数据通信，包括读和写。所以需要用到barrier_block
            */
            per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]); // 注意！！！per_rank_buffer是rank thread_id的NVLink buffer
            // per_expert_buffer的大小是：kNumRanks * num_experts_per_rank
            per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
        }

        // After this loop:
        //  - `per_rank_buffer[rank * kNumRanks + thread_id]` means the number of tokens from rank `rank` to rank `thread_id`
        //  - `per_expert_buffer[rank * num_experts_per_rank + i]` means the number of tokens from rank rank to local expert i
        int num_experts_per_rank = num_experts / kNumRanks;
        if (thread_id < kNumRanks) {
            // per_rank_buffer[rank * kNumRanks + thread_id] 表示从rank rank发送到rank thread_id的token数量。
            // num_tokens_per_rank[thread_id] 表示当前的rank rank发送到rank thread_id的token数量。
            per_rank_buffer[rank * kNumRanks + thread_id] = num_tokens_per_rank[thread_id];
            #pragma unroll
            for (int i = 0; i < num_experts_per_rank; ++i)
                // 想象per_expert_buffer是行优先二维数组（大小是kNumRanks * num_experts_per_rank），第m行表示rank m要发送到rank thread_id内的各个expert的token数量。这个数据是写在rank thread_id上的。
                // num_tokens_per_expert[thread_id * num_experts_per_rank + i] 表示rank rank要发送到rank thread_id内的expert i的token数量。
                per_expert_buffer[rank * num_experts_per_rank + i] = num_tokens_per_expert[thread_id * num_experts_per_rank + i];
        }

        // Wait for all ranks to be finished
        /*
        注意：为什么这里需要同步所有rank？
        回答：上面的代码中，涉及跨GPU的数据交互，不同的rank并行执行读写buffer_ptrs[thread_id]的数据，
             因此这种类型的代码需要锁在两个barrier_block之间。
             假如不写第二个barrier_block，那么可能导致非rank rank的rank还没有读写完buffer_ptrs[rank]的数据，
             但是接下来的代码中rank rank要读写buffer_ptrs[rank]的数据，这样就会导致数据冲突。
        
        注意：为什么第一个barrier_block不需要设置内存屏障，但是这里需要呢？
        回答：因为前面的代码涉及跨rank的数据写入操作（各rank向其他rank的buffer写入），需要确保这些写入操作对所有rank可见，然后才能进行后续的读取操作。
        */
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

        // Sum per-rank counts and return to CPU
        // Also pre-compute the prefix sum for data sending
        // 下面这行代码执行后，local_per_rank_buffer[i * kNumRanks + thread_id]表示的就是从rank i发送到rank thread_id的token数量。
        auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]); // 当前GPU已经知道，每个rank发往每个rank的token数量。
        if (thread_id < kNumRanks) {
            // 经过下面的循环后，local_per_rank_buffer[i * kNumRanks + thread_id]表示的就是从rank 0到rank i发送到rank thread_id的token数量之和。
            #pragma unroll
            for (int i = 1; i < kNumRanks; ++i)
                local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
            if (thread_id == rank)
                // local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank] 就是所有rank发送到rank rank的token数量之和。
                //所以*moe_recv_counter_mapped就是所有rank发送到rank rank的token数量之和。 CPU上轮询等待这个值被更新。deep_cp.cpp 469行左右
                *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
        }

        // Sum per-experts counts and return to CPU
        auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
        if (thread_id < num_experts_per_rank) {
            int sum = 0;  // sum表示rank rank内的expert thread_id收到的token数量。
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i)
                // local_per_expert_buffer[i * num_experts_per_rank + thread_id] 表示rank i发送到rank rank内的expert thread_id的token数量。
                sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
            sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
            // moe_recv_expert_counter_mapped[thread_id] = sum 表示rank rank内的expert thread_id收到的token数量。CPU上轮询等待这个值被更新。deep_cp.cpp 474行左右
            moe_recv_expert_counter_mapped[thread_id] = sum;
        }
        /* 
        这里只是同步线程，而不用同步所有rank，是因为上面的两段代码处理的都是buffer_ptrs[rank]的数据，没有跨rank的数据交互。

        注意：为什么这里需要同步线程？
        回答：因为从第二次barrier_block到当前行的这段代码中，有两个地方涉及对thread_id的条件分支，
             而且这些条件分支之间没有读写的依赖关系，只是读写的位置是不相邻的内存空间而已，
             因此，在这两个条件分支之间不需要__syncthreads()。到那时接下来的代码中，需要依赖前面两个条件分支的执行结果，
             比如需要读local_per_rank_buffer的数据，这样就产生了依赖关系，因此需要同步线程。
             无论这里的依赖关系是读依赖还是写依赖，都需要同步线程。
        */
        __syncthreads();

        // Copy rank size prefix matrix to another tensor
        #pragma unroll
        for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)  // 这就是“block步进”的实现。以前Bob讲过的“grid步进”。
            rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];

        // rank_prefix_matrix_copy[i * kNumRanks + j]表示的是从rank 0到rank i累计发送到rank j的token数量之和。

        // Extra memset for later communication queue
        // num_memset_int 在 csrc/deep_ep.cpp 第425行：int num_memset_int = num_channels * num_ranks * 4;
        // num_channels = config.num_sms / 2  // 第325行。  4 = 每个channel需要清零的元数据项数，即start、end、head、tail。
        #pragma unroll
        for (int i = thread_id; i < num_memset_int; i += num_threads)  // 对buffer_ptrs[rank]的kNumRanks * kNumRanks之后的数据清零
            local_per_expert_buffer[i] = 0;

        // Barrier
        /*
        注意：为什么这里需要同步所有rank？我感觉好像不需要了。
        回答：数据一致性：清零操作确保后续dispatch阶段使用的通信队列元数据是干净的初始状态。
             跨rank依赖：虽然每个rank只清零自己的buffer，但这些buffer可能被其他rank在后续dispatch阶段使用。
             时序保证：确保所有rank都完成notify_dispatch阶段的所有工作后，才开始后续的dispatch或通信阶段。
             notify_dispatch阶段：统计token数量、计算前缀和、清零元数据。
             dispatch阶段：使用这些清零后的元数据进行实际的数据传输。
             如果不同步，不同rank可能在dispatch阶段开始时，某些rank的元数据本应在notify_dispatch阶段清零完毕的还未清零完毕，导致通信错误。
             
             即使notify_dispatch和dispatch两个核函数在同一个stream（comm_stream）中顺序执行，也可能存在以下情况：
             1、多GPU执行时间差：不同GPU上的notify_dispatch核函数可能完成时间不同。
             2、CPU端等待：在deep_ep.cpp中，CPU会等待notify_dispatch完成。如果CPU中要去访问其它rank的内存，就需要确保其它rank的notify_dispatch已经完成。
             3、流同步机制：虽然在同一个stream中，但CUDA的执行模型中，核函数的完成并不意味着所有内存操作立即对其他GPU可见。
        
        而且这里也需要内存屏障，即 kSyncOnly = false。
        */
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {  
        /* 
        对于这里的 num_ranks 个SM，
        每个 SM 负责一个目标 rank（dst_rank = sm_id - 1）的 channel 级 token 计数与前缀和计算。
        就是为了计算channel_prefix_matrix
        */
       
        int dst_rank = sm_id - 1;  // 每个SM处理一个destination rank的通信
        for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
            int token_start_idx, token_end_idx;  // 获取这个channel要处理的token范围
            // get_channel_task_range函数确保每个channel处理大致相同的token数
            get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

            // Iterate over tokens
            // count：当前rank（发送端）的channel_id中，需要发送到dst_rank（接收端）的token数量
            int count = 0;
            for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
            // is_token_in_rank[i * kNumRanks + dst_rank] 表示第 i 个token是否需要发送到rank dst_rank。
                count += is_token_in_rank[i * kNumRanks + dst_rank];
            count = warp_reduce_sum(count);
            if (elect_one_sync())
                /*
                channel_prefix_matrix存储在发送端rank的GPU全局内存中（每个rank都有自己的channel_prefix_matrix）。
                将count写入channel_prefix_matrix，表示当前rank（发送端）的channel_id发送到dst_rank（接收端）的token数量
                注意：这是每个rank独立计算的自己发送的数据，不是聚合所有发送端的数据
                */
                channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
        }
        __syncthreads();

        // Pre-compute prefix sum for all channels
        if (thread_id == 0) {
            #pragma unroll
            for (int i = 1; i < num_channels; ++i)
                // channel_prefix_matrix[dst_rank * num_channels + i]表示的是dispatch发送端rank rank从channel 0到channel i（包含channel i）累计要发送到rank dst_rank的token数量之和（channel级别前缀和）。
                channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];
        }
    }
}

/* 
1、据我所知，一个SM可以放置多个block，一个block最多1024个线程，但是一个SM上最多同时出现2048个线程，因此，当block配置最多的1024个线程的时候，
    SM上也可以放置两个block，而在notify_dispatch函数中，好像把block等价于SM来看待了，尤其是auto sm_id = static_cast<int>(blockIdx.x);这句。
    请问这是为什么？难道 LyricZhao 没有考虑到SM和block的关系吗？还是说为了简单起见就让block和SM一一对应？
2、在CUDA核函数中定义的int token_start_idx, token_end_idx;传入到get_channel_task_range函数中，在get_channel_task_range函数中给它们设置值，
    这样的传参方式并且修改值好像在C++中是不能这么操作的吧？C++中想在函数中修改传进来的参数的值要么用指针，要么是引用。
    这里能直接传int类型的参数进来进行修改是因为这是CUDA核函数中运行吗？
*/
void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     int num_tokens,
                     const bool* is_token_in_rank,
                     int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy,
                     int num_memset_int,
                     int expert_alignment,
                     void** buffer_ptrs,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)        \
    LAUNCH_KERNEL(&cfg,                           \
                  notify_dispatch<ranks>,         \
                  num_tokens_per_rank,            \
                  moe_recv_counter_mapped,        \
                  num_tokens_per_expert,          \
                  moe_recv_expert_counter_mapped, \
                  num_experts,                    \
                  num_tokens,                     \
                  num_channels,                   \
                  is_token_in_rank,               \
                  channel_prefix_matrix,          \
                  rank_prefix_matrix_copy,        \
                  num_memset_int,                 \
                  expert_alignment,               \
                  buffer_ptrs,                    \
                  barrier_signal_ptrs,            \
                  rank);                          \
    break

    constexpr int kNumThreads = 128;
    EP_HOST_ASSERT(num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);

    SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
    SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// launch 配置固定为 gridDim.x = 1、blockDim.x = 128
template <int kNumRanks>
__global__ void cached_notify_dispatch(
    const int* rank_prefix_matrix, int num_memset_int, void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
    // A simplified version for cached handles
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank); // 同步所有 rank

    // Copy and clean
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto ptr = static_cast<int*>(buffer_ptrs[rank]);
    #pragma unroll
    // 将 rank_prefix_matrix 复制到 buffer_ptrs[rank] 的开头，供 dispatch 核函数使用
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
        // rank_prefix_matrix[i * kNumRanks + j] 表示从 rank 0 到 rank i 发送到 rank j 的 token 数量之和
        ptr[i] = rank_prefix_matrix[i];
    /*
    每次 dispatch 之前，会清零 4 个元数据项：
    channel_start_offset（num_channels * num_ranks）
    channel_end_offset（num_channels * num_ranks）
    channel_head_idx（num_channels * num_ranks）
    channel_tail_idx（num_channels * num_ranks）
    */
    #pragma unroll
    /* num_memset_int = num_channels * num_ranks * 4
    每个 rank 在本地 GPU 上跑这一核函数，清空自己作为接收端的NVLink Buffer的元数据。
    每个 rank 上只启动 1 个 block；循环体用 “线程总数” 作为步长遍历 num_memset_int 个元素，
    因此 128 个线程刚好把num_memset_int个连续内存分工写完，不存在多 block 重复写的问题。
    */
    for (int i = thread_id; i < num_memset_int; i += num_threads)
        ptr[kNumRanks * kNumRanks + i] = 0;

    // Barrier after cleaning
    // 确保所有 rank 完成清零操作。保证 dispatch 核函数开始时元数据已清零
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void cached_notify_dispatch(const int* rank_prefix_matrix,
                            int num_memset_int,
                            void** buffer_ptrs,
                            int** barrier_signal_ptrs,
                            int rank,
                            int num_ranks,
                            cudaStream_t stream) {
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                                                   \
    LAUNCH_KERNEL(&cfg, cached_notify_dispatch<ranks>, rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 128, stream);
    SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) dispatch(int4* recv_x,
                                                           float* recv_x_scales,
                                                           int* recv_src_idx,
                                                           topk_idx_t* recv_topk_idx,
                                                           float* recv_topk_weights,
                                                           int* recv_channel_offset,
                                                           int* send_head,
                                                           const int4* x,
                                                           const float* x_scales,
                                                           const topk_idx_t* topk_idx,
                                                           const float* topk_weights,
                                                           const bool* is_token_in_rank,
                                                           const int* channel_prefix_matrix,
                                                           int num_tokens,
                                                           int num_worst_tokens,
                                                           int hidden_int4,
                                                           int num_topk,
                                                           int num_experts,
                                                           int num_scales,
                                                           int scale_token_stride,
                                                           int scale_hidden_stride,
                                                           void** buffer_ptrs,
                                                           int rank,
                                                           int num_max_send_tokens,
                                                           int num_recv_buffer_tokens) {
    /*
    recv_x:                 [num_recv_tokens, hidden],      要写入的, 接收端接收到的token组成的tensor。
    recv_x_scales:          [num_recv_tokens, num_scales],  要写入的, 接收端接收到的token的scale，每个token有num_scales个scale，每个scale表示128个float8_e4m3fn。
    recv_src_idx:           num_recv_tokens,                要写入的, 表示recv_x中每个token在这个token对应的发送端rank发送的tensor中的src_idx。recv_src_idx中会记录来自多个rank的tensor中的token。
    recv_topk_idx:          [num_recv_tokens, num_topk],    要写入的, 接收的每个token激活的专家在当前接收端rank中的专家的局部转件ID，不在这个rank中的专家就是-1。
    recv_topk_weights:      [num_recv_tokens, num_topk],    要写入的, 与recv_topk_idx对应，表示接收的每个token激活的专家在当前接收端rank中的专家的权重，不在这个rank中的专家权重就是0。
    recv_channel_offset:    [kNumRanks, num_channels],      要写入的, recv_channel_offset[responsible_rank * num_channels + responsible_channel] 表示从发送端rank responsible_rank经由channel responsible_channel发送到接收端rank的token在接收端输出数组recv_x中的给rank responsible_rank的channel responsible_channel准备的空间的起始索引。
    send_head:              [num_tokens, kNumRanks],        要写入的, send_head[token_idx * kNumRanks + responsible_rank]：token token_idx发送到rank responsible_rank的环形缓冲区的单调递增索引。
    x:                      [num_tokens, hidden],           要读取的, 从MultiHeadAttention 传入的token hidden数据。
    x_scales:               [num_tokens, num_scales],       要读取的, x_scales[i, j]：第 i 个token的第 j 组（每组128个元素）的反缩放因子。接收端用它来还原FP8数据到原始精度，还原公式：x_bf16 = x_fp8 * scale_inv。
    topk_idx:               [num_tokens, num_topk],         要读取的, 表示每个token要路由到的topk个expert的全局ID
    topk_weights:           [num_tokens, num_topk],         要读取的, 与topk_weights对应，表示每个token要路由到的topk个expert的权重
    is_token_in_rank:       [num_tokens, kNumRanks],        要读取的, is_token_in_rank[token_idx * kNumRanks + responsible_rank]表示token token_idx是否发送到rank responsible_rank.
    channel_prefix_matrix:  [kNumRanks, num_channels],      要读取的, channel_prefix_matrix[dst_rank * num_channels + i]表示的是dispatch发送端rank rank从channel 0到channel i（包含channel i）累计要发送到rank dst_rank的token数量之和（channel级别前缀和）。
    */

    /*
    问：为什么在这里，LyricZhao 要将block的id，即blockIdx.x，称作sm_id，即SM的id。这好像不对吧？因为据我所知，一个SM可以容纳2048个线程，
       一个block最多可以有1024个线程，也就是最大的一个block只能占一个SM的一半，SM至少可以同时容纳两个block，
       而SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream)设置的 kNumThreads 是一个block中有768个线程，对于这样的block，
       一个SM可以容纳2个，所以SM和block并不是对应的，既然如此，为什么192行 LyricZhao 将sm_id与block的id等同？
    答：代码中的sm_id不是真正的物理SM ID，而是一个逻辑标识符。Channel架构的逻辑模型不依赖物理SM位置！
        这个dispatch核函数是使用线程块集群（Thread Block Clusters）启动的，一个cluster中包含两个block，
        而在资源足够的情况下，GPU倾向于将同一个cluster中的多个block依次排列到相邻的SM上，这样有几个好处：
        1、共享L2 Cache（更快）
        2、使得同一个GPU内的相邻SM之间使用 NVLink-C2C 和 片上交换结构（On-Chip Switch Fabric）通信的效率更高，
        3、内存控制器协同工作
    ... 另外，在CUDA中，“相邻”有诸多好处，你懂的...
    因此，同一个channel中的在同一个cluster中的两个block，它们的blockIdx.x是相邻的，并且所在的物理SM也是相邻的。
    但是在GPU资源紧张时，Global Block Scheduler会根据GPU的资源情况，将同一个cluster中的两个block分散到不相邻的SM上，
    这样就无法保证同一个channel中的在同一个cluster中的两个block，它们的blockIdx.x是相邻的，并且所在的物理SM也是相邻的。
    因此，LyricZhao 将block的id，即blockIdx.x，称作sm_id，即SM的id，而不是真正的物理SM ID。

    num_sms 就是有多少个cluster。
    */
    const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    // Several warps are response for a single rank
    // thread_id 的范围就是[0, kNumThreads - 1]
    // num_threads_per_rank表示的是当前blcok内的多少个线程负责一个rank的通信，不是表示一个rank中有多少个线程
    // 比如：num_threads_per_rank = 768 / 8 = 96
    const auto num_threads_per_rank = kNumThreads / kNumRanks;
    const auto num_channels = num_sms / 2;
    /*
    responsible_rank 表示当前的线程负责哪个rank的通信，而连续的96个线程负责对同一个rank（responsible_rank）的通信。
    比如 responsible_rank = 213 / 96 = 2
    每组96个线程 = 3个warp (96/32 = 3)

    Block 0 (Channel 0 的发送方):
    Threads 0-95:    负责发送到 Rank 0
    Threads 96-191:  负责发送到 Rank 1
    Threads 192-287: 负责发送到 Rank 2
    ...
    Threads 672-767: 负责发送到 Rank 7

    Block 1 (Channel 0 的接收方):
    Threads 0-95:    负责从 Rank 0 (GPU 0)接收
    Threads 96-191:  负责从 Rank 1 (GPU 1)接收
    Threads 192-287: 负责从 Rank 2 (GPU 2)接收
    ...
    Threads 672-767: 负责从 Rank 7 (GPU 7)接收

    发送方还是接收方是对应着SM的，同一个SM里的所有线程要么全部用于发送，要么全部用于接收。
    而同一个block里的各个线程负责的rank是不同的，这就是为什么需要用responsible_rank来表示当前的线程负责哪个rank的通信。
    */
    const auto responsible_rank = (static_cast<int>(thread_id)) / num_threads_per_rank;
    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    /*
    假设：
    num_tokens = 4096。 当前rank中的 4096个 token 需要发送到各个rank
    num_sms = 24，num_channels = num_sms / 2 = 12，num_ranks = 8
    分解为12个channel并行处理，每个channel独立计算和发送。
    例如当前rank中的token划分：
    [0-341]:      Channel 0处理
    [342-682]:    Channel 1处理
    [683-1023]:   Channel 2处理
    ...
    [3756-4095]:  Channel 11处理

    Channel 0:  SM 0 (send) + SM 1 (recv)：
      - 统计要发送的token数
      - 执行数据传输
      - 使用独立的queue（head/tail）
    Channel 1:  SM 2 (send) + SM 3 (recv)
    Channel 2:  SM 4 (send) + SM 5 (recv)
    ...
    Channel 11: SM 22 (send) + SM 23 (recv)
    一个channel由两个SM负责，一个SM负责发送，一个SM负责接收。
    */
    const auto responsible_channel = sm_id / 2;  // channel是个逻辑概念，是由两个连续id的SM组成的，所以这句就是说当前SM属于哪个channel

    int num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

    // Calculate pointers by the specific layout
    /*
    `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
    rank_prefix_matrix[i * kNumRanks + j]表示的就是从rank 0到rank i发送到rank j的token数量之和。

    注意：channel的元数据和环形队列里存储的数据都在接收方所在的rank的buffer中。
    如果当前线程所在的SM是发送方SM：访问目标rank的buffer，写入token数据到接收方（目标rank）的NVLink buffer
    如果当前线程所在的SM是接收方SM：访问自己的buffer，从自己所在的rank的NVLink buffer中读取token数据
    */
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[is_sender ? responsible_rank : rank]) +
                                       kNumRanks * kNumRanks * sizeof(int));
    int target_rank = is_sender ? rank : responsible_rank;

    /*
    有一个总的原则：通信，是在channel这个逻辑概念维度进行的。每个channel都要知道自己和每个rank的每个channel的通信情况，包括发送和接收的offset，
    发送的offset和接收的offset都是用于接收方的channel buffers中。
    */

    // 所有rank的所有channel的总数
    auto num_channels_total = num_channels * kNumRanks;
    // 想象一共有num_channels行，每行kNumRanks列。
    // channel_rank_offset表示的是responsible_channel针对rank target_rank的偏移量。
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    // Channel buffer metadata  Channel 缓存的元数据
    // Senders are responsible for tails, and receivers are responsible for heads  发送者负责队尾，接收者负责队头，而这里的“队”都是在接收端的NVLink buffer中的内存中。
    // Stored on the receiver side  存储在接收端
    // The retired signals are actually boolean flags, but to align with 16 bytes, we make it `int64_t`
    /*
    下面的四个Buffer都可以理解为是kNumChannels * kNumRanks的矩阵，每个元素记录的是索引位置

    channel_start_offset 和 channel_end_offset表示的是接收端输出数组（recv_x，即GPU显存中的tensor）中的起始索引和结束索引，不是在NVLink buffer中的索引。
    channel_start_offset 里面的ptr指针指向的数值是当前channel（responsible_channel）要写入的一些token在接收端输出数组（recv_x，即GPU global memory中的tensor）中的起始索引。
    channel_end_offset   里面的ptr指针指向的数值是当前channel（responsible_channel）要写入的一些token在接收端输出数组（recv_x，即GPU global memory中的tensor）中的结束索引。

    channel_head_idx 和 channel_tail_idx表示的是接收端的target_rank的responsible_channel的环形缓冲区的head索引和tail索引，用于同步NVLink buffer中的数据。
    channel_tail_idx：发送端的写指针，表示发送端下一个要写入接收端的buffer_ptrs的索引。
    channel_tail_idx：接收端的读指针，表示接收端下一个要读取接收端的buffer_ptrs的索引。

    注意：下面每次定义一个Buffer，都会在Buffer的构造函数里移动ptr的位置到ptr的后num_channels_total个元素的位置，所以下面的代码的声明顺序很重要，不能乱。
    */
    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset =   Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_head_idx =     Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx =     Buffer<int>(ptr, num_channels_total, channel_rank_offset);

    // Channel data buffers, stored on the receiver side  存储在接收端
    // `x_buffers`:             kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
    // `src_idx_buffers`:       kNumChannels * kNumRanks * num_recv_buffer_tokens               * sizeof(int)
    // `topk_idx_buffers`:      kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk    * sizeof(int64_t)
    // `topk_weights_buffers`:  kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk    * sizeof(float)
    // `x_scales_buffers`:      kNumChannels * kNumRanks * num_recv_buffer_tokens * num_scales  * sizeof(float)
    // channel_x_buffers存储当前channel中的：每个rank的每个channel接收到的token hidden数据
    // channel_src_idx_buffers存储当前channel中的：每个rank的每个channel接收到的token的在发送方GPU的原始输入数据中的索引位置
    // channel_topk_idx_buffers存储当前channel中的：每个rank的每个channel接收到的token的topk_idx，即每个token要发送到的topk个expert（rank内，局部的expert ID）
    /* channel_topk_weights_buffers存储当前channel中的：每个rank的每个channel接收到的token的topk个expert各自对应的weight。
    在这topk个权重值中，如果这个token发送到这个rank的expert有m个，那么这个rank中记录的这个token的topk个权重值中就有m个权重值，
    其余（topk-m）个权重值都为0，但即使是0，也要记录着。
    除非这个token完全没有发送到这个rank，那这个rank的环形缓冲区x_buffers中不会记录这个token的数据，src_idx_buffers、topk_idx、topk_weights和x_scales也都不会记录这个token的相应数据。
    */
    // channel_x_scales_buffers存储当前channel中的：每个rank的每个channel接收到的token的缩放因子

    // channel_x_buffers：通过 NVLink 可以被其他 GPU 直接写入。固定大小的环形队列，循环复用内存。高速、低延迟，但容量有限。需要通过 head_idx 和 tail_idx 同步。
    auto channel_x_buffers =            Buffer<int4>      (ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4,  channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers =      Buffer<int>       (ptr, num_channels_total * num_recv_buffer_tokens,                channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_idx_buffers =     Buffer<topk_idx_t>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk,     channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_topk_weights_buffers = Buffer<float>     (ptr, num_channels_total * num_recv_buffer_tokens * num_topk,     channel_rank_offset * num_recv_buffer_tokens * num_topk);
    auto channel_x_scales_buffers =     Buffer<float>     (ptr, num_channels_total * num_recv_buffer_tokens * num_scales,   channel_rank_offset * num_recv_buffer_tokens * num_scales);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    /*
    在CUDA kernel中，extern __shared__声明了一个外部shared memory数组，其含义是：
    extern: 表示这个变量的存储空间由外部提供（CUDA runtime负责分配和管理）。
    不占用编译时确定的内存空间，大小在kernel启动时动态指定，就是下面的sharedMemSize。
        """
        // 声明：告诉编译器存在这样一个数组，但不分配空间
        extern __shared__ uint8_t smem_buffer[];

        // 使用：CUDA runtime在启动kernel时分配实际空间
        kernel<<<grid, block, sharedMemSize>>>(...);
        """
    uint8_t：按字节访问，方便后续的指针运算和地址计算。
    __align__(1024)表示内存对齐要求:
        smem_buffer的起始地址必须对齐到1024字节边界, 也就是 1KB对齐。
    
    TMA (Tensor Memory Accelerator)  的特点:
    1、TMA是专门设计用于在global memory和shared memory之间进行异步bulk传输的硬件单元。它不直接支持global memory到global memory的传输。
    2、TMA引擎直接集成在shared memory控制器中，包括功能耦合、数据路径和控制逻辑的紧密集成。
       shared memory作为GPU上的高速缓存，是TMA进行高效数据传输的必要中间缓冲区。这是Hopper架构引入TMA的一个关键设计亮点。
       参考: https://osvicyu5w5.feishu.cn/wiki/PYS5wKQS7iOa1VkNuGqcpYe7n9b
    3、TMA的关键要求: 必须1024字节（1KB）对齐，这是TMA硬件的强制要求。
    */
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    // 将隐藏层数据分成两半
    auto half_hidden_int4 = hidden_int4 / 2;  // hidden_int4 = 4096 / 16 / 2 = 128
    // 计算一半隐藏层数据占用的字节数。用以确定TMA缓冲区的数据存储空间大小。  TMA的cp.async.bulk指令规定: dstMem和srcMem地址和传输的数据大小都必须按16字节对齐。
    auto half_hidden_bytes = half_hidden_int4 * static_cast<int>(sizeof(int4));  // 128 * 16 = 2048
    /*
    thread_id / 32: 计算warp ID（每个warp有32个线程）
    kNumTMABytesPerWarp: 8192, 即8KB。每个warp分配8192字节的TMA缓冲区（必须位于共享内存中）。每个warp对共享内存的使用不能超过这个大小。
    tma_buffer的特点: 每个warp独立拥有自己的TMA缓冲区，避免warp间竞争。共享内存 smem_buffer 被划分为warp-sized个区域。
    tma_buffer: 计算当前warp的TMA缓冲区起始地址。
    */
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
    /*
    tma_mbarrier: 计算mbarrier（内存屏障）的存储位置。注意这里将字节指针转换为uint64_t指针，这8字节是mbarrier的空间占用。
    TMA要求：mbarrier必须与TMA缓冲区在同一shared memory段中，便于硬件管理同步。
    加上half_hidden_bytes表明 tma_mbarrier 紧跟在数据缓冲区之后。
    +--------+-------------------+--------+-------------------+-----------------------+----------+
    | Phase  | Arrive_Count      | Lock   | Transaction_Count | Expected_Arrive_Count | Reserved |
    | (1bit) | (20bit) [int20_t] | (1bit) | (21bit) [int21_t] | (20bit) [int20_t]     | (1bit)   |
    +--------+-------------------+--------+-------------------+-----------------------+----------+
    | bit 63 | bit 62~43         | bit 42 | bit 41~21         | bit 20~1              | bit 0    |
    +--------+-------------------+--------+-------------------+-----------------------+----------+
    mbarrier在数据表示上表达为一个64bit的整数类型数据，其内部各个域的定义如上图所示，整体分为六个域，分别是:
        最低位的第0比特的保留域，
        第1比特到第20比特的Expected Arrive Count域，
        第21比特到第41比特的Transaction Count域，
        第42比特位置的Lock域，即错误锁标志位，为0表示没有错误，为1表示有错误。
        第43比特到第62比特的Arrive Count域，
        第63比特位置的Phase域。
    */
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + half_hidden_bytes);
    /*
    tma_phase: 初始化TMA操作的相位变量。用于在后续的TMA等待操作中判断异步传输是否完成。
    TMA特点：TMA是异步操作，相位用于跟踪操作完成状态。
    */
    uint32_t tma_phase = 0;
    if (elect_one_sync()) {  // TMA要求只选择一个线程执行初始化，避免重复初始化。
        /*
        mbarrier_init是创建一个CTA级的mbarrier，PTX指令中已经说明了，而不是cluster级的mbarrier。
        mbarrier_init(tma_mbarrier, 1)中的arrive_count = 1实际上设置的是上面的Expected_Arrive_Count，
        表示mbarrier期望有且仅有1个TMA写共享内存操作，mbarrier_wait函数到时候只用等待一个TMA操作完成，屏障就会打开。
        读共享内存不需要mbarrier。
        而每个TMA写共享内存操作的线程到底需要各自传输多少字节的数据，
        则是由每个线程调用 mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes)来告知mbarrier的。
        */
        mbarrier_init(tma_mbarrier, 1);
        /*
        fence.mbarrier_init.release.cluster: release前不到后。 cluster作用域，确保cluster内所有SM都能看到一致的初始化状态。
        因为使用mbarrier的tma_load_1d是将全局内存的数据写入shared::cluster，而不是shared::cta。
        发布内存屏障，确保之前的mbarrier.init操作对所有线程可见（集群内的其他SM）。 
        */
        fence_barrier_init();
        /*
        确保一个warp实际使用到的一个token的hidden数据和mbarrier占用的字节数不能超过原本给这个warp分配的TMA缓冲区大小kNumTMABytesPerWarp。
        */
        EP_DEVICE_ASSERT(hidden_int4 % 2 == 0 and half_hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    /*
    同步整个warp的所有线程。TMA要求: 确保所有线程都能看到mbarrier的初始化结果。
    */
    __syncwarp();
#endif

    /*
    发送端将输入的tensor数据，即token hidden数据，发送到接收端的channel_x_buffers。
    在发送端真正发送之前，先写入不为0的值到接收端的channel_start_offset和channel_end_offset，
    用于高速接收端的输出tensor（recv_x，即GPU显存中的tensor）中的起始索引和结束索引部分可以开始写入数据了。
    虽然在接收端中使用自旋等待循环（spin-wait loop），直到channel_start_offset的值不是0就退出循环，
    也就是发现了我接收端这边的输出tensor（recv_x，即GPU显存中的tensor）中的起始索引和结束索引部分可以开始写入数据了，
    但是！
    接收端中如果不满足cached_channel_head_idx != cached_channel_tail_idx，即接收端的对应recv_x的区间需要读取数据的环形缓冲区中没有数据，
    那么接收端就会继续等待，直到发送端写入数据到接收端的环形缓冲区中，此时接收端才可以读取这部分数据到输出tensor（recv_x，即GPU显存中的tensor）中的起始索引和结束索引部分。
    */
    if (is_sender) {  // 此时，responsible_rank是接收端，当前rank是发送端
        // Workers for sending
        constexpr int num_send_warps = kNumThreads / 32;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        const auto send_thread_id = thread_id;
        // num_threads_per_rank表示的是当前blcok内的多少个线程负责一个rank的通信，不是表示一个rank中有多少个线程
        // send_warp_id_in_rank表示当前线程所在的warp用于给接收端的rank通信的局部warp id，对于同一个接收端 rank的warp id的范围都是[0, num_send_warps_per_rank - 1]。
        const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);  // kNumRanks一般最大不超过8，即一个节点内最多8个GPU
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
        // NOTES: this is for distinguishing zero tokens
        if (send_warp_id_in_rank == 0 and elect_one_sync()) {
            /* channel_prefix_matrix存储在发送端rank的GPU全局内存中（每个rank都有自己的channel_prefix_matrix）。
            channel_prefix_matrix想象成[kNumRanks * kNumChannels]的二维数组，第i行第j列表示当前rank（发送端）从channel 0到channel j（包含channel j）累计发送到接收端rank i的token数量（前缀和）。
            在notify_dispatch函数中（发送端执行），每个rank计算自己发送到各个接收端rank的每个channel的token数量，然后计算前缀和。
            发送端的channel和接收端的channel都是逻辑概念，是对等的，用于记录发送的token数据的索引位置。
            在发送端中，确定了是哪个SM，就确定channel_id（responsible_channel），而发送端发送的token数据也会发送到接收端的同一个channel_id（responsible_channel）中。
            */
            int value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1] : 0;
            // 将value值写入到channel_start_offset的buffer中，表示当前发送端rank的responsible_channel要发送到接收端responsible_rank的token数据在接收端recv_x中的起始索引。
            // 这里不使用0作为有效数据，是因为接收端通过判断channel_start_offset.buffer()是否为0来确定发送端是否已写入数据，避免了"数据未写入"和"写入值为0"之间的歧义。
            st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
            // 这个value值表示包括当前channel在内，累计要发送多少个token到接收端responsible_rank（包含responsible_channel）
            value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
            st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1); // 将结束偏移量写入到接收端的 channel_end_offset 缓冲区
        }
        __syncwarp(); // warp内的线程同步，确保第0号线程写入的offset值对warp内所有线程可见。原因：后续的发送操作可能需要这些offset信息，需要等待写入完成

        // Get tasks
        int token_start_idx, token_end_idx;  // token_start_idx表示当前channel要发送的token的开始索引，token_end_idx表示当前channel要发送的token的结束索引
        // num_tokens个token被均分到num_channels个channel处理，而当前的channel是responsible_channel，所以token_start_idx和token_end_idx表示的是当前channel要发送的token的开始索引和结束索引。
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

        // Iterate over all tokens and send by chunks
        /*
        cached_channel_tail_idx：环形缓冲区写指针（缓存）。[0, ∞)递增。之所以叫tail，是因为缓冲区的尾部写入，头部消费。
        环形队列相关判断逻辑：
            head == tail：队列为空
            head != tail：队列有数据
            tail - head：可读的token数量（无需考虑环绕）
        为什么不需要担心环绕？
            head 和 tail 是单调递增的计数器，不是slot索引。
        head 和 tail 的值域：[0, ∞)
        slot 的值域：[0, num_recv_buffer_tokens-1]
        转换公式：slot_idx = (tail or head) % num_recv_buffer_tokens

        发送端写入token的情况：

        1、发送端写三个token到接收端的环形缓冲区，接收端还没有消费
        ┌───┬───┬───┬───┬───┬───┬───┬───┐
        │ T0│ T1│ T2│   │   │   │   │   │
        └───┴───┴───┴───┴───┴───┴───┴───┘
          ↑           ↑
        head=0     tail=3

        2、接收端消费了三个token，并更新了head指针
        ┌───┬───┬───┬───┬───┬───┬───┬───┐
        │ T0│ T1│ T2│   │   │   │   │   │  (已读取，但还在内存中。队列为空，可以直接写入)
        └───┴───┴───┴───┴───┴───┴───┴───┘
                      ↑
                    head=3
                    tail=3

        3、环形队列的环绕情况。发送端继续写入token 3-8：
        发送端写入 token 3-8：
        slot 3, 4, 5, 6, 7（正常写入）
        slot 0（环绕！）
        tail = 3 + 6 = 9

        环形缓冲区（逻辑视图）：
        ┌───┬───┬───┬───┬───┬───┬───┬───┐
        │ T8│   │   │ T3│ T4│ T5│ T6│ T7│
        └───┴───┴───┴───┴───┴───┴───┴───┘
                      ↑                    ↑(环绕到slot 0)
                    head=3               tail=9

        实际slot位置：T8 → slot (9-1) % 8 = 0  ← 环绕！

        判断：head(3) != tail(9) → 有数据
        可读数量：9 - 3 = 6 个token

        接收端读取环绕的数据：
        num_recv_tokens = tail - head = 6;
        for (int chunk_idx = 0; chunk_idx < num_recv_tokens; chunk_idx++) {
            // 关键的模运算！
            int token_idx_in_buffer = (head + chunk_idx) % 8;

            // chunk_idx = 0: (3+0) % 8 = 3 → 读取 slot 3 (T3)
            // chunk_idx = 1: (3+1) % 8 = 4 → 读取 slot 4 (T4)
            // chunk_idx = 2: (3+2) % 8 = 5 → 读取 slot 5 (T5)
            // chunk_idx = 3: (3+3) % 8 = 6 → 读取 slot 6 (T6)
            // chunk_idx = 4: (3+4) % 8 = 7 → 读取 slot 7 (T7)
            // chunk_idx = 5: (3+5) % 8 = 0 → 读取 slot 0 (T8) ← 环绕！
        }
        cached_channel_head_idx = head + num_recv_tokens = 9;  // 更新接收端的 head

        为什么 cached_channel_tail_idx 一开始是 0 ？
        因为第一次循环时：
        cached_notify_dispatch 会 清零所有接收端中对应的各个发送端rank的 channel 的 head 和 tail。
        每次 dispatch 都是新的开始。每次 dispatch 之前，所有 channel 的元数据都被清零。

        执行顺序：
            1. cached_notify_dispatch() 被调用
            └─ 清零所有 channel 的 head 和 tail（包括 channel_head_idx = 0）

            2. dispatch() 被调用
            └─ cached_channel_tail_idx = 0（初始化）
            └─ num_used_slots = 0 - 0 = 0（计算）

            3. 循环中更新 cached_channel_tail_idx
            └─ cached_channel_tail_idx++（递增）
            └─ st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx)（更新全局 tail）
        */
        int cached_channel_tail_idx = 0;
        /* 这是当前channel要发送的一个个的token，一个发送的channel对应一个SM，也就是一个block，
        但是当前的代码是在线程维度执行的，肯定不是为了每个线程都去发送这block中需要发送的所有token，那么是如何实现这一点的呢？
        答案就是在这个for循环里面使用各种条件判断，实现负载均衡。这样，虽然每个线程中都会执行一次关于channel级别的循环，但是每个线程中只会发送一部分token。
        */
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            // NOTES: the head index received by different warps may not be the same
            auto start_time = clock64();
            if (elect_one_sync()) {  // 先确保接收端的NVLink buffer有足够的slot可以接收token，否则等待。
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    /* 一开始计算时，num_used_slots = 0 - 0 = 0 
                    这里读取channel_head_idx的head值，是为了等待dispatch的接收端从head消费数据消费到哪个head位置了，如果dispatch接收端没有消费完发送端之前发送的数据，
                    那么if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)就返回false，不退出循环，继续等待。
                    所以在dispatch核函数内，发送端一批批地发送，接收端一批批地消费。发送端每次发送都得等到满足if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)才能发送。
                    */
                    int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    /* 写入保护
                    这个写入保护是为了保护使得 combine 阶段从send_data中读取缓冲区的cached_channel_tail_idx索引进而读取缓冲区中的token不出错
                    即使环形缓冲区会循环使用，也不会读取到错误数据，因为：
                        dispatch 写入前检查 head_idx，避免覆盖未读取的数据
                        combine 读取前等待 tail_idx > expected_head，确保数据已写入
                        combine 读取后更新 head_idx，释放已读取的 slot
                    如果接收端还有足够的slot可以接收token，则退出循环，否则继续等待。实现这种等待的方式是这个while循环之后的__syncwarp()，让所有线程都同步到这个位置。
                    */
                    if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {   // 如果超时，则打印错误信息并退出程序。prod环境是100秒，debug环境是10秒
                        printf("DeepEP timeout for dispatch senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();

            // 到这里，说明接收端的NVLink buffer有足够的slot可以接收token，可以开始发送token了。

            int chunk_token_idx = 0;
            // 这个while循环其实可以去掉，把chunk_token_idx这个变量也从代码中去掉，直接用token_idx来控制循环。
            while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
                // NOTES: for the same token, the warp assigned to save `send_head` may be different from the warp assigned to send the
                // following data
                // 说明：对于同一个token，负责写send_head的warp可能与负责发送数据的warp不同。  原因：使用轮询（round-robin）方式在多个warp之间分配任务
                // 只有warp的第0号线程执行（避免重复写入）。轮询分配：每个warp的lane_id为0的线程负责一个token的send_head的写入。例如：有3个warp（id=0,1,2），token_idx=5，则5 % 3 = 2，所以warp 2负责写入send_head。
                if (token_idx % num_send_warps_per_rank == send_warp_id_in_rank and elect_one_sync())
                    /*
                    send_head 数组作用：记录每个token在NVLink buffer中的位置。大小：[num_tokens, kNumRanks]。
                    send_head[token_idx * kNumRanks + responsible_rank]：token token_idx发送到rank responsible_rank的环形缓冲区的单调递增索引。
                    如果该token要发送到该rank，则写入cached_channel_tail_idx，记录这个token在接收端环形缓冲区中的位置。写入-1表示不发送。
                    后续可以用来查询某个token被发送到哪个位置
                    */
                    send_head[token_idx * kNumRanks + responsible_rank] =
                        is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;

                // Skip if not selected.  跳过不需要发送的token。注意：跳过的token不计入chunk_token_idx（因为没有实际发送）
                if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
                    token_idx++;
                    continue;
                }

                // Get an empty slot
                /*
                dst_slot_idx 作用：在NVLink环形缓冲区中分配一个空闲slot。
                实际写入缓冲区的物理位置也是用同样的取模运算。这个缓冲区大小num_recv_buffer_tokens和combine阶段的接收端缓冲区大小必须是一样的。
                num_recv_buffer_tokens：表示接收端NVLink buffer中对于每个rank的每个channel可以接收的token数量。
                cached_channel_tail_idx：表示发送端下一个要写入接收端的buffer_ptrs的索引，从队列头到队列尾，地址值递增。每发送一个token就递增1。最终会同步到channel_tail_idx。
                % num_recv_buffer_tokens：模运算实现环形队列。
                num_recv_buffer_tokens 含义：环形缓冲区的总容量（token数量）。例如：如果为128，则slot索引在[0, 127]之间循环
                */
                int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;
                // 负载均衡：让所有warp都有工作，让不同warp处理不同的token。
                // 同一个warp内的线程
                // 当前线程中，num_send_warps_per_rank和send_warp_id_in_rank是固定的，而cached_channel_tail_idx是在循环中变化的。
                if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
                    // Copy data
                    /*
                    shifted_channel_x_buffers：目的地址，NVLink buffer中的地址，偏移到第dst_slot_idx个slot。
                    需要通过 head_idx 和 tail_idx 同步，tail_idx的同步在发送代码 if (is_sender) { ... } 的最后一步。
                    hidden_int4：每个token的hidden size（以int4为单位，在CUDA中，int4 是一个向量类型，不是"4字节整数"）
                    int4 在这里只是作为内存容器，实际存储的是 nv_bfloat16 的位模式。
                    // CUDA的int4定义：
                    struct int4 {
                        int x, y, z, w;  // 4个int
                    };
                    之所以涉及int4结构，是因为现代 GPU（如 NVIDIA GPU）支持向量化内存操作，可以一次加载/存储 16 字节（128 位）
                    例如：hidden_int4 = 256 = 256个int4 = 256 * 16 bytes = 4096 bytes
                    */
                    auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                    // shifted_x: 源地址，发送端的输入tensor的起始地址（int4*类型），即token hidden数据在GPU显存中的地址，偏移到第token_idx个token的地址。
                    auto shifted_x = x + token_idx * hidden_int4;
                    // 关于UNROLLED_WARP_COPY的解释：参见：https://osvicyu5w5.feishu.cn/wiki/Wv2bwLGDmipgapkhOdrcZCtynrg#share-QMc4d5u2xo82cxxT8VacmkTuntg
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);
                    // UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4,       shifted_buffer_x_int4, ld_nc_global, st_na_global);

                    // Copy source index
                    if (elect_one_sync()) {  // 只需要一个线程写入
                        // 记录这个token在原始输入中的索引。接收端可以知道这个token来自发送端的哪个位置
                        // channel_src_idx_buffers[dst_slot_idx]：第dst_slot_idx个slot的token在发送端的输入tensor中的源索引。
                        channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);
                    }

                    // Copy `topk_idx` and `topk_weights` with transformed index
                    if (lane_id < num_topk) {  // 只有前num_topk个线程参与（通常num_topk ≤ 32）
                        // Top-k index
                        // recv_expert_begin：接收端rank的第一个expert ID。
                        // recv_expert_end：接收端rank的最后一个expert ID + 1。编程中，往往区间右边开，所以是 + 1。
                        int recv_expert_begin = responsible_rank * num_experts_per_rank,
                            recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;
                        /*
                        topk_idx, shape: [num_tokens, num_topk]，类型为 torch.int64。表示每个token要路由到的topk个expert的全局ID
                        */
                        auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id);  // 此时得到的idx_value是全局expert ID。
                        // 索引转换：将全局expert ID转换为接收端的局部expert ID。
                        // 例如：发送端token要去expert 18，接收端是rank 2（experts [16,23]），转换后：18 - 16 = 2（局部的expert ID）
                        /*
                        举个例子：
                        一个 token 的 topk_idx = [expert_0, expert_1, expert_2, expert_3]（全局 expert ID）
                        expert_0 和 expert_1 在 rank 1 上
                        expert_2 和 expert_3 在 rank 2 上
                        当前接收端是 rank 1
                        那么 rank 1 接收这个 token 时，rank 1 上的关于这个token在接收端 recv_topk_idx 上会记录为：
                        recv_topk_idx[dst_slot_idx * num_topk + 0] = 局部 expert ID（如果 expert_0 在 rank 1 的范围内）
                        recv_topk_idx[dst_slot_idx * num_topk + 1] = 局部 expert ID（如果 expert_1 在 rank 1 的范围内）
                        recv_topk_idx[dst_slot_idx * num_topk + 2] = -1（因为 expert_2 不在 rank 1 上）
                        recv_topk_idx[dst_slot_idx * num_topk + 3] = -1（因为 expert_3 不在 rank 1 上）

                        注意：expert在各个expert上的分布是均匀的，而且全局序号是按照rank的id从小到大排序的。
                        注意：idx_value存放的是在接收端的expert的局部expert id，即idx_value的范围是 [0, num_experts_per_rank - 1]，或者-1。
                        idx_value 是局部专家编号
                        */
                        idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;
                        /*
                        dst_slot_idx的范围是 [0, num_recv_buffer_tokens - 1]，所以 dst_slot_idx * num_topk + lane_id 的范围是 [0, num_recv_buffer_tokens * num_topk - 1]
                        设置第dst_slot_idx个slot的第lane_id个expert的topk_idx为idx_value。
                        之所以让channel_topk_idx_buffers和channel_topk_weights_buffers的列数都是num_topk，是因为不管怎样，
                        一个token最多只能路由到num_topk个专家，何况一个接收端rank上的专家中激活的数量最多也不会超过num_topk，所以就把列数设置为num_topk
                        
                        注意：idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id)是按照全局激活专家来查找的，每次肯定是一个激活的全局专家ID。
                        但是这个专家是否在当前接收端rank的num_experts_per_rank个专家上就不一定了，但是：
                            注意: 如果在，那么这个专家所在channel_topk_idx_buffers[dst_slot_idx * num_topk]中的位置肯定和激活的全局专家的位置是一样的！
                                 这也是为什么在combine的接收端中，可以在每个land中使用for (int i = 0; i < num_topk_ranks; ++i)来遍历num_topk_ranks个rank求和，从而得到token token_idx的全局第land_id个权重！
                        */
                        channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

                        // Top-k weights
                        auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                        weight_value = (idx_value >= 0) ? weight_value : 0.0f;
                        // channel_topk_weights_buffers中不是topk的权重，设置为0。
                        // 由下标用到dst_slot_idx可以看出，channel_x_buffers、channel_src_idx_buffers、channel_topk_idx_buffers、channel_topk_weights_buffers和channel_x_scales_buffers都是按slot_idx索引的，是一一对应的。
                        channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = weight_value;
                    }

                    // Copy `x_scales`
                    #pragma unroll
                    /* num_scales = hidden / 128 = 56， 含义参见per_token_cast_to_fp8函数中x_view在dim=1上的shape。
                    x_scales.shape: (num_tokens, num_scales) = (4096, 56)
                    x_scales[i, j]：第 i 个token的第 j 组（每组128个元素）的反缩放因子。接收端用它来还原FP8数据到原始精度，还原公式：x_bf16 = x_fp8 * scale_inv
                    += 32：Warp 优化的典型标志。关于warp优化的好处，参见：https://osvicyu5w5.feishu.cn/wiki/Ry0zwtXGZi5m0Mkob8scntOFnEd#share-V9Elda72CoqoRvxjpxwcLl7nnVf

                    Stride（步幅）：在多维数组中，从一个元素在某一维度移动到这个维度的下一个元素、而且其它维度保持不变时，需要跨越的元素数量。这里的“下一个元素”是指在当前维度上，索引值加1的元素。
                    这个步幅跟实际的内存布局有关。
                    scale_token_stride: token 维度的步幅 = static_cast<int>(x_scales->stride(0));
                        如果是x_scales行优先布局，那么 scale_token_stride = num_scales，移动到下一个token需要跨越 num_scales 个元素。
                        如果是x_scales列优先布局，那么 scale_token_stride = 1，移动到下一个token需要跨越 1 个元素
                    scale_hidden_stride: scale 维度的步幅 = static_cast<int>(x_scales->stride(1));
                        如果是x_scales行优先布局，那么 scale_hidden_stride = 1，移动到下一个scale需要跨越 1 个元素。
                        如果是x_scales列优先布局，那么 scale_hidden_stride = num_scales，移动到下一个scale需要跨越 num_tokens 个元素。

                    为什么需要支持不同布局？
                        灵活性：上层可以选择最优的数据组织方式
                        性能优化：不同访问模式下，不同布局性能不同
                        兼容性：与PyTorch的stride机制完全兼容

                    channel_x_scales_buffers中不用考虑像Tensor那样的内存布局，因此它的下标中的“列数”就是num_scales。
                    */
                    for (int i = lane_id; i < num_scales; i += 32) {  // 加速比：32倍
                        auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                        channel_x_scales_buffers[dst_slot_idx * num_scales + i] = __ldg(x_scales + offset);
                    }
                }

                // Move token index
                chunk_token_idx++, token_idx++;
            }

            // Move tail index
            // NOTES: here all warps should share the same new tail
            /*
            asm 是一个关键字，用于在 CUDA C++ 代码中直接嵌入 PTX 汇编指令。ASseMbly： 汇编。指代的是汇编语言本身。
            volatile 关键字的含义：1、防止编译器优化；2、保证执行顺序；3、防止指令重排序；4、防止缓存不一致。等等...
            volatile 配合 bar.sync 保证内存操作的可见性，确保barrier前的写操作对barrier后的读操作可见。

            PTX官方文档：bar.sync  barrier_id, thread_count;
            功能：Named Barrier Synchronization
            作用：在指定的barrier上同步CTA内指定数量的线程

            参数说明：
            1、barrier_id（对应 %0，即 responsible_rank）
                Barrier标识符（0-15，Compute Capability 8.0+支持16个命名barrier）。
                用于区分不同的同步组，同一个barrier_id的线程会在同一个barrier上等待。
            2、thread_count（对应 %1，即 num_threads_per_rank）
                参与同步的线程数量，必须精确等于实际到达barrier的线程数，如果不匹配会导致死锁或未定义行为。

            每个SM有16个硬件barrier（Barrier Units）。
            bar.sync  barrier_id, thread_count; 的执行流程：
                线程到达 bar.sync 指令：
                Step 1: 线程检查 barrier_id（例如：2）
                    访问 Barrier 2 的计数器counter[2]
                Step 2: 原子递增计数器
                    counter[2] += 1
                Step 3: 读取计数器值
                    current_count = counter[2]
                Step 4: 判断
                    if (current_count < thread_count):
                        进入等待状态（阻塞）
                    else:  // 最后一个到达的线程
                        重置计数器：counter[2] = 0
                        唤醒所有等待的线程
                Step 5: 所有线程继续执行
            */
            // responsible_rank = (static_cast<int>(thread_id)) / num_threads_per_rank;
            // 所以这里是使用Barrier同步，确保所有负责同一个接收端rank的线程都完成了数据拷贝，再更新tail指针。
            // 如果不用这个，那么当有的线程还没有完成数据拷贝，tail指针还没有确定最终值，就会导致数据不一致。
            // 而且这里也不需要使用 __syncthreads()（整个block barrier）等待所有的rank完成，因为每个rank的线程是独立的，互不干扰。
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            // send_warp_id_in_rank表示当前线程所在的warp用于给接收端的rank通信的局部warp id
            // 下面的条件表示修改接收端的channel_tail_idx，有且仅有发送端处理这个接收端的第一个warp的第一个线程去修改。
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
        }
    } else {  // 此时，responsible_rank是发送端，当前rank是接收端
        // Workers for receiving and copying into buffer
        constexpr int num_recv_warps = kNumThreads / 32;  // 当前blcok的warp数，768 / 32 = 24个warp
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;  // 当前channel处理一个rank需要多少个warp，24 / 8 = 3个warp
        const auto recv_thread_id = thread_id;
        // 当前线程在它要处理的发送端rank（responsible_rank）中的局部线程id，连续num_threads_per_rank个线程负责的是同一个rank的通信。
        const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
        const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32);
        EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0); // 当前blcok的warp数必须是kNumRanks的倍数

        // Calculate offset first
        // rank_prefix_matrix[i * kNumRanks + j]表示的就是从rank 0到rank i发送到rank j的token数量之和。
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        // rank_offset表示的就是从rank 0到rank (responsible_rank - 1)发送到rank rank的token数量之和。注意这里的减一。
        // 当responsible_rank是0时，表示rank的id比0小的所有rank要发送到rank 0的token之和为0，因为不存在rank的id比0小的rank，所以rank_offset是0。我啰嗦一下。
        int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;

        // Receive channel offset
        int total_offset, num_tokens_to_recv;
        if (elect_one_sync()) {
            /*
            自旋等待循环（spin-wait loop），直到total_offset的值不是0就退出循环。
            关键理解：赋值表达式的值是赋值的值。int x; int y = (x = 5);  // x被赋值为5，y也被赋值为5
            */
            while ((total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0);
            while ((num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0);
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            if (recv_warp_id_in_rank == 0)  // 每个rank-channel对只需记录一次offset
                /*
                recv_channel_offset的分配位置：auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
                此时responsible_rank是发送端，responsible_channel是接收端接收的channel。
                所以这里记录的就是当前接收端中处理发送端responsible_rank的channel responsible_channel的offset。
                recv_channel_offset[i, j] = 从rank i接收的数据中，channel j要写入的token在接收端输出数组recv_x中的给rank i准备的空间的起始索引。
                recv_channel_offset[responsible_rank * num_channels + responsible_channel] 表示从发送端rank responsible_rank经由channel responsible_channel发送
                到接收端rank的token在接收端输出数组recv_x中的给rank responsible_rank的channel responsible_channel准备的空间的起始索引。
                */
                recv_channel_offset[responsible_rank * num_channels + responsible_channel] = total_offset;
            /* num_tokens_to_recv = end_offset - start_offset = num_tokens_to_recv - total_offset
            num_tokens_to_recv表示发送端responsible_rank的responsible_channel要发送到接收端rank的token数量。
            */
            num_tokens_to_recv -= total_offset;
        }
        /*
        __shfl_sync(0xffffffff, total_offset, 0) 的作用：
        从 lane_id == 0 的线程读取 total_offset 的值（无论调用线程的 lane_id 是什么），将这个值广播给 warp 内的所有线程。
        __shfl_sync函数有sync后缀，也就是说需要等到一个warp中的32个线程都到达这一行代码，才能一起都执行__shfl_sync函数，
        否则即使有一个线程没有达到这个函数，就不能执行，而且也得等到被mask的线程也到达__shfl_sync函数这里，才能一起执行__shfl_sync函数，而被mask的线程就不参与此次从land_id为0的线程中读取total_offset的数据。
        */
        total_offset = __shfl_sync(0xffffffff, total_offset, 0);
        /* 看到这行，太激动了，印证了我之前理解的内容，都串起来了。在此记录一下。
        rank_prefix_matrix，shape:[kNumRanks, kNumRanks]，是rank的buffer_ptrs指针最开始指向的内容，
        rank_prefix_matrix[i * kNumRanks + j]表示的就是从rank 0到rank i发送到rank j的token数量之和。
        rank_prefix_matrix这份数据在节点内的所有rank都是一样的。

        channel_start_offset，shape:[num_channels, kNumRanks]，存储在接收端，channel_start_offset 记录的是
        responsible_channel 要传输的一些token在接收端输出数组（recv_x，即GPU global memory中的tensor）中的起始索引。

        在接收端的输出数组（recv_x，即GPU global memory中的tensor，用于输入给expert）中，来自不同发送端rank的token数据是紧邻在一起的，
        而同一个rank的不同channel的token数据是按channel序号递增的顺序紧邻在一起的，
        就是说这个tensor中，先存放rank 0的channel 0的token数据，再存放rank 0的channel 1的token数据，以此类推，直到rank 0的channel num_channels - 1的token数据存放完。
        然后存放rank 1的channel 0的token数据，再存放rank 1的channel 1的token数据，以此类推，直到rank 1的channel num_channels - 1的token数据存放完。
        以此类推，直到rank (kNumRanks - 1)的channel (num_channels - 1)的token数据存放完。

        因此，当需要把某个发送端的某个channel的数据写入到接收端的输出数组（recv_x，即GPU global memory中的tensor）中时，
        需要先找到这个发送端rank的前一个rank在接收端中的结束位置的偏移量，即rank_offset，
        然后再进一步找到发送端的responsible_channel在接收端recv_x中的起始索引，即channel_start_offset.buffer()，也就是前面的total_offset。
        最后将二者相加，就是在接收端的输出数组（recv_x，即GPU global memory中的tensor）中，这个发送端的这个channel的数据在接收端recv_x写入的起始索引。
        */
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

        // Shared tail indices for different warps
        /*
        关于在共享内存中使用 volatile 的意义：
        1. 防止编译器优化：
           - 编译器可能会认为共享内存的值不会改变，因此可能会将读取优化到寄存器中缓存
           - 使用 volatile 可以强制编译器每次都从内存中读取，而不是使用寄存器缓存的值
        2. 确保内存可见性：
           - 虽然有 barrier 同步（asm volatile("bar.sync ...")），但 volatile 可以确保在 barrier 之后，所有线程都能看到最新写入的值
           - 这特别重要，因为对shared_channel_tail_idx的写入和读取发生在不同的线程上。
        3. 防止编译器重新排序：
           - volatile 可以防止编译器对内存访问进行重新排序优化
           - 确保写入和读取的顺序符合代码的逻辑顺序
        如果没有 volatile，编译器可能会：
        - 将读取的值优化到寄存器中，导致读取到旧值
        - 或者重新排序内存访问，导致读取和写入的顺序不符合预期

        注意：volatile 不能替代 barrier 同步。这里既有 barrier（第753行），也有 volatile，它们共同确保了正确的内存可见性和同步。
        */
        __shared__ volatile int shared_channel_tail_idx[kNumRanks];  // 记录responsible_channel在每个rank中的环形缓冲区的tail索引。

        auto start_time = clock64();
        int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
        while (num_tokens_to_recv > 0) { // num_tokens_to_recv表示发送端responsible_rank的responsible_channel要发送到接收端rank的token数量。
            /* NOTES: unlike the sender, the receiver must ensure that the tail indices hold by different warps are the same
               接收端必须确保不同 warp 持有的 tail 索引是相同的，而发送端不需要这样做。
            发送端之所以不用确保不同 warp 持有的 tail 索引是相同的，是因为：
               发送端的不同 warp 持有的 tail 索引本身就是相同的。在发送端，只要 token 需要发送，每个线程都会执行cached_channel_tail_idx ++，
               不同线程的区别只是在if条件判断中有的线程满足这个条件而有的线程不满足，但是每个线程都执行了cached_channel_tail_idx ++。
               而发送端最后写入cached_channel_tail_idx时，实际上所有的发送端的线程的cached_channel_tail_idx都是相同的。这是设计保证的，不是偶然。
            */
            while (recv_thread_id_in_rank == 0) {  // 当前线程在它要处理的发送端rank（responsible_rank）中的局部线程id
                /* 当前接收端记录的rank responsible_rank的channel responsible_channel在接收端的环形缓冲区的tail索引。
                读取 tail（使用 acquire，保证在此加载之后的节点内所有线程的内存操作不会被重排到此加载之前）。
                当前线程用ld_acquire_sys_global读取tail值之后，说明接收端缓冲区必定已经接收到了tail之前的token数据了，并且这部分数据已经在整个sys作用域内可见了。
                                   使得这里可以安全地先读tail，再读channel_x_buffers。
                接下来复制数据时，使用的 ld_nc_global 是使用非一致性缓存读取数据（这样更快），而非一致性缓存可能读取到过期数据（其他 GPU 的写入尚不可见）。
                这里实际上是通过同步机制保证正确性，而不是依赖缓存一致性。
                */
                cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());

                // Ready to copy.  判断是否有新数据
                if (cached_channel_head_idx != cached_channel_tail_idx) {  // 环形缓冲区的head != tail，说明有新数据。而相等就是没有数据。
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    break;  // 有数据了，退出等待循环
                }

                // Timeout check
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = %d, tokens remained: %d\n",
                           rank,
                           responsible_channel,
                           num_tokens_to_recv);
                    trap();
                }
            }

            // Synchronize queue tail
            /* 使用Barrier同步，确保所有负责同一个发送端rank responsible_rank的线程都知道了线程recv_thread_id_in_rank == 0的线程更新好了shared_channel_tail_idx[responsible_rank]。
            知道了这个以后，所有负责同一个发送端rank responsible_rank的线程就可以一起来消费来自rank responsible_rank的channel responsible_channel发送到缓冲区中的token数据了。
            不得不说，bar.sync这个汇编指令真是好用。专门针对同一个SM中的业务逻辑分组后的线程进行同步，而且一个SM中同时最多可以有16个这样的barrier，也就是最多分为16个组。
            为什么要用shared_channel_tail_idx？
            回答：如果只需要warp中的第一个线程拿到某个数据，然后将这个数据传给这个warp中的所有线程，那么就使用__shfl_sync();
                 这是因为__shfl_sync()同时具有创建同步点和交换数据的功能。
                 类似地，当需要在block的所有线程都同步，并且交换数据时，就需要使用__syncthreads()和共享内存，也是为了创建同步点和交换数据。
                 现在，需要在block的分组线程中组内同步并且交换数据，该怎么办？就是要同时解决对这些分组线程的同步和共享数据。
                 关于同步，"bar.sync %0, %1;" 这个同步指令很好用。关于交换数据，毕竟这些线程都是属于当前的block的，所以使用共享内存也是可以满足的，
                 这就是使用共享内存shared_channel_tail_idx的原因。因为当前block中所有的线程会分为kNumRanks个组，同一个组内的线程需要有且仅有一个tail索引，所以共享内存的长度就是组数kNumRanks。
                 如果同一个block中的线程分为 m 组，每组需要记录 n 个数据，那么共享内存的大小就是 m * n。
            */
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            /* strided loop（步进循环） 模式，用于在多个 warp 之间分配工作。
            循环步进分配任务时，是将数据展开，然后将所有的线程去配对相应数量的数据。不要每次去把线程或者warp展开去想，那样就理解偏了。
            理解的时候，以warp为基本单位，先不要以线程为基本单位。在循环体里面使用land_id区分线程。
            这个for循环看起来是以warp为基本单位复制数据，实际上每个线程都要执行每次循环，这样就使得同一个warp的线程都执行了相同的指令，实现SIMT。
            假设要复制10个toekn，有3个warp，那么在warp并行的时候:
              for循环的第一轮循环中：warp 0 处理 chunk_idx 0，warp 1 处理 chunk_idx 1，warp 2 处理 chunk_idx 2
              for循环的第二轮循环中：warp 0 处理 chunk_idx 3，warp 1 处理 chunk_idx 4，warp 2 处理 chunk_idx 5
              for循环的第三轮循环中：warp 0 处理 chunk_idx 6，warp 1 处理 chunk_idx 7，warp 2 处理 chunk_idx 8
              for循环的第四轮循环中：warp 0 处理 chunk_idx 9，warp 1 和 warp 2 不满足“chunk_idx < num_recv_tokens”，不用处理。
            */
            for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) {
                // num_recv_buffer_tokens 含义：环形缓冲区的总容量（token数量）。
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;

                // 源地址，NVLink buffe中的地址，需要通过 head_idx 和 tail_idx 同步
                auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;

                // 目标地址，输出tensor中的地址，普通的GPU显存，最终返回给 expert 模型使用，通过 start_offset 和 end_offset 确定写入位置。
                auto shifted_recv_x_int4 = recv_x + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
#ifndef DISABLE_SM90_FEATURES
                /*
                循环：将数据分成两半处理（half_hidden_int4）。
                注意: 为什么要分两部分，这里的tma_store_wait<0>()才给出了答案: 数据复制流水线优化。
                1、全局内存到共享内存的复制，使用 mbarrier 实现同步。这里的同步，必须等到全局内存的数据完全复制到共享内存完毕，因为下面的共享内存到全局内存的复制
                   必须依赖共享内存中的这部分数据完全可见。此时发起这个TMA任务的线程对共享内存中的这部分数据可见，于是这个线程就可以发起从共享内存到另一个全局内存的复制。
                2、共享内存到全局内存的复制，使用"cp.async.bulk.commit_group;"和"cp.async.bulk.wait_group.read %0;"实现同步。
                    而与上一种情况中的全局内存到共享内存的复制不同，tma_store_wait<0>()只需要等到当前线程把共享内存中的数据都读取完毕就可以了，不用等待数据完全复制到全局内存中，
                    “数据复制到全局内存”这一过程还在特殊硬件TMA中进行，不消耗SM的资源。而且接下来要开启另一半的token数据从全局内存复制到共享内存tma_buffer中，
                    共享内存tma_buffer是两部分token数据共用的，当TMA已经完成对前一半token数据在共享内存中的读取之后，第二部分就可以开始写入到这个共享内存tma_buffer中了。
                */
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    /* 
                    注意: tma_store_wait<0>() 用了.read修饰符，只需要等到当前线程的bulk_group中的tma_store_1d必须完成对共享内存的读取即可，
                          不用等待把共享内存中的数据完全都复制到全局内存之后才等待结束。
                    注意: 当前warp的所有线程都要等待之前的 bulk_group 中的tma_store_1d 完成，因为tma_buffer是每个warp独立拥有的TMA缓冲区。
                          如果接下来warp中的另一个线程选中了要往共享内存tma_buffer中写入数据，那么得保证“上一个tma_store_1d的任务中TMA已经完全
                          读取了共享内存tma_buffer的这部分数据”这一信息对新选择到的线程是可见的。
                    */
                    tma_store_wait<0>();
                    // if (elect_one_sync())：仅 lane 0 执行，避免 warp 内重复操作。
                    if (elect_one_sync()) {  // tma_mbarrier 也是一个CTA中只有一个线程初始化的，如果一个CTA中多个线程同时初始化会导致未定义行为。
                        /*
                        使用 TMA 从 global memory 异步加载到 shared memory。
                        实现：cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes，硬件加速的批量异步传输。
                        参数：
                            tma_buffer：目标 shared memory 地址
                            shifted_buffer_x_int4 + i * half_hidden_int4：源 global memory 地址（NVLink 缓冲区）
                            tma_mbarrier：内存屏障，用于同步
                            half_hidden_bytes：传输字节数
                        */
                        tma_load_1d(tma_buffer, shifted_buffer_x_int4 + i * half_hidden_int4, tma_mbarrier, half_hidden_bytes);
                        /*
                        通知内存屏障：期望接收 half_hidden_bytes 字节。
                        实现：mbarrier.arrive.expect_tx，设置屏障期望值（Transaction_Count），用于等待传输完成。
                        这个调用使mbarrier能够检测传输是否完成。这个函数本身不累加事务字节计数器，只是设置事务字节检测条件。
                        half_hidden_bytes: 当前线程要传输的字节数。不是多线程都对tma_mbarrier对应的共享内存区域传输的字节数之和。
                            当当前线程传输了 half_hidden_bytes 字节的数据之后，mbarrier_wait才会认为当前线程对于tma_mbarrier的屏障完成TMA写共享内存任务了。
                            mbarrier的期待事务计数器会**增加**num_bytes。
                            只有所有参与使用TMA写入tma_mbarrier对应的共享内存的线程都达到了各自在mbarrier_arrive_and_expect_tx中设置的half_hidden_bytes，这些线程才能都通过mbarrier_wait。
                        使用同一个tma_mbarrier的多个线程对同一共享内存区域使用TMA进行覆盖写（不同线程写入同一地址）的情况：
                            如果发生覆盖写（不同线程写入同一地址），expect_tx的累加仍然有效。
                            关键是各自传输的字节数是否达到各自在mbarrier_arrive_and_expect_tx中设置的half_hidden_bytes。
                        */
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, half_hidden_bytes);
                        /*
                        等待内存屏障：阻塞当前线程，直到TMA传输完成。所谓完成，就是满足“那两个条件”。
                        实现：mbarrier.try_wait.parity，轮询等待，完成后更新 tma_phase。
                        */
                        mbarrier_wait(tma_mbarrier, tma_phase);
                        /*
                        使用 TMA 从 shared memory 异步存储到 global memory。 每次都commit_group。
                        实现：cp.async.bulk.global.shared::cta.bulk_group，硬件加速的批量异步存储。
                        参数：
                        tma_buffer：源 shared memory 地址
                        shifted_recv_x_int4 + i * half_hidden_int4：目标 global memory 地址（输出 tensor）
                        half_hidden_bytes：传输字节数
                        false：不使用 evict-first 缓存策略
                        */
                        tma_store_1d(tma_buffer, shifted_recv_x_int4 + i * half_hidden_int4, half_hidden_bytes, false);
                    }
                }
                // 同步 warp：确保所有线程看到 TMA 操作完成。
                // 作用：保证数据可见性，lane 0 的 TMA 操作对 warp 内其他线程可见。
                __syncwarp();
#else
                /* ld_nc_global是使用非一致性缓存读取数据（这样更快），而“非一致性缓存可能读取到过期数据”这一弱点也被
                “通过 head_idx 和 tail_idx 指针同步判断是否有数据（使用 acquire/release 语义保证可见性）”
                这一机制解决了。
                因为读写head_idx 和 tail_idx 指针的频次和数据量毕竟比读写token数据要少得多，所以不会影响性能。
                而将读写token数据改为使用非一致性缓存读取数据（这样更快），则对于提升整体性能来说具有重大意义。
                关于UNROLLED_WARP_COPY的解释：参见：https://osvicyu5w5.feishu.cn/wiki/Wv2bwLGDmipgapkhOdrcZCtynrg#share-QMc4d5u2xo82cxxT8VacmkTuntg
                */
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4, shifted_buffer_x_int4, ld_nc_global, st_na_global);
#endif
            }

            // Copy `src_idx`
            /*
            上面复制数据的数据量大，所以那样优化，这里复制src_idx的数据量小，所以直接使用ld_nc_global读取数据。
            chunk_idx += 32 * num_recv_warps_per_rank 这里表示的循环步进很清晰。
            循环语句内每次并行执行一次，都是执行了当前blcok中的所有线程（数量为 blockDim.x），之所以这里写为 32 * num_recv_warps_per_rank，是为了健壮性。
            chunk_idx 是缓冲区中当前接受到的需要处理的token的索引。
            recv_src_idx：记录recv_x中每个token在这个token对应的发送端rank发送的tensor中的src_idx。recv_src_idx中会记录来自多个rank的tensor中的token。
            同一个发送端rank发送的所有token都来自同一个输入tensor。
            而recv_src_idx是通过total_offset来划分不同rank的不同channel的token是属于哪个rank的，所以recv_src_idx是知道每个元素值是由哪个发送端rank发送的，
            也知道每个token是在这个发送端发送的tensor中的哪个位置。

            为什么要记录 recv_src_idx？
            1、用于反向传播（backward pass）时的 combine 操作。
                反向传播流程：
                Forward (dispatch)：
                    发送端将 token 分发到不同的接收端 rank。
                    记录每个 token 在发送端原始 tensor 中的索引 → recv_src_idx。
                Backward (combine):
                    接收端收到梯度后，需要将梯度发送回对应的发送端 rank。
                    需要知道每个梯度对应发送端原始 tensor 的哪个位置 → 使用 recv_src_idx
            2、在 combine 中的使用：
            从 combine 函数签名combine(..., const int* src_idx, ...)可以看到：
            src_idx 就是 recv_src_idx，用于：
                确定梯度发送位置：知道每个接收到的 token 对应发送端原始 tensor 的哪个位置
                梯度聚合：同一个发送端 token 可能被多个接收端处理（如果该 token 选择了多个 experts），需要将梯度聚合回正确的位置
            */
            #pragma unroll 4
            for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx;
                 chunk_idx += 32 * num_recv_warps_per_rank)
                recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] =
                    ld_nc_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);

            // Copy `topk_idx` and `topk_weights`
            // 写 for 循环时，要在for(XXX）里面写的能遍历需要处理的数据量，然后再在循环体里面计算这次循环需要的各种变量
            #pragma unroll 4
            for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk; idx += 32 * num_recv_warps_per_rank) {
                // token_topk_idx 表示具体是topK中的第几个expert
                int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk; // 这是按照 aaabbbcccddd 这样想象排列的token数据（这是topK=3, tokenCount=4的情况）
                /* token_idx_in_buffer 表示在环形缓冲区中的实际索引。
                然而，channel_topk_idx_buffers中对应的位置与channel_src_idx_buffers、channel_x_buffers和channel_x_scales_buffers中对应的位置的含义一致。
                因为它们都属于缓存，环形缓冲区中的数据都是按照一定的顺序排列的，所以它们在环形缓冲区中的索引是相同的。
                所以这里可以用cached_channel_head_idx来计算出token_idx_in_buffer，然后计算出buffer_idx，然后用于找channel_topk_idx_buffers中的值。
                */
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                /* recv_idx表示来自发送端rank responsible_rank的channel responsible_channel发送的第chunk_idx个token在当前接收端被这个接收端的局部第几个专家激活。
                如果有激活，就是局部expert id，否则就是-1表示这个位置没有被当前接收端rank中的所有专家中的任意一个激活。
                */
                auto recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx; // 也是按照 aaabbbcccddd
                auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
                /*
                recv_topk_idx 的shape是 [num_recv_allrank_tokens, num_topk]
                num_recv_allrank_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank]，即所有的rank发送到当前接收端的token数量之和，在 notify_dispatch 中计算得到。
                rank_prefix_matrix[i * kNumRanks + j]表示的就是从rank 0到rank i发送到rank j的token数量之和。
                num_recv_allrank_tokens有三种情况：
                    num_recv_allrank_tokens = cached_num_recv_tokens;               // 模式1：Cached模式。使用缓存的已知值
                    num_recv_allrank_tokens = num_worst_tokens;                     // 模式2：Worst case模式。使用最坏情况的值（预分配）
                    num_recv_allrank_tokens = static_cast<int>(*moe_recv_counter);  // 模式3：正常模式。从GPU计算的结果读取
                */
                recv_topk_idx[recv_idx] = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
                recv_topk_weights[recv_idx] = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
            }

            // Copy `x_scales`
            #pragma unroll 4
            for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales; i += 32 * num_recv_warps_per_rank) {
                int chunk_idx = i / num_scales, scales_idx = i % num_scales;
                int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
                recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scales_idx] =
                    ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_buffer * num_scales + scales_idx);
            }

            // Move queue
            // 发送端从tail写数据，也由发送端更新tail。接收端从head读数据，也由接收端更新head。
            cached_channel_head_idx += num_recv_tokens;
            total_offset += num_recv_tokens;
            /* 使用bar.sync同步，让所有参与此次针对发送端rank responsible_rank的通信的线程都处理完了各自要处理的：
            recv_x、recv_src_idx、recv_topk_idx、recv_topk_weights和recv_x_scales这些数据。
            因为只有等这些线程都处理完了，接下来使用这些线程中的一个线程去更新channel_head_idx，这样就保证了channel_head_idx的更新是正确的，也是合乎代码设计思路的。
            */
            asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
            if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and elect_one_sync())
                st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

            // Exit
            /* num_tokens_to_recv表示发送端responsible_rank的responsible_channel要发送到接收端rank的token数量。
            这里表示经过这次通信，发送端responsible_rank的responsible_channel要发送到接收端rank的token数量减少了num_recv_tokens。
            */
            num_tokens_to_recv -= num_recv_tokens;
        }
    }

    // Clean unused `recv_topk_idx` as -1
    // 如果使用模式2：Worst case模式。使用最坏情况的值（预分配），那么recv_topk_idx 的shape是 [num_worst_tokens, num_topk]
    if (num_worst_tokens > 0) {
        auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
        // num_recv_tokens 表示所有rank发送给当前rank的token数量之和。
        const auto num_recv_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
        // num_recv_tokens * num_topk表示对于当前rank真正有意义的token要发送到的topK个expert的id
        // sm_id * kNumThreads 表示当前线程所在的block的第一个线程在整个grid中的全局线程id
        const auto clean_start = num_recv_tokens * num_topk + sm_id * kNumThreads;
        const auto clean_end = num_worst_tokens * num_topk;
        const auto clean_stride = num_sms * kNumThreads;
        // grid循环步进，遍历所有需要清理的token。循环体内每执行一次，就是整个grid的线程执行了一次。
        // 第二次执行时，就是整个grid的线程执行后续的clean_stride个写recv_topk_idx[i]数据的任务。
        #pragma unroll
        for (int i = clean_start + thread_id; i < clean_end; i += clean_stride){
            recv_topk_idx[i] = -1;
        }
    }
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              int* recv_src_idx,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              int* recv_channel_offset,
              int* send_head,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              const bool* is_token_in_rank,
              const int* channel_prefix_matrix,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_topk,
              int num_experts,
              int num_scales,
              int scale_token_stride,
              int scale_hidden_stride,
              void** buffer_ptrs,
              int rank,
              int num_ranks,
              cudaStream_t stream,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 8192;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

    // Make sure never OOB
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks)                                      \
    {                                                                    \
        auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                               \
        LAUNCH_KERNEL(&cfg,                                              \
                      kernel,                                            \
                      reinterpret_cast<int4*>(recv_x),                   \
                      recv_x_scales,                                     \
                      recv_src_idx,                                      \
                      recv_topk_idx,                                     \
                      recv_topk_weights,                                 \
                      recv_channel_offset,                               \
                      send_head,                                         \
                      reinterpret_cast<const int4*>(x),                  \
                      x_scales,                                          \
                      topk_idx,                                          \
                      topk_weights,                                      \
                      is_token_in_rank,                                  \
                      channel_prefix_matrix,                             \
                      num_tokens,                                        \
                      num_worst_tokens,                                  \
                      hidden_int4,                                       \
                      num_topk,                                          \
                      num_experts,                                       \
                      num_scales,                                        \
                      scale_token_stride,                                \
                      scale_hidden_stride,                               \
                      buffer_ptrs,                                       \
                      rank,                                              \
                      num_max_send_tokens,                               \
                      num_recv_buffer_tokens);                           \
    }                                                                    \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(num_sms % 2 == 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

/*
gridDim = 1 + num_channels、blockDim = max(128, 32 * num_ranks)
这样 block0 负责全局清零与 cross-rank barrier，其余 block 逐个 channel 进行 send_head 预处理
*/
template <int kNumRanks>
__global__ void cached_notify_combine(
    void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens, int num_memset_int, int** barrier_signal_ptrs, int rank) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    // 部分1：SM 0 负责清零
    if (sm_id == 0) {
        // Barrier before cleaning
        barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

        // Clean
        auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
        auto ptr = static_cast<int*>(buffer_ptrs[rank]);
        #pragma unroll
        /* 
        在 combine 执行之前，rank A 会通过 cached_notify_combine 清零自身在 dispatch 阶段作为接收端的环形缓冲区的元数据（head 和 tail 指针），
        从而释放缓冲区供 combine 阶段使用。这是通过重置指针实现的，而不是物理删除数据。

        num_memset_int = num_channels * num_ranks * 4
        四倍系数对应要清零的四块元数据（channel_start_offset、channel_end_offset、channel_head_idx、channel_tail_idx）
        清零了channel_head_idx、channel_tail_idx，那么channel_x_buffers、channel_src_idx_buffers、channel_topk_idx_buffers、
        channel_topk_weights_buffers、channel_x_scales_buffers里面的数据就实际上没有意义了，因为无法通过正确的索引找到里面的数据，
        所以就相当于清空了，就都可以被combine阶段使用了。
        */
        for (int i = thread_id; i < num_memset_int; i += num_threads)  // block循环步进
            ptr[i] = 0;

        // Barrier after cleaning
        barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
    } else {   // 部分2：其他 SM 处理 send_head
        const auto channel_id = sm_id - 1;
        const auto thread_id = static_cast<int>(threadIdx.x);
        const auto rank_id = thread_id / 32;   // 当前block的warp id，每个warp对应一个rank
        const auto lane_id = thread_id % 32;
        if (rank_id >= kNumRanks)  // 当前block只需要kNumRanks个warp
            return;

        int token_start_idx, token_end_idx;
        /* num_recv_tokens多个token平均分给num_channels个channel处理，
        第channel_id个channel要处理的token范围是 [token_start_idx, token_end_idx)
        */
        get_channel_task_range(num_recv_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

        // NOTES: `1 << 25` is a heuristic large number.  处理 send_head，对于 -1 的值设置特殊值
        int last_head = 1 << 25;   // 33554432  last_head 记录“最近一次出现的正 head 值”
        /*
        之所以使用倒序排列token_idx_tail，是因为这样在访问send_head中的token位置索引时，是从大的位置索引逐渐读到小的位置索引，
        而根据token_idx = token_idx_tail - lane_id可知，lane_id越大，token_idx越小，
        在内层for循环使用token_idx正序排列时，就是lane_id越来越大，所以token_idx越来越小，所以越来越能找到同一个warp中head最小的那个head。

        所有 warp 处理相同 token 范围但对应不同 rank，通过 warp_id（rank_id）区分访问send_head的列。
        */
        #pragma unroll
        for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= 32) {
            int token_idx = token_idx_tail - lane_id, expected_head = 0;
            // current_head 在下面理解为：在send_head中连续32个token对应rank rank_id的head_idx
            /*
            LyricZhao 说：
            对于普通核函数（normal kernels），需要在合并前（before combine）手动乘以 topk_weights。由于普通核函数通常处理较小的 EP 规模（如 16/32/64），一个 token 可能会在单个 GPU 中选择多个专家（expert），因此需要在合并前执行一个归约核函数（融合了 topk_weights 乘法操作），以节省带宽。
            而对于低延迟核函数（low-latency kernels），单个 GPU 中的一个令牌通常不会选择多个专家，因此无需归约操作，直接在合并阶段使用 topk_weights 即可。

            send_head[token_idx * kNumRanks + rank_id]表示：token token_idx发送到rank rank_id的环形缓冲区的单调递增索引。

            在dispatch的发送端写入send_head是从tail写入，而这里读取send_head将读取到的值命名为current_head，即head，
            是因为send_head是将要用于combine中读取（消费）环形缓冲区中的token的索引，是从head消费的，所以使用head命名

            send_head是行优先存储，Z字形（列优先是倒N字形），连续两个线程要跨越kNumRanks个元素，但是kNumRanks ≤ 8，
            所以跨越的元素数量不多，但是32个线程无法在同一个warp调度器中一次执行完对内存的访问，所以需要分多次访问。再加上 ldg 走只读 cache，所以可接受。

            **current_head是send_head的同一列的连续32行的缓冲区单调递增索引**
            */
            auto current_head = (token_idx >= token_start_idx) ? __ldg(send_head + token_idx * kNumRanks + rank_id) : -1;
            for (int i = 0; i < min(32, token_idx_tail - token_start_idx + 1); ++i) {
                const int head = __shfl_sync(0xffffffff, current_head, i);  // 每次广播这32个token中第i个token的head_idx到这32个线程的current_head中
                // 如果一直不满足if (head < 0)，那么current_head就肯定大于0，所以就不会执行下面的给send_head[token_idx * kNumRanks + rank_id]赋值
               
                // **为什么连续32个token索引都是-1的话，expected_head就是-(1 << 25)-1？**
               
                if (head < 0) {  // 就是-1，-1表示不发送token token_idx到rank rank_id
                    /*
                    若读到的是负数（意味着 dispatch 未发到该 rank，原值是 -1），则由 lane i 把 expected_head 设成 -last_head - 1。
                    这是为了给这些 -1 赋一个带含义的负数，需要知道“最近一次出现的正 head 是多少”。
                    */
                    if (lane_id == i)
                        expected_head = -last_head - 1;
                } else {
                    // 若读到的是非负数（说明 dispatch 已写过真实 tail），就更新 last_head = head。
                    last_head = head;
                }
            }
            // combine 要用send_head查询发往rank rank_id 的 token 在环形缓冲区中的 slot，并等待对应 tail_idx 就绪。
            /*
            循环结束后，若某个 token 的 current_head 为负并且属于本块（token_idx >= token_start_idx），就把 expected_head 写回 send_head。
            这样 combine 看到负值就知道“不需要等待新写入（不会有数据），直接当作尚无负载处理”，同时余下正值继续维持对缓冲区真实 slot 的索引。
            */
            if (current_head < 0 and token_idx >= token_start_idx)
                send_head[token_idx * kNumRanks + rank_id] = expected_head;   // 设置特殊值，供 combine 阶段判断使用
        }
    }
}

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** barrier_signal_ptrs,
                           int rank,
                           int num_ranks,
                           cudaStream_t stream) {
#define CACHED_NOTIFY_COMBINE(ranks)            \
    LAUNCH_KERNEL(&cfg,                         \
                  cached_notify_combine<ranks>, \
                  buffer_ptrs,                  \
                  send_head,                    \
                  num_channels,                 \
                  num_recv_tokens,              \
                  num_memset_int,               \
                  barrier_signal_ptrs,          \
                  rank);                        \
    break

    const int num_threads = std::max(128, 32 * num_ranks);
    EP_HOST_ASSERT(num_ranks <= num_threads);
    EP_HOST_ASSERT(num_threads <= 1024);
    EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);
    SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
    SWITCH_RANKS(CACHED_NOTIFY_COMBINE);
#undef CACHED_NOTIFY_COMBINE
}

template <typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(kNumThreads, 1) combine(dtype_t* recv_x,
                                                          float* recv_topk_weights,
                                                          const dtype_t* x,
                                                          const float* topk_weights,
                                                          const dtype_t* bias_0,
                                                          const dtype_t* bias_1,
                                                          const int* src_idx,
                                                          const int* rank_prefix_matrix,
                                                          const int* channel_prefix_matrix,
                                                          int* send_head,
                                                          int num_tokens,
                                                          int num_recv_tokens,
                                                          int hidden,
                                                          int num_topk,
                                                          void** buffer_ptrs,
                                                          int rank,
                                                          int num_max_send_tokens,
                                                          int num_recv_buffer_tokens) {
    /*
    recv_x:                 [num_recv_tokens, hidden],  要写入的, 接收端接收到的token组成的tensor。
    recv_topk_weights:      [num_recv_tokens, num_topk],要写入的, recv_topk_weights[token_idx * num_topk + lane_id] 记录的token token_idx的全局第lane_id个topk的权重，这是按照全局专家id来记录的，而不是按照局部专家id来记录的。记录的权重值是float类型的。   
    x:                      [num_tokens, hidden],       要读取的, 当前rank上所有专家模型输出的token hidden数据。按dispatch发送端的rank id排序，同一个rank内按channel排序。
    topk_weights:           [num_tokens, num_topk],     要读取的, 当前rank作为combine发送端（也就是dispatch接收端），要发送给其他rank的token的topk权重。
    bias_0:                 [num_recv_tokens, hidden],  要读取的, 接收端数据聚合时的偏差补偿项。
    bias_1:                 [num_recv_tokens, hidden],  要读取的, 接收端数据聚合时的偏差补偿项。
    src_idx:                num_recv_tokens,            要读取的, __ldg(src_idx + token_idx + i)：当前combine发送端要发送的第(token_idx + i)个token对应到的dispatch发送端原始输入tensor中的索引，这个索引可以用于combine接收端去取值。
    rank_prefix_matrix:     [num_ranks, num_ranks],     要读取的, rank_prefix_matrix[i * num_ranks + j]表示的是dispatch阶段从rank 0到rank i累计发送到rank j的token数量之和（rank级别前缀和）。
    channel_prefix_matrix:  [kNumRanks, num_channels],  要读取的, channel_prefix_matrix[dst_rank * num_channels + i]表示的是dispatch发送端rank rank从channel 0到channel i（包含channel i）累计要发送到rank dst_rank的token数量之和（channel级别前缀和）。
    
    send_head:              [num_tokens, kNumRanks],    要读取的, send_head[token_idx * kNumRanks + responsible_rank]: 表示dispatch阶段token token_idx发送到rank responsible_rank的环形缓冲区的单调递增索引。
        虽然send_data使用的是dispatch阶段的缓冲区中的索引，但是dispatch阶段的缓冲区中的token数据已经被清空了，
        但是send_data记录的这个token在缓冲区中的单调递增索引确是依然会对应到combine的发送端中的token数据的具体索引的，
        因为combine的发送端的发送token的顺序和索引位置和dispatch阶段的接收端的缓冲区中的索引位置是一致的。
        我认为这是DeepEP能实现如此高效MoE通信的关键，因为这一点解决了使用缓冲区的一个难点，
        就是使用缓冲区就意味着“一批批地传输数据”，而在这“一批批地传输数据”的过程中，需要依然保持combine发送端和dispatch接收端的token的全局对应关系。
    num_tokens:             num_tokens,                 要读取的  表示当前rank要发送的token数量  
    num_recv_tokens:        num_recv_tokens,            要读取的  表示当前rank要接收的token数量  
    hidden:                 hidden,                     要读取的  表示每个token的维度  
    num_topk:               num_topk,                   要读取的  表示每个token的激活专家数量  
    buffer_ptrs:            void** buffer_ptrs,         要读写的  表示当前rank的buffer_ptrs指针  
    rank:                   rank,                       要读取的  表示当前rank的rank ID  
    num_max_send_tokens:    num_max_send_tokens,        要读取的  表示当前rank每次单次发送的最大token数量（chunk大小），用于限制分批发送token数据。
    num_recv_buffer_tokens: num_recv_buffer_tokens,     要读取的  表示当前rank针对每个发送端rank的每个channel的环形缓冲区的总token容量。
    
    */
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto sm_id = static_cast<int>(blockIdx.x), lane_id = get_lane_id();
    const auto num_channels = num_sms / 2;
    const bool is_sender = sm_id % 2 == 0;
    const int responsible_channel = sm_id / 2;
    EP_DEVICE_ASSERT(num_topk <= 32);

    constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
    int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
    int hidden_int4_aligned = align_down(hidden_int4, 32);
    /*
    x 和 topk_weights 都是combine的输入参数，在发送端被读取。
    x：shape:[num_tokens, hidden]，expert的输出
    topk_weights：shape:[num_tokens, num_topk]，对应x的topk权重，表示每个token的topk权重
    num_tokens是发送端rank B上expert处理的token数量，即来自dispatch阶段的接收端rank B接收到的token数量

    LyricZhao说：对于普通核函数（normal kernels），需要在合并前（before combine）手动乘以 topk_weights。
                由于普通核函数通常处理较小的 EP 规模（如 16/32/64），一个token可能会在单个 GPU 中选择多个专家（expert），
                因此需要在合并前执行一个归约核函数（融合了 topk_weights 乘法操作），以节省带宽。
                而ll模式不是这样。
    */
    auto x_int4 = reinterpret_cast<const int4*>(x);
    /*
    bias_0 和 bias_1（可选输入参数）
    shape: [num_recv_tokens, hidden]
    num_recv_tokens是接收端要接收的token数量（原本dispatch的发送端从Multihead Attention传入的token数量）
    不是当前 rank 的，而是接收端（原本 dispatch 的发送端 rank）的 bias。
    作用：bias_0和bias_1刚好匹配接收端 token 的隐藏层维度。它们本质是数据聚合时的偏差补偿项，用于修正分布式传输中可能出现的数值偏移（如低精度传输误差、GPU 间计算精度差异等），
    这与论文中强调的 “FP8 混合精度训练”“低损耗通信” 等优化目标高度契合。
    */
    auto bias_0_int4 = reinterpret_cast<const int4*>(bias_0);
    auto bias_1_int4 = reinterpret_cast<const int4*>(bias_1);

    /*
    recv_x 和 recv_topk_weights 都是combine的输出参数，在接收端被写入，都是供给下一层Multihead Attention使用的。
    recv_x：shape: [num_recv_tokens, hidden]，聚合后的结果。接收端（原本 dispatch 的发送端 rank）接收并聚合来自多个 expert rank 的输出
    recv_topk_weights：shape: [num_recv_tokens, num_topk]，聚合后的 topk 权重
    */
    auto recv_int4 = reinterpret_cast<int4*>(recv_x);

    // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    // 也是每个warp又都有自己的TMA 共享内存缓冲区 
    auto tma_buffer = smem_buffer + (thread_id / 32) * kNumTMABytesPerWarp;
#endif

    /*
    在MoE架构中，典型流程是：Input → Router → Dispatch → Expert Processing → Combine → Next Layer (MHA/FFN)

    dispatch阶段：发送端rank A → 接收端rank B: 发送token到expert rank

    expert处理：rank B: 各个expert处理接收到的token，得到输出 x [num_tokens, hidden]

    combine阶段：
        发送端：即rank B。将x和topk_weights发送回rank A。
            x，shape:[num_tokens, hidden]，expert的输出
            topk_weights，shape:[num_tokens, num_topk]，对应的topk权重，与x对应，表示每个token的topk权重
            num_tokens是当前rank B上expert处理的token数量，即来自dispatch阶段的接收端rank B收到的token数量
        接收端：即rank A。接收来自多个rank的expert输出，聚合得到recv_x，shape:[num_recv_tokens, hidden]，
            recv_x供给下一层Multihead Attention使用。
            num_recv_tokens是接收端要接收的token数量（原本dispatch的发送端从Multihead Attention传入的token数量）
    */
    if (is_sender) {   // combine发送端负责把参数中的经过专家计算后的token数据发动到dispatch阶段的发送端（现在是combine接收端）的环形缓冲区中。
        // Workers for sending
        // Several warps are responsible for a single rank 多个warp负责向一个rank发送token数据
        constexpr int num_send_warps_per_rank = (kNumThreads / 32) / kNumRanks; // 当前channel负责发送给一个rank使用的warp数量
        constexpr int num_send_warps = num_send_warps_per_rank * kNumRanks; // 当前channel负责发送token数据的warp数量
        const auto num_threads_per_rank = num_send_warps_per_rank * 32; // 当前channel负责发送给一个rank使用的线程数量
        const auto send_thread_id = thread_id;
        const auto send_warp_id = send_thread_id / 32; // 当前线程所在的warp的warp id
        /*
        responsible_channel + send_warp_id 的目的： 将不同 channel 的 warp 轮询分配给各个 rank，实现负载均衡。
        加上 responsible_channel 是为了让不同 channel 的分配产生偏移，避免所有 channel 的 warp 0 都分配给同一个 rank，从而均衡负载。
        send_rank_id表示的是dispatch阶段的发送端rank的id，即当前combine阶段的发送端rank要发送到的接收端rank的id。
        */
        const auto send_rank_id = (responsible_channel + send_warp_id) % kNumRanks;
        // 当前线程所在的warp用于给接收端的rank通信的局部warp id，对于同一个接收端 rank的warp id的范围都是[0, num_send_warps_per_rank - 1]。
        const auto send_warp_id_in_rank = send_warp_id / kNumRanks;
        EP_STATIC_ASSERT(num_send_warps * 32 == kNumThreads, "Invalid warp count");

        // Calculate pointers by the specific layout
        /* 当前warp要发送到的rank的NVLink Buffer的指针
        Combine 中没有：
        rank_prefix_matrix
            因为combine函数直接传入了rank_prefix_matrix，和dispatch阶段的是一样的。
        channel_start_offset 和 channel_end_offset
            dispatch：接收端需要知道每个 channel 在输出 tensor 中的写入范围，用于将接收到的 token 写入正确位置。
            combine：接收端直接聚合所有 rank 的数据，不需要按 channel 划分输出范围。
        channel_topk_idx_buffers:
            dispatch：需要保存每个 token 的 topk expert 索引，用于后续 expert 路由。
            combine：expert 已处理完成，只需聚合结果和权重，不需要 expert 索引。
        channel_x_scales_buffers
            dispatch：需要保存量化/缩放因子，用于低精度传输。
            combine：combine 阶段可能不需要单独的 scales buffer，或使用不同的量化策略。
        */
        auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[send_rank_id]));
        auto num_channels_total = num_channels * kNumRanks;
        auto channel_rank_offset = responsible_channel * kNumRanks + rank;

        // Channel meta data
        // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
        auto channel_x_buffers = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
        auto channel_src_idx_buffers = Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
        // 这里不像dispatch那样，需要channel_topk_idx_buffers，因为combine阶段不需要topk_idx，只需要topk_weights。
        auto channel_topk_weights_buffers = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);

        // Get tasks
        // NOTES: `channel_offset` is already shifted
        // rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank]
        // 表示从 rank 0 到 rank (send_rank_id - 1) 发送到当前rank的 token 数量之和
        int rank_offset = send_rank_id > 0 ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank] : 0;
        // num_rank_tokens：dispatch阶段rank send_rank_id发送到当前rank的所有token数量之和
        int num_rank_tokens = rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
        /*
        注意：combine阶段使用的channel_prefix_matrix实际上是recv_channel_prefix_matrix

        在buffer.py中：
        - dispatch返回的handle结构（第382行）：
          handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
        - combine阶段解包handle（第422行）：
          rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
        注意：第二个位置（发送端的channel_prefix_matrix）被忽略了（用_），第三个位置才是combine阶段使用的channel_prefix_matrix。
             所以，combine阶段使用的channel_prefix_matrix实际上是recv_channel_prefix_matrix！

        在dispatch阶段的接收端：recv_channel_offset[i, j] = 从rank i接收的数据中，channel j要写入的token在dispatch接收端输出数组recv_x中的给rank i准备的空间的起始索引。

        channel_offset = recv_channel_prefix_matrix[send_rank_id * num_channels + responsible_channel]
        表示：在dispatch阶段从rank send_rank_id接收的数据中，responsible_channel要写入的token在recv_x中的起始索引

        这正是combine 发送端需要的信息，它需要知道recv_x中来自rank send_rank_id的responsible_channel的token的起始位置。

        2025:11:17 23:25:00 关键发现：channel_prefix_matrix在combine阶段是recv_channel_prefix_matrix、recv_channel_offset

            */
        int channel_offset = channel_prefix_matrix[send_rank_id * num_channels + responsible_channel];
        int num_channel_tokens =
            (responsible_channel == num_channels - 1 ? num_rank_tokens
                                                     : channel_prefix_matrix[send_rank_id * num_channels + responsible_channel + 1]) -
            channel_offset;
        int token_start_idx = rank_offset + channel_offset, token_end_idx = rank_offset + channel_offset + num_channel_tokens;

        // Iterate over all tokens and send by chunks
        int current_channel_tail_idx = 0;
        // 把经过expert计算过的token数据，原路返回给dispatch的发送端。当前线程只需要处理对rank send_rank_id的responsible_channel的token数据。
        for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
            // Check destination queue emptiness, or wait a buffer to be released (rare cases)
            auto start_time = clock64();
            /* num_max_send_tokens的含义和传入方式：
            1. 含义：每次发送的最大token数量（chunk大小），用于分批发送token数据
            2. 传入方式：
               - 在deep_ep.cpp第642行，通过config.num_max_nvl_chunked_send_tokens传入
               - 默认值：6（在deep_ep.cpp第1364行）
               - 这是Config结构体中的一个配置参数，用于控制NVLink通信的chunk大小
            3. 用途：
               - 限制每次发送的token数量，避免一次性发送太多数据导致缓冲区溢出
               - 实现分批发送（chunked sending），提高通信效率
               - 与num_recv_buffer_tokens配合使用，确保接收端有足够的缓冲区空间
            4. 使用方式：
               - num_round_tokens = min(num_max_send_tokens, token_end_idx - token_idx)
               - 这确保每次发送的token数量不超过num_max_send_tokens，也不超过剩余的token数量
             */
            int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
            if (elect_one_sync()) {
                while (true) {
                    // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
                    /*
                    接收端写入channel_head_idx，表示接收端已经消费到了的索引位置。接收端使用共享变量warp_channel_head_idx来帮助更新channel_head_idx。
                    num_used_slots表示有num_used_slots多个token数据在缓冲区中，还没有被接收端消费。
                    */
                    int num_used_slots = current_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
                    // num_recv_buffer_tokens - num_used_slots表示接收端环形缓冲区空余的可以存放token数据的slot数量。
                    // 如果空余的slot数量 >= 本次要发送的token数量，则可以发送。
                    if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens)
                        break;

                    // Rare cases to loop again
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
                        trap();
                    }
                }
            }
            __syncwarp();
            /* 除了当前发送端rank的warp要往接收端rank send_rank_id的channel responsible_channel的环形缓冲区发送token，
             没有别的什么东西会发送数据去消耗这个环形环形缓冲区，所以，在这里等到了就是等到了，接下来当前warp往里面写就行，很安全，没人跟你争用缓冲区。
            */

            // Send by chunk
            #pragma unroll
            // 以num_send_warps_per_rank个warp为单位的循环步进
            for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) {
                // Get an empty slot  写入数据是从tail写入，读取数据是从head读取。
                int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;

                // Copy data
                // shifted_x_buffers：目的地址。shifted_x：源地址。ld_nc_global：读取源地址的数据。st_na_global：写入目的地址的数据。
                auto shifted_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                auto shifted_x = x_int4 + (token_idx + i) * hidden_int4;  // 当前warp的第一个线程要发送的token数据在x_int4中的起始地址。
                UNROLLED_WARP_COPY(4, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);

                // Send source index
                if (elect_one_sync()) {
                    /*
                    在deep_ep.cpp第638行，调用combine时传入的是src_idx.data_ptr<int>()
                    src_idx是从handle中解包出来的，对应的是recv_src_idx，recv_src_idx是在dispatch阶段从channel_src_idx_buffers中复制过来的。
                    __ldg(src_idx + token_idx + i)：当前combine发送端要发送的第(token_idx + i)个token对应到的dispatch发送端原始输入tensor x 中的索引，这个索引可以用combine接收端去取值。
                    channel_src_idx_buffers[dst_slot_idx]：接收端rank send_rank_id的channel responsible_channel的环形缓冲区的第dst_slot_idx个slot的token在当前发送端原始输入tensor x 中的源索引。
                    */
                    channel_src_idx_buffers[dst_slot_idx] = __ldg(src_idx + token_idx + i);
                }

                // Send `topk_weights`
                if (num_topk > 0 and lane_id < num_topk)  // 就知道激活专家数不超过32，V3是8个
                    /* 记录channel_topk_weights_buffers，用于给接收端聚合使用。
                    https://osvicyu5w5.feishu.cn/wiki/MWp0wGlFDi6zSJkwSy1cRrXan2X#share-QZYUd4pAbohEOqxuFWncVG5PnR4
                    中的g_{i,t}
                    */
                    channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = __ldg(topk_weights + (token_idx + i) * num_topk + lane_id);
            }
            token_idx += num_round_tokens;
            current_channel_tail_idx += num_round_tokens;  // 又写了num_round_tokens个token数据到环形缓冲区，所以环形缓冲区又满了num_round_tokens个token数据。

            // Move tail index
            // 每个SM有16个硬件barrier（Barrier Units）。
            asm volatile("bar.sync %0, %1;" ::"r"(send_rank_id), "r"(num_threads_per_rank));
            if (send_warp_id_in_rank == 0 and elect_one_sync())
                // 当前发送端rank rank的channel responsible_channel给rank send_rank_id的channel responsible_channel的环形缓冲区写入tail索引，只需一个线程写入即可。
                st_release_sys_global(channel_tail_idx.buffer(), current_channel_tail_idx);
        }
    } else {
        // Workers for receiving
        // One warp for moving the queue head, others for reduction
        constexpr int num_recv_warps = kNumThreads / 32;
        const auto recv_warp_id = thread_id / 32;
        EP_DEVICE_ASSERT(kNumRanks <= 32 and kNumThreads > 32);
        EP_DEVICE_ASSERT(thread_id >= 0 and kNumThreads % 32 == 0);

        // Shared head, tail and retired flags for receiver warps
        /* 
        warp_channel_head_idx 记录每个warp在当前channel responsible_channel的rank lane_id的offset的值。
        每个 warp 可以独立跟踪每个接收 warp 从每个发送 rank 读取到的 head 位置。
        warp 0 可以安全地更新全局 head（取所有 warp 的最小值）。避免过早释放仍被其他 warp 使用的缓冲区空间。
        volatile 表示warp 0 的更新对所有 warp 可见。

        warp_channel_head_idx 存在的唯一目的就是帮助combine接收端更新channel_head_idx_ptr，以便于在发送端通过读取head指针来判断接收端把数据消费到了什么位置。
        由于一个warp每次在读取一个逻辑概念上的token时，这个逻辑概念上的token实际上是要去kNumRanks个combine接收端缓冲区中读取的，因此，需要知道每个warp当前消费到了这kNumRanks个缓冲区的哪个head位置。
        然后找出同一个接收端缓冲区在不同warp中消费到的当前最小的head才是这个接收端缓冲区可以安全更新的head。
        */
        __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
        // 记录每个rank在当前channel responsible_channel的tail索引的值。接收端只用读取tail索引以判断环形缓冲区是否有token数据以及多少token数据，接收端不用写tail索引。
        __shared__ volatile int channel_tail_idx[kNumRanks];
        __shared__ volatile bool warp_retired[num_recv_warps];  // 注意: 退休的是warp（local_rank）
        if (thread_id < num_recv_warps)
            warp_retired[thread_id] = false;  // num_recv_warps个warp都初始化为false
        if (lane_id < kNumRanks)
            warp_channel_head_idx[recv_warp_id][lane_id] = 0;
        if (thread_id < kNumRanks)
            channel_tail_idx[thread_id] = 0;
        /*
        bar.sync  barrier_id, thread_count:
        bar.sync  0, kNumThreads  : 同步当前SM中的所有线程。而不是同步当前combine核函数在整个grid上的所有线程。
        */
        asm volatile("bar.sync 0, %0;" ::"r"(kNumThreads));

        // 这里是用warp 0 来移动环形队列的head，其他warp负责聚合。这些warp都是并行运行的，warp 0 是使用轮询，直到所有warp都retired
        if (thread_id < 32) {
            int* channel_head_idx_ptr = static_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks + lane_id;
            /* channel_head_idx_ptr是找到了对应channel responsible_channel的rank lane_id的offset的位置的指针，
            而把channel_head_idx_ptr往后移动num_channels * kNumRanks个int，就找到了对应channel responsible_channel的rank lane_id的tail索引的指针。
            */
            int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;

            // Queue head updater
            int last_head = 0;
            while (lane_id < kNumRanks) {
                // Check retired
                bool retired = true;
                #pragma unroll
                for (int i = 1; i < num_recv_warps; ++i)
                    // 哪怕只有一个warp没有retired，那么retired就是false
                    retired = retired and warp_retired[i];
                if (retired)  // 只有所有warp都retired了，才能退出循环。说明 combine 的接收端已处理完分配给该 channel 的数据，可以退出。
                    break;

                // Update queue tail
                /*
                channel_tail_idx中的tail索引在dispatch发送端最后写入，写入的是从0到正无穷范围的环形队列索引值，不是对缓冲区长度取余，也不是取负减一。
                发送端会不断更新tail，接收端会在 while (__any_sync(0xffffffff, channel_tail_idx[lane_id] <= expected_head and expected_head >= 0)) 
                中等待该有token token_idx的所有缓冲区都被写入了token token_idx，否则，接收端就会一直等待发送端做到这一点。

                另外，这里和dispatch一样，读写tail用的是“acquire-release”，而读写head用的是“volatile-relaxed”。原因同样参见：https://osvicyu5w5.feishu.cn/wiki/OpnVw2z8gi4DCdkWkpwcZSGinIg
                */
                channel_tail_idx[lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);

                // Update minimum head
                int min_head = std::numeric_limits<int>::max();
                #pragma unroll
                // 相当于高级语言中的32个伴随线程，这里就不用像高级语言那样需要sleep()，因为GPU中的线程是单独的资源，访问的是共享内存，共享内存和L1缓存在物理上是同一块可配置的SRAM池，极高效。
                for (int i = 1; i < num_recv_warps; ++i)
                    if (not warp_retired[i])
                        // warp 0负责找到所有没有retired的warp中rank lane_id的最小head。只有所有warp都读取过的缓冲区位置才能安全释放
                        min_head = min(min_head, warp_channel_head_idx[i][lane_id]);
                if (min_head != std::numeric_limits<int>::max() and min_head > last_head)
                    // 如果最小head大于last_head，则更新last_head。
                    st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head);
            }
        } else {
            // Receivers
            // Channel metadata
            // All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`
            Buffer<int4> channel_x_buffers[kNumRanks];
            Buffer<float> channel_topk_weights_buffers[kNumRanks];

            // Calculate pointers by the specific layout
            #pragma unroll
            for (int i = 0; i < kNumRanks; ++i) {
                auto channel_rank_offset = responsible_channel * kNumRanks + i;
                auto num_channels_total = num_channels * kNumRanks;
                // `head_idx` & `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
                auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[rank]) + 2 * num_channels * kNumRanks * sizeof(int));

                // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
                channel_x_buffers[i] = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);

                // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
                ptr = reinterpret_cast<void*>(static_cast<int8_t*>(ptr) + num_channels_total * num_recv_buffer_tokens * sizeof(int));

                // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
                channel_topk_weights_buffers[i] = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
            }

            // The same tokens as the dispatch process
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

            // Iterate over all tokens and combine
            /* 
            1、相邻的channel处理num_recv_tokens个token中相邻的一部分token，即token token_start_idx到token (token_end_idx-1)
            2、一个channel中的所有warp要处理的token token_start_idx到token (token_end_idx-1)的这些token中，不同的token发往不同的接收端rank。
                但是，不同的warp去把不同的token发往哪个rank呢？
                答：用land_id表示对应的rank id。去找在dispatch阶段当前rank发送到rank lane_id中的第token_idx个token。
                    如果有发送，就取token token_idx在rank lane_id环形缓冲区中的单调递增索引。没发送就是负数。
                每个线程都会去 **遍历** 自己所在channel（即 responsible_channel）在dispatch阶段负责发送的token范围内的所有token，并以当前线程自身的land_id作为dispatch接收端的rank id。
                等到warp中的8个负责rank的线程都等待到了对应rank缓冲区中的数据时，就开始聚合。
            
            token_idx += num_recv_warps - 1。这里减一是因为warp 0用来移动环形队列的head了。
            recv_warp_id - 1。这里减一也是因为warp 0用来移动环形队列的head了。用来聚合的warp的编号范围是[1, num_recv_warps - 1]。
            循环变量每次token_idx += num_recv_warps - 1，是因为每一轮循环都用出了所有的num_recv_warps - 1个warp，处理了num_recv_warps - 1个token。
            下一轮的循环处理下一批量的num_recv_warps - 1个token。即Bob说的 (32 *（num_recv_warps - 1）) 循环步进。
            注意：这里的for循环执行的是一个warp要负责的所有token的聚合，这个循环执行完了，也就是当前的warp的任务执行完了。所以for循环之后就可以 retire 这个warp了。
            */
            for (int64_t token_idx = token_start_idx + recv_warp_id - 1; token_idx < token_end_idx; token_idx += num_recv_warps - 1) {
                // Read expected head
                int expected_head = -1;
                if (lane_id < kNumRanks)  // 32个线程中，最多只有8个线程需要读取send_head。
                    /* send_head记录的是当前接收端rank A作为dispatch发送端的时候，发送各个token到各个rank的环形缓冲区的单调递增索引。
                    send_head[token_idx * kNumRanks + lane_id]表示：token token_idx在dispatch 阶段的发送端rank A（就是当前combine阶段的接收端）上发送到rank lane_id的环形缓冲区的单调递增索引。
                    一个token如果发送到的多个expert在同一个rank中，那么在这个rank的环形缓冲区内也只记录一次，在send_head中只记录一次，在后面的slot_indices和topk_ranks中也只记录一次。
                    
                    为什么使用dispatch发送端的send_head来索引combine接收端的读取的token索引是正确的？
                    答案：位置映射关系保持不变！
                        dispatch阶段: send_head[token_idx] = tail 表示token token_idx存储在位置tail
                        expert计算: token token_idx经过expert处理，得到tensor x，输出仍然存储在tensor x的位置tail
                        combine阶段: 使用send_head[token_idx] = tail读取位置tail，得到的就是token token_idx经过expert的输出
                    */
                    expected_head = ld_nc_global(send_head + token_idx * kNumRanks + lane_id);

                auto start_time = clock64();
                /* expected_head >= 0 表示 rank A 在 dispatch 阶段确实将 token_idx 发送到了 rank lane_id。如果为负数，表示没有发送。
                有数据时：tail > head。无数据时：tail <= head。所以 channel_tail_idx[lane_id] <= expected_head 表示来自 rank lane_id 的数据尚未就绪。
                __any_sync(0xffffffff, condition) 会检查 warp 中所有线程的条件，只要有一个线程的条件为真，就返回真。因此，只要有一个线程需要等待数据，整个 warp 就会继续循环。
                8个线程读取到的expected_head中，可能有些head是负数，表示dispatch阶段没有发送过token token_idx到rank lane_id，比如选择的8个激活专家有多个在同一个rank上。

                等到该有这个token数据的所有rank的缓冲区都有了这个token数据才罢休，否则一直等。
                等的是谁往缓冲区写呢？答：等 UNROLLED_WARP_COPY(4, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);
                */
                while (__any_sync(0xffffffff, channel_tail_idx[lane_id] <= expected_head and expected_head >= 0)) {
                    // Timeout check
                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        printf("DeepEP timeout for combine receivers, rank %d, responsible_channel = %d, expect = %d\n",
                               rank,
                               responsible_channel,
                               expected_head);
                        trap();
                    }
                }
                __syncwarp();
                // 此时，应该有token需要被读取的缓冲区都有数据了。不应该有数据的环形缓冲区就是expected_head < 0的缓冲区。

                // Broadcast current heads  广播当前的heads
                /* 记录token
                slot_indices[kNumRanks]：记录token token_idx在这些rank的环形缓冲区的物理索引。
                topk_ranks[kNumRanks]：记录token token_idx在dispatch阶段分发到哪些rank了。
                
                一个token如果在dispatch阶段发送的多个expert在同一个rank中，那么这个token的多个专家输出在输入到combine时，就已经在python代码中进行了加权求和。
                所以接收端rank的环形缓冲区还是只有一份token数据。
                在send_head中只记录一次，在slot_indices和topk_ranks中也只记录一次。

                slot_indices[num_topk_ranks]记录的是在dispatch阶段token token_idx分发到第num_topk_ranks个激活专家所在的rank的环形缓冲区的物理索引。
                */
                int num_topk_ranks = 0, topk_ranks[kNumRanks], slot_indices[kNumRanks];
                // 这里为什么是去每个rank都查询一遍？就没有更方便的方式吗？
                #pragma unroll
                for (int i = 0; i < kNumRanks; ++i) {
                    /* warp中的前kNumRanks个线程才去读取过rank lane_id的环形缓冲区的token token_idx数据的，而且有的读取到的还可能是负数。
                    因此，要遍历token token_idx在dispatch阶段发送到各个rank的情况，并且将这个情况告诉warp中的所有线程，这样，这个warp中的所有线程对token token_idx当初在dispatch阶段的发送情况都掌握，
                    于是就可以在后面读写token数据时用warp的coalescing访问方式提升性能。

                    对于不同的token，dispatch阶段发送到的rank的个数不一样，因为它的多个激活专家可能在同一个rank上，也可能因为“每个token最多发送到$$M=4$$ 个节点”导致num_topk_ranks不一样。
                    */
                    auto expected_head_i = __shfl_sync(0xffffffff, expected_head, i);
                    if (expected_head_i >= 0) {  // 如果dispatch阶段有发送过token token_idx到rank i。
                        /* 由expected_head_i % num_recv_buffer_tokens可见，expected_head记录的也是从0到正无穷范围的环形队列索引值。
                        
                        topk_ranks[num_topk_ranks]记录的是slot_indices的每个索引对应这个激活专家所在的rank的编号。
                        */
                        slot_indices[num_topk_ranks] = expected_head_i % num_recv_buffer_tokens;
                        topk_ranks[num_topk_ranks++] = i;
                    }
                }

                // Wait shared memory release
#ifndef DISABLE_SM90_FEATURES
                tma_store_wait<0>();  // 等待下面的for循环中之前的TMA任务全部完成。
                __syncwarp(); // 同步warp中的所有线程对TMA任务全部完成这一信息的可见性。
#endif

                // Reduce data with pipeline，一个warp对应一条pipeline
                constexpr int kNumStages = 8;
                // kNumTMABytesPerWarp是4096
                /*
                每个warp中的每个线程都可以同时把自己要复制的kNumStages个int4写入到共享内存tma_buffer中。
                */
                EP_STATIC_ASSERT(kNumStages * 32 * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid count");
                #pragma unroll
                /* 对于token token_idx hidden的每个int4，都聚合dispatch阶段分发到各个激活expert输出的hidden的int4。
                每轮for循环，有 32 *（num_recv_warps - 1）个线程执行，并且每个warp的32个线程coalescing访问同一个token的连续的32 * 128字节的数据，即32个int4。
                重点来了！每个warp负责一个token在不同的num_topk_ranks个rank中的访问，并且这个token不会被别的warp染指去访问。
                不同的warp访问的token是不一样的。这一点和一个很长的数据的循环步进访问不一样。
                因此，这里的循环步进应该叫做“warp循环步进”。
                */
                for (int i = lane_id; i < hidden_int4; i += 32) {  // 例如：hidden_int4 = 256 = 256个int4 = 256 * 16 bytes = 4096 bytes
                    // Read bias
                    // TODO: make it as a template
                    /*
                    循环体内，每次处理一个int4的值。
                    在CUDA中，int4 是一个向量类型，不是"4字节整数"）。CUDA的int4定义：
                    struct int4 {
                        int x, y, z, w;  // 4个int
                    };
                    之所以涉及int4结构，是因为现代 GPU（如 NVIDIA GPU）支持向量化内存操作，可以一次加载/存储 16 字节（128 位）
                    bias_0_int4和bias_1_int4，shape: [num_recv_tokens, hidden]
                    num_recv_tokens是接收端要接收的token数量（原本dispatch的发送端从Multihead Attention传入的token数量）
                    */
                    int4 bias_0_value_int4 = bias_0_int4 != nullptr ? __ldg(bias_0_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);
                    int4 bias_1_value_int4 = bias_1_int4 != nullptr ? __ldg(bias_1_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);

                    // Read buffers
                    int4 recv_value_int4[kNumRanks];
                    /*
                    注意：这里为什么把token token_idx的第lane_id个int4在多个rank上的读取放在最里层的for循环中？
                    答案：为了对各个接收端rank的内存访问负载均衡，也就是使得各个接收端rank的Memory Controller负载均衡。
                         如果把for (int j = 0; j < num_topk_ranks; ++j)放在for (int i = lane_id; i < hidden_int4; i += 32)的外层，
                         那么就会造成当前block的（num_recv_warps - 1）个warp都先去访问在某个接收端rank的对应warp的token的所有int4，
                         这样就会使得一个rank在某一段时间接受来自当前block的（num_recv_warps - 1）个warp的经由NVLink的内存访问，
                         然后在下一时段又完全没有这（num_recv_warps - 1）个warp对这个rank的内存访问，因为这些warp都去访问别的rank的缓冲区中的token数据了。
                         这样就会造成负载不均衡。
                    */
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j)  // 对于token token_idx 发送到的num_topk_ranks个rank，在这里用warp coalescing访问相同位置的int4。
                        /* 对于每个激活专家所在的rank，都读取其在x中的数据。
                        channel_x_buffers[topk_ranks[j]]就是channel responsible_channel的rank topk_ranks[j]的记录cimbine发送端发送过来的expert输出的token数据，
                        其大小是num_recv_buffer_tokens * hidden_int4 * sizeof(int4)

                        slot_indices[j]表示的是在dispatch阶段token token_idx分发到rank topk_ranks[j]后，被rank topk_ranks[j]上由token token_idx激活的多个专家的输出加权求和之后的结果
                        在combine接受端（注意不是当前接收端，channel_x_buffers使用的是[topk_ranks[j]]，即rank topk_ranks[j]）的环形缓冲区中的物理地址。

                        topk_ranks[j]记录的是token token_idx发送到的第j个rank的rank id。
                        
                        环形缓冲区中，每个token hidden的大小是hidden_int4 * sizeof(int4)，
                        因此，slot_indices[j] * hidden_int4 + i 是token token_idx在环形缓冲区中的第i个int4的索引号。
                        */
                        recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);

                    // Reduce bias
                    float values[kDtypePerInt4];  // 注意这里的values是float类型的数组，在进行加法运算的时候，可以保持数据精度，不会因为dtype_t类型而丢失精度。
                    /*
                    dtype_t 是 nv_bfloat16（bfloat16）类型，nv_bfloat16 是 2 字节的浮点类型。
                    &bias_0_value_int4 获取该 int4 变量的地址（类型为 int4*）。
                    reinterpret_cast<const dtype_t*>：将 int4* 重新解释为 const dtype_t*。
                    这样就可以将 16 字节的 int4 当作 dtype_t 数组来访问，从而访问到int4的每个dtype_t类型的值。

                    int4 在这里只是作为内存容器，实际存储的是 nv_bfloat16 的位模式。
                    reinterpret_cast 只改变指针类型，不改变内存内容。数据类型转换发生在static_cast<float>中。
                    4个4字节int值组成的int4，转换为8个2字节nv_bfloat16值组成的dtype_t数组。
                    所以，bias_0_values实际上是nv_bfloat16[8]类型的数组。

                    完整的数据流类型转换：
                        读取阶段：int4 (16字节) → 转换为 nv_bfloat16[8] → 转换为 float[8]
                        计算阶段：float[8] 进行高精度计算
                        写入阶段：float[8] → 转换为 nv_bfloat16[8] → 打包回 int4 (16字节)
                    */
                    auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
                    auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        // 对于每一个具体值，都加上bias_0和bias_1的对应值。但是为什么要搞两个bias，为什么不只搞一个呢？不知道bias是从哪里传入的
                        values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);

                    // Reduce all-to-all results. 聚合dispatch阶段发送到的各个激活专家所在的rank的经过expert计算后加权求和后的token token_idx hidden的第i个int4。
                    #pragma unroll
                    for (int j = 0; j < num_topk_ranks; ++j) {
                        auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                        #pragma unroll
                        for (int k = 0; k < kDtypePerInt4; ++k)
                            /* 每个dtype_t类型的值加上bias。Combine (reduce) tokens (addition **without** weights) from different ranks，不加权重。
                            注意: 传入 combine 的 x 的每个 token，是该 token 在该 rank 上命中的多个专家的输出，在 rank 内部已经按这些专家在 topk_weights 中的权重加权求和 的结果。
                                  rank 间不再在 combine 里做一次按权重的加权，combine 只做 对多 rank 的（已加权）输出做加法。
                            */
                            values[k] += static_cast<float>(recv_value_dtypes[k]);
                    }

                    // Cast back to `dtype_t`
                    int4 out_int4;  // 这里已经在栈上为 out_int4 分配了内存空间。
                    auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
                    #pragma unroll
                    for (int j = 0; j < kDtypePerInt4; ++j)
                        out_dtypes[j] = staticcast<dtype_t>(values[j]);  // 将float类型的values转换为nv_bfloat16类型的out_dtypes，继续保持数据的低精度低内存。
/*
for (int i = lane_id; i < hidden_int4; i += 32) {
    if (i < hidden_int4_aligned) { 
        tma_store_wait<kNumStages - 1>();
        __syncwarp();
        auto tma_stage_idx = (i / 32) % kNumStages;
        reinterpret_cast<int4*>(tma_buffer)[tma_stage_idx * 32 + lane_id] = out_int4;
        tma_store_fence();
        __syncwarp();
        if (elect_one_sync()) {  // 选择warp中的一个线程发起TMA任务，这一个任务就把stage tma_stage_idx的32个int4都传输到全局内存中。
            auto tma_bytes = min(32, hidden_int4 - i) * static_cast<int>(sizeof(int4));
            tma_store_1d(reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * 32,
                            recv_int4 + token_idx * hidden_int4 + i,
                            tma_bytes,
                            false);
        }
        __syncwarp();
    } 
}

for (int i = lane_id; i < hidden_int4; i += 32) {
    if (i < hidden_int4_aligned) {
        if (lane_id < kNumStages) {
            tma_store_wait<0>();
        }
        auto tma_stage_idx = (i / 32) % kNumStages;
        reinterpret_cast<int4*>(tma_buffer)[tma_stage_idx * 32 + lane_id] = out_int4;
        tma_store_fence();
        __syncwarp();
        if (lane_id < kNumStages) {
            auto tma_bytes = min(32, hidden_int4 - i) * static_cast<int>(sizeof(int4));
            tma_store_1d(reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * 32,
                            recv_int4 + token_idx * hidden_int4 + i,
                            tma_bytes,
                            false);
        }
        __syncwarp();
    }
}
                    } else {
#endif
                        // 将token token_idx hidden的第i个int4的值out_int4写入recv_int4，即写入recv_x。
                        recv_int4[token_idx * hidden_int4 + i] = out_int4;
#ifndef DISABLE_SM90_FEATURES
                    }
                }
                // Flush all stores
                tma_store_wait<0>();
                __syncwarp();
#endif
                }
*/
#ifndef DISABLE_SM90_FEATURES
                    if (i < hidden_int4_aligned) { // 虽然这里写的是i < hidden_int4_aligned，但是实际上还是会执行最后的(hidden_int4 - hidden_int4_aligned)个int4。
                        // Wait TMA arrival
                        // 只要剩余的tma_store_1d任务数小于kNumStages，当前线程就可以继续执行。
                        /*
                        注意: 只等待当前执行线程自身提交的批量异步组的任务，而不是等待其他线程提交的批量异步组的任务。
                        如果之前的if (elect_one_sync())没有选中过当前线程去提交一整个stage的TMA数据传输，那么当前线程就可能没有任务，等待必过。
                        如果当前线程之前提交过至少kNumStages次bulk_group任务，那么这里至少要等待还剩（kNumStages - 1）次bulk_group任务未完成，才能继续执行。
                        这个前提是: 同一个线程连续提交的多个bulk_group任务是按照 FIFO 的顺序执行的。如果当前还剩下(kNumStages - 1)次bulk_group任务未完成，
                        kNumStages个bulk_group任务中已经完成了的那个肯定就是tma_stage_idx序号跟当前还剩下(kNumStages - 1)次bulk_group任务的tma_stage_idx序号不一样，
                        因而已经完成了的那个bulk_group任务对应的共享内存就可以覆盖使用了，那么当前线程就可以继续执行了。
                        但是，32个线程被随机分配kNumStages个stage的数据复制TMA任务，同一个线程被被连续选中kNumStages个，概率太低了，概率为：8/32^8 = 1/2^24，几乎不可能发生。
                        因此，每次每个线程执行下面的tma_store_wait<kNumStages - 1>()时，几乎都不会真的需要等待，会直接继续执行下面的代码。
                        */
                        tma_store_wait<kNumStages - 1>();
                        /* 
                        为了保证同一个warp内的32个线程都已经可见这一信息，需要__syncwarp()来保证同一warp内的32个线程都已经可见这一信息。
                        TODO
                        */
                        __syncwarp();

                        // Write into TMA buffer
                        /*
                        256个int4
                        (i / 32) 表示当前warp要把自己负责的所有从全局内存复制到共享内存的int4中的第几个int4。
                        (i / 32) % kNumStages 表示当前warp使用kNumStages级的TMA并行时，当前要处理的int4安排在第几个stage。
                        */
                        auto tma_stage_idx = (i / 32) % kNumStages;
                        /*
                        out_int4 的数据在寄存器上，现在是把数据写入到共享内存tma_buffer中。
                        out_int4 是当前线程负责的token token_idx hidden的第i个int4。
                        tma_buffer是当前warp在共享内存中的大小为 kNumTMABytesPerWarp 的缓冲区的起始地址。
                        (tma_buffer)[tma_stage_idx * 32 + lane_id] 表示当前warp的线程land_id的第tma_stage_idx个stage的int4的共享内存缓冲区位置。

                        每个warp中的每个线程都可以同时把自己要复制的kNumStages个int4写入到共享内存tma_buffer中。这样可以使用TMA并行优化。
                        EP_STATIC_ASSERT(kNumStages * 32 * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid count");

                        之所以要用共享内存缓冲而不是直接把out_int4的数据写入全局内存中，是因为TMA不能从寄存器读写。
                        */
                        reinterpret_cast<int4*>(tma_buffer)[tma_stage_idx * 32 + lane_id] = out_int4;

                        // Issue TMA
                        /* 
                        asm volatile("fence.proxy.async.shared::cta;");  // 官方文档最权威: 等待共享内存写入操作对 TMA 引擎可见。
                        更精确地说是: 当前执行线程可见“共享内存写入操作对 TMA 引擎可见”这一信息所在的内存视图。
                        这也是为什么每个线程都要执行tma_store_fence()，而不是只用warp中的一个线程执行即可。
                        因为GPU内存系统复杂，其他线程可能不会立刻看到完全一致的内存视图。
                        */
                        tma_store_fence();
                        /*
                        After syncthreads, writes by all threads are visible to TMA engine.
                        __syncwarp()之后，所有线程对共享内存的写入操作对TMA引擎可见。
                        这个warp的线程同步是为了使得各个线程都把自己要复制的int4写入到共享内存tma_buffer中，才好为接下来发起TMA任务做准备。
                        虽然前面有tma_store_fence()保证
                        */
                        __syncwarp();
                        if (elect_one_sync()) {  // 选择warp中的一个线程发起TMA任务，这一个任务就把stage tma_stage_idx的32个int4都传输到全局内存中。
                            // 当前warp的所有线程在当前stage tma_stage_idx负责的所有int4都写入到共享内存tma_buffer中之后，就可以发起TMA任务了。
                            // 注意: 一个warp的所有线程在同一个stage的int4是在共享内存中连续存储的。从[tma_stage_idx * 32 + lane_id]就可以看出来。
                            // 因为hidden_int4_aligned = align_down(hidden_int4, 32)，所以如果最后一次min(32, hidden_int4 - i)是(hidden_int4 - i)，那么就是要传输(hidden_int4 - hidden_int4_aligned)个int4。
                            auto tma_bytes = min(32, hidden_int4 - i) * static_cast<int>(sizeof(int4));
                            tma_store_1d(reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * 32,
                                         recv_int4 + token_idx * hidden_int4 + i,
                                         tma_bytes,
                                         false);
                        }
                        // 
                        __syncwarp();
                    } else {
#endif
                        recv_int4[token_idx * hidden_int4 + i] = out_int4;
#ifndef DISABLE_SM90_FEATURES
                    }
#endif
                }
                // Reduce `topk_weights`
                if (lane_id < num_topk) {
                    float value = 0;
                    #pragma unroll
                    for (int i = 0; i < num_topk_ranks; ++i)
                        /*
                        channel_topk_weights_buffers[topk_ranks[i]]是记录当前rank作为dispatch发送端时通过channel responsible_channel发送到rank topk_ranks[i]的所有token的topk权重。
                        其大小是num_recv_buffer_tokens * num_topk * sizeof(float)。权重值是float类型的。

                        slot_indices[i]表示的是在dispatch阶段token token_idx分发到rank topk_ranks[i]后，被rank topk_ranks[i]上由token token_idx激活的多个专家的输出加权求和之后的结果
                        在combine接受端（注意不是当前接收端，channel_x_buffers使用的是[topk_ranks[i]]，即rank topk_ranks[i]）的环形缓冲区中的物理地址。
                        
                        由于环形缓冲区的token维度大小和channel_topk_weights_buffers的token维度大小都是num_recv_buffer_tokens，而且是一一对应的，
                        所以slot_indices[i]也能表示channel_topk_weights_buffers在token维度的索引，即token token_idx在channel_topk_weights_buffers中的token索引。
                        slot_indices[i] * num_topk + lane_id 表示的是在channel_topk_weights_buffers中的token token_idx的第lane_id个topk权重的索引。

                        重点：每个token记录在各个rank的topk个权重值中，如果这个token发送到这个rank的expert有m个，那么这个rank中记录的这个token的topk个权重值中就有m个权重值，
                             其余（topk-m）个权重值都为0，但即使是0，也要记录着。

                        所以，这里的for循环有个特点，那就是对于token token_idx来说，在num_topk_ranks个rank上的第lane_id个权重，有且仅有一个不是0，其余（num_topk - 1）个权重都是0。
                        举例：如果激活专家数是4，即topk=4。4个rank，每个rank有2个expert，
                        4个rank上的expert分布是：rank0: [expert0, expert1]；rank1: [expert2, expert3]；rank2: [expert4, expert5]；rank3: [expert6, expert7]。
                        channel_topk_weights_buffers中不是topk的权重，设置为0。
                        而token token_idx记录在各个rank上的topk weight是：
                            rank0_topk_weight：[0.0, 0.0, 0.0, 0.2]；
                            rank1_topk_weight：[0.1, 0.3, 0.0, 0.0]；
                            rank2_topk_weight：[0.0, 0.0, 0.0, 0.0]；
                            rank3_topk_weight：[0.0, 0.0, 0.4, 0.0]。
                        即token token_idx路由到了4个expert：rank0的expert3，rank1的expert4和expert5，rank3的expert14。

                        在dispatch的接收端写入channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = weight_value;时，
                        每次都会在4个线程中分别把token token_idx的4个全局激活专家id[3, 4, 5, 14]分别去rank0中判断每个全局激活专家的id是否在ranki中的专家id范围中，
                        如果在，那么就记录这个全局专家的权重值weight_value。如果不存在，那么就记录这个全局专家在ranki中的权重值为0。

                        上面的例子中，同一列的4个weight_value **有且仅有** 一个非0的权重值。

                        注意：上面的例子中，rank0的expert3、rank1的expert4和expert5、rank3的expert14都是全局expert ID，而不是局部expert ID。
                        num_topk_ranks = 3，即token token_idx发送到了3个rank：rank0，rank1，rank3。而token token_idx不需要发送到rank2。
                        topk_ranks[0] = 0，topk_ranks[1] = 1，topk_ranks[2] = 3。
                        权重累加是跨rank进行的，每个expert位置的权重会被累加。
                        land 0在for循环中遍历到的是rank0_topk_weight[0] + rank1_topk_weight[0] + rank3_topk_weight[0] = 0.0 + 0.1 + 0.0 = 0.1。
                        land 1在for循环中遍历到的是rank0_topk_weight[1] + rank1_topk_weight[1] + rank3_topk_weight[1] = 0.0 + 0.3 + 0.0 = 0.3。
                        land 2在for循环中遍历到的是rank0_topk_weight[2] + rank1_topk_weight[2] + rank3_topk_weight[2] = 0.0 + 0.0 + 0.4 = 0.4。
                        land 3在for循环中遍历到的是rank0_topk_weight[3] + rank1_topk_weight[3] + rank3_topk_weight[3] = 0.2 + 0.0 + 0.0 = 0.2。

                        */
                        value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);
                    /* 
                    没有使用公式（18）的加权求和吗？ https://osvicyu5w5.feishu.cn/wiki/MWp0wGlFDi6zSJkwSy1cRrXan2X#share-GgD0d0COUoRdb0x5yqoco12encb
                    答案：有！
                    LyricZhao 说：对于普通核函数（normal kernels），一个 token 可能会在单个 GPU 中选择多个专家（expert），因此需要在合并前执行一个归约核函数（融合了 topk_weights 乘法操作），以节省带宽。
                    所以，传入combine的参数x的每一个token数据，都是这个token在这个rank上激活的属于这个rank的多个专家的输出加权求和之后的结果。
                    而不同rank之间的加权求和，也不再combine函数中做，combine函数只是提供recv_int4和recv_topk_weights，供调用DeepEP的python代码使用，进行rank间的token数据（这个token数据是调用combine函数之前已经各个rank内部加权求和过的，比如某个dispatch接收端有多个expert被一个token激活，那么就需要在这个dispatch接收端内部先进行rank内加权求和）加权求和。
                    
                    记录的token token_idx的全局第lane_id个topk的权重，这是按照全局专家id来记录的，而不是按照局部专家id来记录的。记录的权重值是float类型的。
                    */
                    recv_topk_weights[token_idx * num_topk + lane_id] = value;
                }

                // Update head
                if (lane_id < kNumRanks)
                    /* 
                    每个 warp 处理完一个 token 后，更新它从对应 rank 读取到的 head。
                    expected_head < 0表示token token_idx在dispatch的时候并没有发送到rank lane_id，也就是当前rank没有依据send_head中读到这个token。
                    但是，dispatch阶段没有发送是一回事，更新warp_channel_head_idx确是另一回事了，只有更新了warp_channel_head_idx，
                    才能更新channel_head_idx_ptr，才能在combine发送端通过读取head指针来判断接收端把数据消费到了什么位置。

                    这里为什么要搞成二维数组呢？是因为 expected_head 是从 lane_id 读取的，和 land_id 是一一对应的。
                    因为有前面的 while (__any_sync(...)) 和 __syncwarp() ，所以当一个线程执行到当前位置的时候，warp内的前kNumRanks个线程其实都到达了这里。
                    */
                    warp_channel_head_idx[recv_warp_id][lane_id] = (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
            }

            // Retired
            // 上面的for循环执行的是一个warp要负责的所有token的聚合，这个循环执行完了，也就是当前的warp的任务执行完了。所以for循环之后就可以 retire 这个warp了。
            __syncwarp();
            if (elect_one_sync())
                warp_retired[recv_warp_id] = true;
        }
    }
}

void combine(cudaDataType_t type,
             void* recv_x,
             float* recv_topk_weights,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* src_idx,
             const int* rank_prefix_matrix,
             const int* channel_prefix_matrix,
             int* send_head,
             int num_tokens,
             int num_recv_tokens,
             int hidden,
             int num_topk,
             void** buffer_ptrs,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_sms,
             int num_max_send_tokens,
             int num_recv_buffer_tokens) {
    constexpr int kNumThreads = 768;
    constexpr int kNumTMABytesPerWarp = 4096;
#ifndef DISABLE_SM90_FEATURES
    constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / 32);
#endif

#define COMBINE_LAUNCH_CASE(dtype, ranks)                                      \
    {                                                                          \
        auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \
        SET_SHARED_MEMORY_FOR_TMA(kernel);                                     \
        LAUNCH_KERNEL(&cfg,                                                    \
                      kernel,                                                  \
                      reinterpret_cast<dtype*>(recv_x),                        \
                      recv_topk_weights,                                       \
                      reinterpret_cast<const dtype*>(x),                       \
                      topk_weights,                                            \
                      reinterpret_cast<const dtype*>(bias_0),                  \
                      reinterpret_cast<const dtype*>(bias_1),                  \
                      src_idx,                                                 \
                      rank_prefix_matrix,                                      \
                      channel_prefix_matrix,                                   \
                      send_head,                                               \
                      num_tokens,                                              \
                      num_recv_tokens,                                         \
                      hidden,                                                  \
                      num_topk,                                                \
                      buffer_ptrs,                                             \
                      rank,                                                    \
                      num_max_send_tokens,                                     \
                      num_recv_buffer_tokens);                                 \
    }                                                                          \
    break
#define COMBINE_DTYPE_LAUNCH_CASE(dtype)                 \
    SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); \
    break

    // Even-numbered blocks for sending, odd-numbered blocks for receiving
    EP_HOST_ASSERT(num_sms % 2 == 0);
    EP_HOST_ASSERT(kNumThreads >= num_ranks * 32);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
}

}  // namespace intranode

}  // namespace deep_ep
