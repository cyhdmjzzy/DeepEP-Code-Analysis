#pragma once

#include "kernels/api.cuh"
#include "kernels/exception.cuh"

namespace deep_ep {

template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

struct Config {
    int num_sms;
    int num_max_nvl_chunked_send_tokens;
    int num_max_nvl_chunked_recv_tokens;
    int num_max_rdma_chunked_send_tokens;
    int num_max_rdma_chunked_recv_tokens;

    Config(int num_sms,
           int num_max_nvl_chunked_send_tokens,
           int num_max_nvl_chunked_recv_tokens,
           int num_max_rdma_chunked_send_tokens,
           int num_max_rdma_chunked_recv_tokens)
        : num_sms(num_sms),
          num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
          num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
          num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
          num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {
        EP_HOST_ASSERT(num_sms >= 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens > 0 and num_max_nvl_chunked_recv_tokens > 0);
        EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens > 0 and num_max_rdma_chunked_recv_tokens > 0);

        // Ceil up RDMA buffer size
        this->num_max_rdma_chunked_recv_tokens = align_up<int>(num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens < num_max_rdma_chunked_recv_tokens);
        // NOTES: this assertion is related to RDMA lazy head update, we must ensure senders always have space to push
        /*
        关于head指针不需要用acquire-release保持head指针和实际数据的内存读写顺序（tail指针需要acquire-release），
        我一开始是这么理解的：
            head指针只用于空间检查，不涉及数据可见性。在生产者使用ld_volatile_global轮询等待读取channel_head_idx，等到了更大的head，说明消费者已经把新head之前更小的位置的数据消费完了，
            此时生产者可以往老head到新head之间的位置写入数据了，覆盖写入，覆盖了无所谓，反正消费者也已经不需要了。消费者消费数据，在乎的是head到tail之间的位置的数据。
            另外，即使消费者推进了head对生产者尚不可见，生产者读取到的还是老head，那么生产者写数据时也只会导致不写老head到新head之间的空间，增加等待时间，仅此而已。
            生产者在下次读head读到新的head之后，还是会把数据写入到老head到新head之间的位置。所以这只会导致生产者慢一点写入数据。所以不管怎样，都不会导致数据错乱。

        后来我又有一个疑问：
            如果消费者在acquire读取tail之后，满足条件，开始读取token数据，消费者的本意是先读取token数据，读取完了之后再更新head指针，但是因为没有用release写head指针，
            会由于cuda的编译器优化或硬件重排指令导致“更新head的指令的实际执行”发生在消费token数据之前，也就是实际执行时是先推进head再消费数据。如果这样，就可能导致生产者读取到新head，满足生产者发起生产新一批数据的条件，
            然后生产者就往环形队列中的老head到新head之间的以为已经空闲的空间覆盖写入新的token数据，但是此时消费者正在消费老head到新head之间的token数据，这样就会导致对同一片空间同时写数据和读数据，就会造成错乱。
        
        对于上面可能出现的数据竞争导致的数据错乱，然后我就想:
            一般来讲，生产者写数据是从tail开始写，只要环形队列总的物理slot数量足够大，就不会在读到新的head之后在写数据时往老head到新head之间的空间覆盖写（数据竞争时，这段空间就是消费者正在消费的空间）。
            那么就应该有个前提，那就是生产者每次写一个批量的数据所占用的环形队列slot数量不会超过这个环形队列总物理slot数量的一半，这样能避免“缠绕”的问题。
            也就是说，生产者每次往环形队列写一个批量的数据不会覆盖到上一次写一个批量的数据所占用的环形队列的物理slot空间，这也就是缓冲区的价值所在，否则如果每次写数据都会占用上一次写的空间，
            那么这个环形队列缓冲区就失去了意义，相当于没有缓冲至少一个批量的数据的作用，那还不如不要用缓冲区直接端到端写数据。
        
        我和作者对这个问题的理解是一致的: 生产的批量大小 <= 接收的批量大小 / 2 ，
        这里的接收的批量大小（num_max_rdma_chunked_recv_tokens）就是整个环形队列的物理slot的大小。
        证据就在 internode.cu 的dispatch函数和combine函数中创建 rdma_channel_data 的地方。

        感受: 我对这个问题的理解经历了漫长的过程。直到此刻2026-02-13 23:30，我把我上面的三步思考过程告诉cursor的Grok Code模型，它帮我找到了下面这个断言。我非常激动，感觉在此和作者产生了共鸣。
             我之前对“head、tail、release、acquire、relaxed、volatile”这些概念结合具体业务一直有各种“乌云”，到此刻我才感觉彻底理解了（当然了，还没有触及编译器，ptx和硬件的更底层，今后加油追根溯源）
        */
        EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <= num_max_rdma_chunked_recv_tokens / 2);
    }

    size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;  // 最大 topk 值（保守估计），使用保守的最大值，确保 buffer 足够大
        constexpr int kNumMaxScales = 128;  // 最大 scales 数量（保守估计），使用保守的最大值，确保 buffer 足够大
        EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);  // 确保 rank 数量符合 NVLink 限制
        EP_HOST_ASSERT(num_ranks <= NUM_MAX_NVL_PEERS or num_sms % 2 == 0);  // 因为每个 channel 需要 2 个 SM
        const auto num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
        const auto num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
        const int num_channels = num_sms / 2;

        size_t num_bytes = 0;
        /* 2 * num_rdma_ranks：internode 的 nvl_channel_prefix_start 和 nvl_channel_prefix_end（每个大小为 kNumRDMARanks）
        +3：channel_head_idx、channel_tail_idx，以及一个额外项（可能是其他元数据）
        */
        num_bytes += num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);
        // 数据缓冲区channel_x_buffers
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * hidden_bytes;
#ifndef DISABLE_NVSHMEM
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * internode::get_source_meta_bytes();
#endif
        // channel_topk_idx_buffers
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t);
        // channel_topk_weights_buffers
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(float);
        // channel_x_scales_buffers
        num_bytes += num_channels * num_nvl_ranks * num_max_nvl_chunked_recv_tokens * kNumMaxScales * sizeof(float);
        num_bytes = ((num_bytes + 127) / 128) * 128;  // 对齐128字节，确保内存对齐
        return num_bytes;
    }

    /*
    计算RDMA对称内存缓冲区所需的总字节数，用于预分配缓冲区内存。
    */
    size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
#ifndef DISABLE_NVSHMEM
        // Legacy mode
        if (num_ranks <= NUM_MAX_NVL_PEERS)
            return 0;

        // Below are some assumptions
        // TODO: add assertions
        constexpr int kNumMaxTopK = 128;
        constexpr int kNumMaxScales = 128;
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(num_sms % 2 == 0);
        const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
        const int num_channels = num_sms / 2;

        /*
        num_max_rdma_chunked_recv_tokens: 每个节点为每个channel准备的环形接收缓冲区的token数量。
        下面的每次计算都 * 2 是因为decoupled模式，每个rank每个channel有发送和接收两个缓冲区。

        */
        size_t num_bytes = 0;
        /*
        gbl_channel_prefix_matrix:  [kNumRanks, num_channels]:
          [0..7] : 表示当前rank作为发送端从channel 0到channel (channel_id - 1)（包含）累计要发送到节点dst_rdma_rank的 8 个rank的token数量之和（r2r的channel级别前缀和）。
          [8..15]: 表示当前rank作为发送端从channel 0到channel channel_id（包含）      累计要发送到节点dst_rdma_rank的 8 个rank的token数量之和（r2r的channel级别前缀和）。
        rdma_channel_prefix_matrix: [kNumRDMARanks, num_channels]:
          [16]:    表示当前rank作为发送端从channel 0到channel (channel_id - 1)（包含）累计要发送到节点dst_rdma_rank的token数量之和（r2n的channel级别前缀和）。
          [17]:    表示当前rank作为发送端从channel 0到channel channel_id（包含）      累计要发送到节点dst_rdma_rank的token数量之和（r2n的channel级别前缀和）。
        注意: 以上这些所谓的索引，既可以看作是发送端rank中rdma send_buffer的对应写接收端rank或节点的token的索引，
             也可以看作是接收端rank中rdma recv_buffer的对应发送端rank的token的索引。
        */
        num_bytes += num_channels * num_rdma_ranks * (NUM_MAX_NVL_PEERS * 2 + 2) * 2 * sizeof(int);

        /*
        每个rank对每个channel的token的hidden数据缓冲区。
        */
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * internode::get_source_meta_bytes() * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(topk_idx_t) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxTopK * sizeof(float) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * kNumMaxScales * sizeof(float) * 2;
        num_bytes += num_channels * num_rdma_ranks * num_max_rdma_chunked_recv_tokens * sizeof(int4) * 2;
        num_bytes = ((num_bytes + 127) / 128) * 128;
        return num_bytes;
#else
        EP_HOST_ASSERT(false and "NVSHMEM is disable during compilation");
#endif
    }
};

struct LowLatencyBuffer {
    int num_clean_int = 0;

    /*
    为什么LL模式下需要发送缓冲区？为什么不直接将本地专家的输出经过rdma网络传输到其它节点的combine接收端缓冲区上？
    1、RDMA 注册内存要求：
        RDMA 需要使用预注册的内存区域。普通 GPU 内存无法直接用于 RDMA 传输，需要专门的 RDMA 缓冲区。
    2、异步通信处理：
        RDMA 是异步的，发送端发出后立即返回。如果直接使用本地专家的输出，那么通信流中的GPU就得同步一直等待，无法实现基于hook的通信计算重叠。
        需要缓冲区处理发送/接收时序差异，避免发送端覆盖正在传输的数据。
    3、网络协议开销：
        IBGDA 协议需要额外的元数据和控制信息，缓冲区提供空间存储这些信息。
    */
    void* dispatch_rdma_send_buffer = nullptr;
    void* dispatch_rdma_recv_data_buffer = nullptr;
    int* dispatch_rdma_recv_count_buffer = nullptr;

    void* combine_rdma_send_buffer = nullptr;
    void* combine_rdma_recv_data_buffer = nullptr;
    int* combine_rdma_recv_flag_buffer = nullptr;

    // 用于使用torch::from_blob创建tensor时设置内存块的起始指针。 实际上和combine_rdma_recv_data_buffer一样，都是指向dispatch接收端缓冲区的起始地址。
    void* combine_rdma_send_buffer_data_start = nullptr;
    size_t num_bytes_per_combine_msg = 0;

    std::pair<int*, int> clean_meta() {
        EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer == combine_rdma_recv_flag_buffer);
        return {dispatch_rdma_recv_count_buffer, num_clean_int};
    }
};

struct LowLatencyLayout {  // 双缓冲区布局。只涉及缓冲区中指针位置的计算，不涉及内存分配。
    size_t total_bytes = 0;
    /* internode_ll使用双缓冲区的原因：
    每个缓冲区都有独立的发送缓冲区（可以用于dispatch发送或者combine发送）和接收缓冲区（可以用于dispatch接收或者combine接收）。
    */
    LowLatencyBuffer buffers[2];

    template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
    out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
        return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);
    }

    LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
        
        /*
        rdma_buffer: 目标rank的RDMA接收缓冲区起始地址。所有rdma的缓冲数据都存储在这个缓冲区中。
        num_max_dispatch_tokens_per_rank: 接收端rank的每个专家为每个dispatch发送端rank预分配固定大小的缓冲区空间的大小。
        hidden: 每个token的hidden大小。
        num_ranks: 全部rank的数量。
        num_experts: 全部专家的数量。
        */
        
        // 用途：在 combine 阶段的 LogFMT 压缩中，用于存储每 128 个通道的 min/max，用于解码。
        const int num_scales = hidden / 128;

        // Dispatch and combine layout:
        //  - 2 symmetric odd/even send buffer
        //  - 2 symmetric odd/even receive buffers
        //  - 2 symmetric odd/even signaling buffers

        // Message sizes
        // NOTES: you should add a control `int4` for combine messages if you want to do data transformation
        // NOTES: `num_scales * sizeof(nv_bfloat162)` means the per-128-channel min/max
        /*
        nv_bfloat162 是 CUDA 的向量类型，包含两个 nv_bfloat16 值：
            struct __nv_bfloat162 {
                __nv_bfloat16 x;  // 第一个bfloat16值
                __nv_bfloat16 y;  // 第二个bfloat16值
            };
        
        */
        EP_HOST_ASSERT(num_scales * sizeof(float) <= hidden);
        /*
        num_bytes_per_dispatch_msg: dispatch阶段发送每个token数据需要占用的内存大小。

        假设hidden是4096，num_scales是32。
        无量化：hidden * sizeof(nv_bfloat16)（BF16 模式）。每个 token 的 hidden 数据（BF16 格式）
               大小：[16字节: int4索引] + [8192字节: BF16数据] = 8208字节
        有量化：hidden + num_scales * sizeof(float)（FP8 模式）
                hidden：expert 输出的 FP8 数据（每个元素 1 字节）
                num_scales * sizeof(float)：FP8 的缩放因子（每 128 通道一个 float）
               大小：[16字节: int4索引] + [4096字节: FP8数据] + [32字节: 32个float缩放因子] = 4144字节
        
        为什么要加int4索引？
        答：使用其中的第一个4字节的int表示dispatch要发送的这个token在发送端的输入tensor中的索引，
            这样在combine发送端可以知道将expert输出的每个token发送到combine接收端的哪个token，也就是从combine接收端缓冲区中取得token数据后复制到MoE输出tensor的哪个token位置。
        为什么使用 int4 而不是单个 int？
        答：内存对齐：int4（16 字节）对齐，便于向量化访问
            扩展性：预留 3 个 int 字段，便于未来扩展
            向量化优化：与后续数据部分的对齐一致，便于 SIMD 操作
        */
        size_t num_bytes_per_dispatch_msg = sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16), hidden + num_scales * sizeof(float));
        /* 
        num_scales * sizeof(nv_bfloat162)：LogFMT 压缩模式下，为每组存储一对 min/max 值，即每个 nv_bfloat162 存储：(min_value, max_value)
        hidden * sizeof(nv_bfloat16)：BF16 模式下，expert 输出的 hidden 数据（BF16 格式）
        大小：[128字节: 32个nv_bfloat162元数据] + [8192字节: BF16数据] = 8320字节
        */
        size_t num_bytes_per_combine_msg = num_scales * sizeof(nv_bfloat162) + hidden * sizeof(nv_bfloat16);

        // Send buffer
        /*
        num_max_dispatch_tokens_per_rank: dispatch阶段每个rank要发送的token的最大数量。
        num_bytes_per_dispatch_msg: dispatch阶段发送每个token数据需要占用的内存大小。
        dispatch_send_buffer_bytes: dispatch阶段发送缓冲区的大小。
        combine_send_buffer_bytes: combine发送端就是dispatch接收端，而每个dispatch接收端rank有可能接收到来自各个expert的token，
            因此，为了确保哪怕所有rank的所有的专家都往当前dispatch接收端rank发送token，也能有足够的空间，所以分配了num_experts * num_max_dispatch_tokens_per_rank个token的空间。
            因为num_bytes_per_combine_msg肯定是大于num_bytes_per_dispatch_msg的，所以每个token的信息预留的大小是num_bytes_per_combine_msg。
        send_buffer_bytes: 双缓冲区中，可能一个缓冲区一会儿用于dispatch，一会儿用于combine。所以对于send缓冲区的大小设置，需要取dispatch和combine中的较大值。
            类似的，下面的接收缓冲区的大小和信号缓冲区的大小设置，也需要取dispatch和combine中的较大值。
        */
        size_t dispatch_send_buffer_bytes = num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_send_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;  
        size_t send_buffer_bytes = std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
        EP_HOST_ASSERT(send_buffer_bytes % sizeof(int4) == 0);
        total_bytes += send_buffer_bytes * 2;  // 双缓冲区，所以乘以2。

        // Symmetric receive buffers
        // TODO: optimize memory usages
        /*
        dispatch_recv_data_buffer_bytes: dispatch接收端要预留接收的token数据的个数和combine发送端要预留接收的token数据的个数（combine_send_buffer_bytes）是一样的。只是每个token的msg大小不一样。
           注意，全局所有的专家数是num_experts，这里的接收端缓冲区是预留了所有的num_experts的缓冲区，每个专家的缓冲区有num_max_dispatch_tokens_per_rank个token空间，
           因此，即使所有的rank都把token发送给当前的接收端rank，接收端缓冲区都可以可以存下的。这样好奢侈啊！
           接收端每个expert的最大接收token数量是num_ranks * num_max_dispatch_tokens_per_rank，实际上也不可能超过这个值。
        combine_recv_buffer_bytes: combine阶段接收端要接收的token数据的缓冲区大小。
        recv_buffer_bytes: 双缓冲区中，可能一个缓冲区一会儿用于dispatch，一会儿用于combine。所以对于recv缓冲区的大小设置，需要取dispatch和combine中的较大值。
        */
        size_t dispatch_recv_data_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_recv_buffer_bytes = num_experts * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;
        // 接收端缓冲区既可以用于dispatch的，也可以用于combine的
        size_t recv_buffer_bytes = std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);
        EP_HOST_ASSERT(recv_buffer_bytes % sizeof(int4) == 0);
        total_bytes += recv_buffer_bytes * 2;  // 双缓冲区，所以乘以2。

        // Symmetric signaling buffers
        size_t dispatch_recv_count_buffer_bytes = num_experts * sizeof(int);
        // combine接收端和dispatch接收端都是以expert为 TODO
        size_t combine_recv_flag_buffer_bytes = dispatch_recv_count_buffer_bytes;
        size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes, combine_recv_flag_buffer_bytes);
        size_t signaling_buffer_bytes_aligned = align_up<size_t>(signaling_buffer_bytes, 128);
        total_bytes += signaling_buffer_bytes_aligned * 2;  // 双缓冲区，所以乘以2。

        // Assign pointers
        // NOTES: we still leave some space for distinguishing dispatch/combine buffer,
        // so you may see some parameters are duplicated
        for (int i = 0; i < 2; ++i) {
            buffers[i] = {static_cast<int>(signaling_buffer_bytes / sizeof(int)),
                          // signaling_buffer 放在缓冲区最前面
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),
                          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 + recv_buffer_bytes * i),
                          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
                          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
                          num_bytes_per_combine_msg};
        }
    }
};

size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, int hidden, int num_ranks, int num_experts) {
    auto num_bytes = LowLatencyLayout(nullptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts).total_bytes;
    return ((num_bytes + NUM_BUFFER_ALIGNMENT_BYTES) / NUM_BUFFER_ALIGNMENT_BYTES) * NUM_BUFFER_ALIGNMENT_BYTES;
}

}  // namespace deep_ep
