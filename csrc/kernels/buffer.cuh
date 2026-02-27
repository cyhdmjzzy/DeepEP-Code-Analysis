#pragma once

#include "configs.cuh"
#include "exception.cuh"

namespace deep_ep {

template <typename dtype_t>
struct Buffer {
private:
    uint8_t* ptr;

public:
    int64_t total_bytes;

    __device__ __forceinline__ Buffer() : ptr(nullptr), total_bytes(0) {}

    __device__ __forceinline__ Buffer(void*& gbl_ptr, int num_elems, int offset = 0) {
        total_bytes = num_elems * sizeof(dtype_t);
        ptr = static_cast<uint8_t*>(gbl_ptr) + offset * sizeof(dtype_t);
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ Buffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t* buffer() { return reinterpret_cast<dtype_t*>(ptr); }

    __device__ __forceinline__ dtype_t& operator[](int idx) { return buffer()[idx]; }
};

template <typename dtype_t, int kNumRanks = 1>
struct AsymBuffer {
private:
    uint8_t* ptrs[kNumRanks];
    int64_t num_bytes;

public:
    int64_t total_bytes;

    /* 
    举例:auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
    
    gbl_ptr：全局指针的引用，指向缓冲区内存的当前位置，构造后会向前移动
        void* ptr = ...;        // ptr 是 void* 类型的变量
        void*& ref = ptr;       // ref 是 ptr 的引用，是 ptr 的别名
        为什么使用 void*& 类型而不是 void* 类型？
        因为构造函数需要修改传入的指针。AsymBuffer(void* gbl_ptr, ...) 这种值传递的方式，只能修改局部副本，不会影响外部变量。

    num_elems：每个rank的元素数量
    num_ranks：rank数量（本项目中都是NUM_MAX_NVL_PEERS = 8）
    sm_id：当前SM的ID。如果用默认值0，则表示这个AsymBuffer是属于rank的，不是属于SM的。否则是属于SM的。
    num_sms：SM数量。如果用默认值1，则表示这个AsymBuffer是属于rank的，不是属于SM的。否则是属于SM的。
    offset：偏移量（默认0）
    */
    __device__ __forceinline__ AsymBuffer(void*& gbl_ptr, int num_elems, int num_ranks, int sm_id = 0, int num_sms = 1, int offset = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "");  // 只支持单rank的情况。
        num_bytes = num_elems * sizeof(dtype_t);

        // per_channel_bytes: 每个channel的总字节数（所有rank的数据总和）,每个channel会向num_ranks个rank通信。
        int64_t per_channel_bytes = num_bytes * num_ranks;
        /*
        AsymBuffer对象是针对整个rank的，所以total_bytes是整个rank的num_sms个sm使用的内存大小只和。
        一个sm对应一个channel。
        */
        total_bytes = per_channel_bytes * num_sms;
        ptrs[0] = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id + num_bytes * offset;
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ AsymBuffer(void** gbl_ptrs, int num_elems, int num_ranks, int sm_id = 0, int num_sms = 1, int offset = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "");
        num_bytes = num_elems * sizeof(dtype_t);

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms;
        for (int i = 0; i < kNumRanks; ++i) {
            ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + per_channel_bytes * sm_id + num_bytes * offset;
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        }
    }

    __device__ __forceinline__ void advance(int shift) {
        #pragma unroll
        for (int i = 0; i < kNumRanks; ++i)
            ptrs[i] = ptrs[i] + shift * sizeof(dtype_t);
    }

    __device__ __forceinline__ AsymBuffer advance_also(void*& gbl_ptr) {
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
        return *this;
    }

    template <int kNumAlsoRanks>
    __device__ __forceinline__ AsymBuffer advance_also(void** gbl_ptrs) {
        for (int i = 0; i < kNumAlsoRanks; ++i)
            gbl_ptrs[i] = static_cast<uint8_t*>(gbl_ptrs[i]) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks == 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t*>(ptrs[0] + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t* buffer_by(int rank_idx, int idx = 0) {
        EP_STATIC_ASSERT(kNumRanks > 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t*>(ptrs[rank_idx] + num_bytes * idx);
    }
};

/*
kDecoupled: 表示当前SymBuffer是否是双缓冲区模式。一般存储具体要传输token数据时，kDecoupled为true。
            而一般存储配置信息比如环形队列的head索引和tail索引时，kDecoupled为false。
*/
template <typename dtype_t, bool kDecoupled = true>
struct SymBuffer {
private:
    // NOTES: for non-decoupled case, `recv_ptr` is not used
    uint8_t* send_ptr;
    uint8_t* recv_ptr;
    int64_t num_bytes;

public:
    int64_t total_bytes;

    /*
    gbl_ptr: 全局指针。指向全局内存中的一个位置。
    num_elems: 元素数量。
    num_rdma_ranks: 节点数量，应该是rdma rank的数量。
    sm_id: 当前SM的ID。如果用默认值0，则表示这个SymBuffer是属于rank的，不是属于SM的。否则是属于SM的。
    num_sms: 当前GPU的SM数量。如果用默认值1，则表示这个SymBuffer是属于rank的，不是属于SM的。否则是属于SM的。
    */
    __device__ __forceinline__ SymBuffer(void*& gbl_ptr, int num_elems, int num_rdma_ranks, int sm_id = 0, int num_sms = 1) {
        // 当前rank对 (每个节点, 每个channel) 要传输的数据的字节数。
        num_bytes = num_elems * sizeof(dtype_t);

        // per_channel_bytes: 每个channel的字节数。每个channel会向所有的rdma rank发送数据。
        int64_t per_channel_bytes = num_bytes * num_rdma_ranks;
        /* 
        * num_sms: 整个对称内存是属于整个rank的，每个sm拥有自己的独立的对称内存空间。
        如果kDecoupled为true，则每个sm要作为一个channel的发送方和接收方。
        如果kDecoupled为false，则每个sm要作为一个channel的发送方或接收方。
            这就和 intranode.cu 中的 channel 和 sm 的关系一样，一个channel由两个sm负责，一个sm负责发送，一个sm负责接收。
        */
        total_bytes = per_channel_bytes * num_sms * (static_cast<int>(kDecoupled) + 1);
        /*
        如果 sm_id 是默认值0，则表示send_ptr 和 recv_ptr 都是对应一个rank的，因为整个对称内存是属于整个rank的。
        如果 sm_id 不是默认值0，则send_ptr 和 recv_ptr 都是对应一个sm的。
        */
        send_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * sm_id;
        // 如果kDecoupled为false，则recv_ptr指向对称内存最后一个字节的下一个字节，也没关系，反正在buffer函数中也不会用到。
        recv_ptr = static_cast<uint8_t*>(gbl_ptr) + per_channel_bytes * (sm_id + num_sms);
        // 与Buffer、AsymBuffer一样，移动gbl_ptr到下一个对称内存的位置。
        gbl_ptr = static_cast<uint8_t*>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ dtype_t* send_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`send_buffer` is only available for non-decoupled case");
        // idx 表示当前sm的对称内存中用于对rdma rank idx通信。
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t* recv_buffer(int idx = 0) {
        EP_STATIC_ASSERT(kDecoupled, "`recv_buffer` is only available for non-decoupled case");
        return reinterpret_cast<dtype_t*>(recv_ptr + num_bytes * idx);
    }

    // 单缓冲区模式下才用buffer函数
    __device__ __forceinline__ dtype_t* buffer(int idx = 0) {
        EP_STATIC_ASSERT(not kDecoupled, "`buffer` is only available for decoupled case");
        return reinterpret_cast<dtype_t*>(send_ptr + num_bytes * idx);
    }
};

}  // namespace deep_ep
