// Portions derived from NVSHMEM (https://developer.nvidia.com/nvshmem)
// Copyright (c) NVIDIA Corporation.
// Licensed under the NVSHMEM Software License Agreement (version: September 3, 2019).
// See full license at: https://docs.nvidia.com/nvshmem/api/sla.html
//
// Modified from original source:
//  - nvshmem/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh
#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"

namespace deep_ep {

EP_STATIC_ASSERT(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64, "Invalid QP minimum depth");

__device__ static __forceinline__ uint64_t HtoBE64(uint64_t x) {
    uint64_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 lo;\n\t"
        ".reg .b32 hi;\n\t"
        ".reg .b32 new_lo;\n\t"
        ".reg .b32 new_hi;\n\t"
        "mov.b64 {lo,hi}, %1;\n\t"
        "prmt.b32 new_hi, lo, ign, 0x0123;\n\t"
        "prmt.b32 new_lo, hi, ign, 0x0123;\n\t"
        "mov.b64 %0, {new_lo,new_hi};\n\t"
        "}"
        : "=l"(ret)
        : "l"(x));
    return ret;
}

__device__ static __forceinline__ uint32_t HtoBE32(uint32_t x) {
    uint32_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        "prmt.b32 %0, %1, ign, 0x0123;\n\t"
        "}"
        : "=r"(ret)
        : "r"(x));
    return ret;
}

__device__ static __forceinline__ uint16_t HtoBE16(uint16_t x) {
    // TODO: simplify PTX using 16-bit instructions
    auto a = static_cast<uint32_t>(x);
    uint32_t d;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        "mov.b32 mask, 0x4401;\n\t"
        "mov.b32 ign, 0x0;\n\t"
        "prmt.b32 %0, %1, ign, mask;\n\t"
        "}"
        : "=r"(d)
        : "r"(a));
    return static_cast<uint16_t>(d);
}

typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) ibgda_ctrl_seg_t;

typedef struct {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;

__device__ static __forceinline__ nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    /*
    这是nvshmem源码中的一行: __constant__ nvshmemi_device_host_state_t nvshmemi_device_state_d;
    就是说 nvshmemi_device_state_d 是 nvshmemi_device_host_state_t类型的一个对象，是变量名。
    __constant__: 是 CUDA 存储类限定符, 表示变量位于常量内存空间。只读，可通过 __ldg() 高效读取。
    所有线程共享，生命周期与程序相同。
    */
    return &nvshmemi_ibgda_device_state_d;
}

__device__ static __forceinline__ nvshmemi_ibgda_device_qp_t* ibgda_get_rc(int pe, int id) {
    auto state = ibgda_get_state();
    /*
    pe: 当前PE的目标PE的编号。

    num_rc_per_pe: 每个 PE（Processing Element）在每个 NIC 上分配的 RC（Reliable Connection）队列对（Queue Pair, QP）数量。
                   RC 是 InfiniBand 中的一种 QP 类型，提供可靠、有序的点对点通信。
    
    概念	                    含义	                                范围	       计算公式
    num_devices_initialized	   当前 PE 选择并成功初始化的 NIC 设备数量	    PE 级别	      n_devs_selected
    num_rc_per_pe              每个 PE 在每个 NIC 设备上的 RC QP 数量	  PE × 设备	    num_rc_handles / n_devs_selected / n_pes
    每个 PE 的总 RC 数           每个 PE 在所有NIC设备上的 RC 总数	        PE 级别	      num_devices_initialized × num_rc_per_pe
    */
    const auto num_rc_per_pe = ibgda_get_state()->num_rc_per_pe;
    /*
    globalmem: 全局内存区域。
    rcs: RC QP 数组，数组元素类型为 nvshmemi_ibgda_device_qp_t *
    globalmem.rcs 的数组大小: n_pes × num_rc_per_pe × num_devices_initialized
    n_pes 的含义: 整个集群（多机多卡）中的所有 PE 总数

    每个 PE 占用 num_rc_per_pe * num_devices_initialized 个连续的 RC QP。
    id: 即 dst_expert_local_idx 或 local_expert_idx。实际上可以是任何需要进行负载均衡通信的逻辑实体。
    id % (num_rc_per_pe * state->num_devices_initialized): rc_idx

    事实上，有断言:
    1、internode_ll.cu 中: EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);
    2、internode.cu 中:    EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms);

    返回一个负载均衡后的 RC QP 的指针，类型为 nvshmemi_ibgda_device_qp_t *
    */
    return &state->globalmem.rcs[pe * num_rc_per_pe * state->num_devices_initialized + id % (num_rc_per_pe * state->num_devices_initialized)];
}

__device__ static __forceinline__ void ibgda_lock_acquire(int* lock) {
    while (atomicCAS(lock, 0, 1) == 1)
        ;

    // Prevent reordering before the lock is acquired
    /*
    memory_fence_cta(): cta级别的内存屏障
    作用: 防止在获取锁之前的内存操作被重排序到获取锁之。即“acquire后不到前”
    保证: 获取锁之后，之前的所有内存操作对其他线程可见
    */
    memory_fence_cta();
}

__device__ static __forceinline__ void ibgda_lock_release(int* lock) {
    memory_fence_cta();

    // Prevent reordering before lock is released
    st_na_relaxed(lock, 0);
}

__device__ static __forceinline__ void ibgda_update_dbr(nvshmemi_ibgda_device_qp_t* qp, uint32_t dbrec_head) {
    /*
    dbrec_head: 就是 new_prod_idx，门铃记录的索引。
    */
    
    // `DBREC` contains the index of the next empty `WQEBB`
    __be32 dbrec_val;  // 大端序32位整数，存储要写入DBREC的值（大端序转换后的）
    /*
    tx 的全称是 Transmit（发送/传输）
    qp->tx_wq.dbrec: 门铃记录指针，指向门铃记录的内存地址。在GPU内存，映射到NIC的Doorbell区域。
    */
    __be32* dbrec_ptr = qp->tx_wq.dbrec;

    // This is equivalent to `WRITE_ONCE(dbrec_ptr, HtoBE32(dbrec_head & 0xffff))`
    /*
    下面的asm相当于: dbrec_val = HtoBE32(dbrec_head & 0xffff)
    */
    asm("{\n\t"
        ".reg .b32 dbrec_head_16b;\n\t"                     // 声明32位寄存器变量
        ".reg .b32 ign;\n\t"                                // 声明忽略寄存器
        "and.b32 dbrec_head_16b, %1, 0xffff;\n\t"           // 将dbrec_head的低16位赋值给dbrec_head_16b
        "prmt.b32 %0, dbrec_head_16b, ign, 0x123;\n\t"      // 将dbrec_head_16b的值转换为大端序
        "}"
        : "=r"(dbrec_val)                                   // 输出参数，大端序32位整数值
        : "r"(dbrec_head));
    // 将 dbrec_val 写入 dbrec_ptr 指向的内存位置
    st_na_release(dbrec_ptr, dbrec_val);
}

__device__ static __forceinline__ void ibgda_ring_db(nvshmemi_ibgda_device_qp_t* qp, uint16_t prod_idx) {
    // 门铃寄存器（BlueFlame）
    auto bf_ptr = reinterpret_cast<uint64_t*>(qp->tx_wq.bf);
    ibgda_ctrl_seg_t ctrl_seg = {.opmod_idx_opcode = HtoBE32(prod_idx << 8), .qpn_ds = HtoBE32(qp->qpn << 8)};

    EP_STATIC_ASSERT(sizeof(decltype(&ctrl_seg)) == sizeof(uint64_t), "");
    st_na_release(bf_ptr, *(reinterpret_cast<uint64_t*>(&ctrl_seg)));
}

__device__ static __forceinline__ void ibgda_post_send(nvshmemi_ibgda_device_qp_t* qp, uint64_t new_prod_idx) {
    nvshmemi_ibgda_device_qp_management_t* mvars = &qp->mvars;
    uint64_t old_prod_idx;

    // Update `prod_idx` before ringing the doorbell, so that we know which index is needed in quiet/fence
    // 
    /*
    &mvars->post_send_lock: 门铃提交锁。
    在敲响门铃之前使用atomicMax函数更新 prod_idx，这样在 quiet/fence 操作时就知道需要哪个索引。
    与下面的ibgda_lock_release形成“acquire-release”配对。这也可以:
    防止重排序: 确保锁释放之前的所有操作都已完成。 可见性保证: 确保其他线程能看到锁释放后的状态更新。
    “acquire后不到前，release前不到后”。
    */
    ibgda_lock_acquire(&mvars->post_send_lock);

    /*
    &mvars->tx_wq.ready_head: tx_qp中已准备好提交的WQE索引。
    &mvars->tx_wq.prod_idx: tx_qp中已提交给NIC的WQE的索引（生产者索引），由atomicMax函数更新。

    同步模式下，wqe完整的流转过程：预留阶段，写入阶段，准备阶段，提交阶段（批处理触发时）。而异步模式不需要提交阶段。
    qp->tx_wq.wqe 队列状态：
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ WQE 0 │ WQE 1 │ ... │ WQE 8 │ WQE 9 │ WQE 10 │     11 │     12 │     13 │    ...│
    └─────────────────────────────────────────────────────────────────────────────────┘
                              ↑                        ↑                  ↑
                          prod_idx                 ready_head         resv_head
    索引关系：
    - resv_head = 13   (下一个可预留的WQE索引，tx_wq.wqe[11, 13)虽然预留了但是还没有写入WQE )
    - ready_head = 11  (已准备好提交的WQE索引，WQE[8, 11) 已准备好)
    - prod_idx = 8     (已提交给NIC的WQE索引，WQE[0, 8) 已提交)
    注意: 区分 resv_head 和new_wqe_idx。
        resv_head: 全局状态，表示“下一个可预留的WQE索引”
        new_wqe_idx: 局部变量，表示“本次操作（warp级的）使用的最后一个WQE的下一个索引”
        在并发场景下，它们可能不相等（resv_head 可能已被其他warp更新）。
    atomicMax(addr, val): 读取addr指向的值，如果 val > 当前值，则写入val并返回旧值，如果 val <= 当前值，则不写入，返回当前值。
    原子性: 保证多线程并发时的正确性。
    单调递增: prod_idx 只能单调递增，不能回退。
    避免覆盖: 如果多个线程同时更新，atomicMax 确保取最大值。
    */
    old_prod_idx = atomicMax(reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.prod_idx), new_prod_idx);
    /*
    new_prod_idx 是当前warp的最后一个WQE的下一个索引。而&mvars->tx_wq.prod_idx是所有warp共同维护的生产者索引。
    当当前warp的生产者索引大于tx_qp中已提交给NIC的WQE的索引时，就需要触发门铃。
    如果 new_prod_idx <= old_prod_idx，说明已经有其他线程更新了更大的索引，不需要重复触发门铃。
    加条件判断是为了避免重复触发、减少不必要的门铃操作，这样可以提高性能。
    */
    if (new_prod_idx > old_prod_idx) {
        // dbr: 门铃记录（DBR，doorbell record）缓冲区，同样位于 GPU 内存中。
        ibgda_update_dbr(qp, new_prod_idx);
        ibgda_ring_db(qp, new_prod_idx);
    }
    ibgda_lock_release(&mvars->post_send_lock);
}

template <bool kAlwaysDoPostSend>
__device__ static __forceinline__ void ibgda_submit_requests(nvshmemi_ibgda_device_qp_t* qp,
                                                             uint64_t base_wqe_idx,
                                                             uint32_t num_wqes,
                                                             int message_idx = 0) {
    auto state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_management_t* mvars = &qp->mvars;
    // new_wqe_idx: 计算得到的新WQE索引
    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

    // WQE writes must be finished first
    /* 
    确保当前线程的所有内存写入操作对其他线程和设备（如NIC）可见。
    GPU可能有**写缓冲区（Write Buffer）**来暂存写入操作
    __threadfence() 强制刷新当前线程的写缓冲区，它会阻塞当前线程，直到所有待写入的数据都真正写入到全局内存，且对其他线程和设备可见。
    __threadfence() 建立了happens-before关系，屏障之前的所有内存写入，在屏障之后对其他线程和设备可见。
    在此之后，其他线程/NIC才能看到完整的WQE数据。
    */
    __threadfence();

    /*
    state->use_async_postsend: 是否使用异步提交模式。
    QP结构体中的 tx_wq:
    typedef struct nvshmemi_ibgda_device_qp {
        struct {
            uint16_t nwqes;                    // WQE队列大小
            void *wqe;                         // WQE队列内存基地址
            __be32 *dbrec;                     // Doorbell Record指针
            void *bf;                          // 入门铃寄存器（BlueFlame）指针
            nvshmemi_ibgda_device_cq_t *cq;    // Completion Queue指针
            uint64_t *prod_idx;                // ← 这是一个指针！
            // 注释: "May point to mvars.prod_idx or internal prod_idx"
        } tx_wq;
        
        nvshmemi_ibgda_device_qp_management_v1 mvars;  // 管理变量
    } nvshmemi_ibgda_device_qp_t;

    管理变量中的 tx_wq: 
    typedef struct {
        struct {
            uint64_t resv_head;    // 已预留的WQE索引, ibgda_reserve_wqe_slots 函数中设置。
            uint64_t prod_idx;     // ← 这是一个值！
            uint64_t ready_head;   // ← 这是一个值！
            uint64_t get_head;     // 最后一个"fetch"操作的WQE索引
            uint64_t get_tail;     // 最后一个被轮询的WQE索引
        } tx_wq;
        // ... 其他字段
    } nvshmemi_ibgda_device_qp_management_v1;
    */
    /*
    qp->tx_wq.prod_idx: NIC已经处理到哪个WQE了。跟踪已提交给NIC的WQE索引。在异步模式下使用: NIC自动轮询，更新这个索引。
    &mvars->tx_wq.ready_head: GPU已经准备好提交到哪个WQE了。跟踪已准备好提交的WQE索引。
                              在同步模式下使用: GPU显式控制提交，使用这个索引跟踪准备好的WQE。
    ready_idx 不指向WQE本身，而是指向存储WQE索引的变量。

    情况1：异步提交模式 (use_async_postsend == true)，选择: qp->tx_wq.prod_idx
        含义: 指向已提交给NIC的WQE索引（生产者索引）
        位置: 指向 qp->tx_wq.prod_idx
        作用: NIC自动轮询，使用prod_idx跟踪已处理的WQE。NIC硬件自动轮询WQE队列，检查新的WQE。
        无需门铃: GPU不需要显式触发门铃，NIC会自动发现新的WQE。
        使用 prod_idx: 通过 qp->tx_wq.prod_idx 跟踪已提交给NIC的WQE索引
        优势：更高吞吐量: NIC可以批量处理多个WQE，减少中断开销。适合高吞吐量的场景（如训练阶段）。
             减少GPU开销: GPU不需要显式触发门铃，减少GPU线程的开销，减少门铃寄存器的写入次数。
             更好的批处理: NIC可以自动批量处理WQE，提高效率。
        劣势：更高延迟: NIC轮询有延迟，不能立即处理WQE。不适合低延迟的场景（如推理阶段）。
             CPU辅助: 可能需要CPU辅助来管理异步提交，增加了系统复杂性。

    情况2：同步提交模式 (use_async_postsend == false)，选择: &mvars->tx_wq.ready_head
        含义: 指向已准备好提交的WQE索引
        位置: 管理变量中的 ready_head 字段
        作用: GPU显式控制提交，使用ready_head跟踪已准备好提交的WQE索引。但是这些WQE还没有真正提交。
        需要门铃: GPU显式调用 ibgda_post_send 触发门铃（GPU提交WQE到NIC），通过写入门铃寄存器通知NIC处理WQE
        支持批处理: 可以批量提交多个WQE，减少门铃触发次数
        优势：更低延迟: GPU可以立即触发NIC处理WQE，适合低延迟的场景（如推理阶段）。
             更好的控制: GPU可以精确控制何时提交WQE，支持批处理策略，减少门铃开销。
             无需CPU辅助: 完全由GPU控制，不需要CPU辅助。
        劣势：需要批处理: 如果不使用批处理，门铃开销较大，需要合理设置批处理大小。
             GPU开销: GPU需要显式触发门铃，增加GPU线程的开销。

    state->use_async_postsend 在NVSHMEM初始化时设置: use_async_postsend 是IBGDA全局状态的一部分。
    通过环境变量或配置: 通常在NVSHMEM库初始化时根据配置设置。运行时不可变: 一旦设置，在运行时不会改变。
    典型使用场景：
        训练阶段: 通常使用异步模式（use_async_postsend = true），追求高吞吐量。
        推理阶段: 通常使用同步模式（use_async_postsend = false），追求低延迟。
    
    注意: ready_idx的具体值，需要程序员自己设置，不是rdma库自动设置的。
    */
    unsigned long long int* ready_idx =
        (unsigned long long int*)(state->use_async_postsend ? qp->tx_wq.prod_idx : &mvars->tx_wq.ready_head);

    // Wait for prior WQE slots to be filled first
    /*
    函数签名: atomicCAS(address, compare, val)
    读取 address 指向的值，如果值等于 compare，则写入 val 并返回旧值。如果值不等于 compare，则不写入，返回当前值。返回值: 操作前的旧值。
    在while循环之后，其他线程/NIC才知道有新的WQE需要处理。
    */
    while (atomicCAS(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx);

    // Always post, not in batch
    // 同步模式中，ready_idx指向的是已准备好提交的 WQE 索引，但是并没有真正提交，需要手动提交。
    // 异步模式下NIC自动轮询，不需要显式触发门铃。
    if (!state->use_async_postsend) {
        //  每4个消息提交一次（批处理），减少门铃触发次数，提高性能。
        constexpr int kNumRequestInBatch = 4;
        /*
        kAlwaysDoPostSend: 是否总是立即提交，不进行批处理。如果是false，则根据批处理策略决定是否提交。
        用途: 用于需要立即提交的场景（如最后一个消息）。
        message_idx + 1: 加1是为了处理“索引从0开始”的情况，每4个连续的 message_idx 为一组消息，一起提交。
        每次提交，具体提交多少个wqe得看相同的message_idx中有多少个wqe。对于非关键路径，批处理可以隐藏延迟
        */
        if (kAlwaysDoPostSend or (message_idx + 1) % kNumRequestInBatch == 0)
            ibgda_post_send(qp, new_wqe_idx);
    }
}

__device__ static __forceinline__ void ibgda_write_rdma_write_inl_wqe(
    nvshmemi_ibgda_device_qp_t* qp, const uint32_t* val, uint64_t raddr, __be32 rkey, uint16_t wqe_idx, void** out_wqes, uint32_t imm) {
    ibgda_ctrl_seg_t ctrl_seg;
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_inl_data_seg inl_seg;

    auto* ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    auto* raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    auto* inl_seg_ptr = reinterpret_cast<mlx5_wqe_inl_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
    auto* wqe_data_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(inl_seg_ptr) + sizeof(*inl_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    inl_seg.byte_count = HtoBE32(4 | MLX5_INLINE_SEG);

    // `imm == std::numeric_limits<uint32_t>::max()` means no imm writes
    ctrl_seg = {0};
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode =
        HtoBE32((wqe_idx << 8) | (imm != std::numeric_limits<uint32_t>::max() ? MLX5_OPCODE_RDMA_WRITE_IMM : MLX5_OPCODE_RDMA_WRITE));
    if (imm != std::numeric_limits<uint32_t>::max())
        ctrl_seg.imm = HtoBE32(imm);

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*inl_seg_ptr) == 4, "sizeof(*inl_seg_ptr) == 4");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<uint32_t*>(inl_seg_ptr), *reinterpret_cast<const uint32_t*>(&inl_seg));
    st_na_relaxed(reinterpret_cast<uint32_t*>(wqe_data_ptr), *reinterpret_cast<const uint32_t*>(val));
}

__device__ static __forceinline__ uint64_t
ibgda_get_lkey_and_rkey(uint64_t laddr, __be32* lkey, uint64_t raddr, int dst_pe, uint64_t* out_raddr, __be32* out_rkey, uint32_t dev_idx) {
    /*
    输入:
        laddr: 本地地址（会被更新为chunk内地址）
        raddr: 远程地址（原始地址）
        dst_pe: 目标PE
        dev_idx: NIC设备索引号（qp->dev_idx）
            标识当前QP使用的NIC设备
            用于查找该NIC设备对应的内存注册密钥
            因为每个NIC设备都有独立的内存注册（memory registration），需要不同的lkey/rkey
    输出:
        lkey: 本地chunk的访问密钥
        out_raddr: 转换后的远程地址
        out_rkey: 本地存储的远程PE的chunk raddr在NIC dev_idx中的rkey
    返回值: 当前chunk可传输的最大字节数（取本地和远程chunk剩余大小的最小值）
    */
    auto state = ibgda_get_state();
    // 获取当前PE的对称堆（symmetric heap）基地址。用于计算地址相对于堆基址的偏移量，从而定位到对应的chunk
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
    /* 
    获取CUDA统一内存粒度（以2为底的对数），一般为21（表示2^21 = 2MB）
    用途: 用于将地址偏移量右移，得到chunk索引
    含义: 内存按2^log2_cumem_granularity字节对齐划分为chunk。granularity: 粒度。
    */
    auto log2_cumem_granularity = state->log2_cumem_granularity;

    // Local key
    /* 
    本地key索引计算
    laddr - heap_start: 计算相对于堆基址的偏移量
    >> log2_cumem_granularity: 将地址偏移量右移，得到chunk在PE中的索引
    注意:chunk在GPU内存中：chunk是GPU内存的物理划分
        每个NIC需要独立注册：同一个GPU内存chunk，需要在每个NIC设备上独立注册，获得不同的lkey/rkey
        这是因为: 
            不同NIC有不同的PCIe地址映射;
            不同NIC有不同的访问权限和上下文;
            每个NIC需要自己的内存访问密钥来验证和授权。
    所以，每个PE的lkeys数组需要为每个chunk的每个NIC都存储一个密钥。
    * state->num_devices_initialized: 乘以设备数量，得到本地key索引
    + dev_idx: 加上设备索引，得到本地key索引
    */
    uint64_t idx = ((laddr - heap_start) >> log2_cumem_granularity) * state->num_devices_initialized + dev_idx;
    /*
    从常量内存中读取对应laddr所在chunk的device_key结构。
    device_key结构: 包含key（lkey值）和next_addr（下一个chunk的起始地址）
    constmem.lkeys: 常量内存中的本地密钥数组，存储所有chunk的lkey信息
    */ 
    auto device_key = state->constmem.lkeys[idx];
    /*
    计算当前本地chunk的剩余大小（从laddr到chunk边界的字节数）。
    device_key.next_addr: 下一个chunk的起始地址，即当前chunk的结束地址
    lchunk_size: 当前chunk从laddr开始还能传输多少字节
    用途: 用于限制单次RDMA操作的最大传输大小，不能越界
    */
    auto lchunk_size = device_key.next_addr - laddr;
    *lkey = device_key.key;  // 将获取到的lkey写入输出参数

    // Remote key
    uint64_t roffset = raddr - heap_start;

    /*
    远程key索引计算
    注意: rkey需要PE维度: 不同PE的相同chunk索引对应不同的物理内存，需要不同的rkey。
    单个chunk的rkey的全部大小是: npes * num_devices_initialized
    */
    idx = ((roffset >> log2_cumem_granularity) * nvshmemi_device_state_d.npes) * state->num_devices_initialized +
        dst_pe * state->num_devices_initialized + dev_idx;
    /*
    这里居然还搞这么一出。😂
    NVSHMEMI_IBGDA_MAX_CONST_RKEYS: 常量内存中rkeys数组的最大容量。常量内存: 容量有限但访问速度快，用于存储常用的rkey
    NVSHMEMI_IBGDA_MAX_CONST_RKEYS 存储不下的就存在全局内存中。
    */
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {
        device_key = state->constmem.rkeys[idx];
    } else {
        // globalmem.rkeys 是可以存储所有的rkey的
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    }
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;  // 本地存储的远程PE的chunk raddr在NIC dev_idx中的rkey

    // Return the minimum of local and remote chunk sizes
    auto rchunk_size = device_key.next_addr - roffset;  // 远程PE的这个chunk从raddr开始还能传输多少字节
    return min(lchunk_size, rchunk_size);  // 返回较小的那个，让两边都能不跨chunk传输。
}

__device__ static __forceinline__ void ibgda_get_rkey(uint64_t addr, int dst_pe, uint64_t* out_raddr, __be32* out_rkey, uint32_t dev_idx) {
    auto state = ibgda_get_state();
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);

    uint64_t roffset = addr - heap_start;
    uint64_t idx = ((roffset >> state->log2_cumem_granularity) * nvshmemi_device_state_d.npes * state->num_devices_initialized) +
        dst_pe * state->num_devices_initialized + dev_idx;
    nvshmemi_ibgda_device_key_t device_key;
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS)
        device_key = state->constmem.rkeys[idx];
    else
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;
}

__device__ static __forceinline__ uint64_t ibgda_reserve_wqe_slots(nvshmemi_ibgda_device_qp_t* qp, uint32_t num_wqes) {
    /*
    qp:        每个qp包含: 发送队列（TX WQ）、完成队列（CQ）、管理变量（mvars）等。
    um_wqes:   需要预留的WQE数量，即本次操作需要多少个WQE槽位。
    qp->mvars: qp的管理变量（management variables）。
               存储位置: GPU全局内存（每个QP有独立的管理变量）。作用: 存储QP的状态信息
    mvars 的类型是 nvshmemi_ibgda_device_qp_management_t*
    */
    auto mvars = &qp->mvars;
    /*
    作用: 原子地预留num_wqes个连续的WQE槽位
    resv_head: 发送队列中下一个可用的WQE索引（生产者指针）
    返回值: 预留的WQE索引范围的起始位置
    下面的 tx_wq 也是 mvars 里的一个字段。
    struct {
        uint64_t resv_head;    // 发送队列中下一个可用的WQE索引（生产者指针）
        uint64_t prod_idx;     // 已提交给NIC的WQE索引（门铃指针）
        uint64_t ready_head;   // 已准备好提交的WQE索引
        uint64_t get_head;     // 最后一个"fetch"操作的WQE索引
        uint64_t get_tail;     // 最后一个被轮询的WQE索引
    } tx_wq;

    mvars->tx_wq.resv_head: resv_head表示下一个可预留的WQE索引, 也表示已预留的WQE数量。
    atomicXXX 都是返回旧值。

    指针取字段用 ->
    对象或引用取字段用 .
    */
    return atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.resv_head), 
                     static_cast<unsigned long long>(num_wqes));
}

__device__ static __forceinline__ void* ibgda_get_wqe_ptr(nvshmemi_ibgda_device_qp_t* qp, uint16_t wqe_idx) {
    /*
    作用: 根据WQE索引计算WQE在发送队列中的实际内存地址

    返回值: WQE内存地址指针

    注意: 区分下面两种 tx_wq 字段：
    1、qp->tx_wq: 配置和资源（nwqes, wqe, dbrec 等），运行时不变的静态字段。
    2、&qp->mvars->tx_wq: 运行时状态（resv_head, prod_idx 等），运行时变化的动态字段。

    nwqes: Number of WQEs，即WQE队列的大小（容量）。nwqes 必须是2的幂次方（如 256, 512, 1024, 2048 等），
    这样可以用于后续的位掩码取模操作，2的幂次方可以高效地使用位运算代替除法。
    */
    uint16_t cnt = qp->tx_wq.nwqes;
    /*
    其实很简单。当cnt是2的幂次方时，cnt - 1的二进制表示是全1。
    当 wqe_idx 比cnt小时，wqe_idx & (cnt - 1) 的结果就是 wqe_idx 本身，也就是余数部分。
    当 wqe_idx 比cnt大时，wqe_idx 超出cnt的范围高位二进制对应的(cnt - 1) 是0，这部分与0求与运算，结果肯定是0。
        就只剩下比cnt小的部分与(cnt - 1)求与运算，结果就是比cnt小的部分，也就是余数部分。
    */
    uint16_t idx = wqe_idx & (cnt - 1);  // 环形缓冲区取模
    /*
    qp->tx_wq.wqe: WQE队列的内存基地址（void*类型）
    MLX5_SEND_WQE_SHIFT: WQE大小对齐的位移量。实际WQE对齐的大小是(1 << MLX5_SEND_WQE_SHIFT)字节，也就是(2^MLX5_SEND_WQE_SHIFT)字节。
    通常值为 6（因为MLX5硬件要求WQE必须对齐到64字节边界），如果WQE大小是128字节，则 MLX5_SEND_WQE_SHIFT = 7（2^7 = 128）
    idx << MLX5_SEND_WQE_SHIFT: 相当于idx * (2^MLX5_SEND_WQE_SHIFT)，也就是idx * 64，得到idx对应的WQE在WQE队列中的偏移量。
    */
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(qp->tx_wq.wqe) + (idx << MLX5_SEND_WQE_SHIFT));
}

__device__ static __forceinline__ void nvshmemi_ibgda_rma_p(
    int* rptr, const int value, int dst_pe, int qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max()) {
    // Get rkey
    // NOTES: the `p` operation will not cross multiple remote chunks
    __be32 rkey;
    uint64_t raddr;
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), dst_pe, &raddr, &rkey, qp->dev_idx);

    // Write WQEs
    uint64_t base_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
    void* wqe_ptrs;
    wqe_ptrs = ibgda_get_wqe_ptr(qp, base_wqe_idx);
    ibgda_write_rdma_write_inl_wqe(qp, reinterpret_cast<const uint32_t*>(&value), raddr, rkey, base_wqe_idx, &wqe_ptrs, imm);

    // Submit requests
    ibgda_submit_requests<true>(qp, base_wqe_idx, 1);
}

__device__ static __forceinline__ void ibgda_write_rdma_write_wqe(nvshmemi_ibgda_device_qp_t* qp,
                                                                  uint64_t laddr,
                                                                  __be32 lkey,
                                                                  uint64_t raddr,
                                                                  __be32 rkey,
                                                                  uint32_t bytes,
                                                                  uint16_t wqe_idx,
                                                                  void** out_wqes) {
    /*
    qp,         // Queue Pair指针
    laddr,      // 本地内存地址（当前chunk）
    lkey,       // 本地内存键（大端序32位）
    raddr,      // 远程内存地址（当前chunk）
    rkey,       // 远程内存键（大端序32位）
    bytes,      // 传输字节数
    wqe_idx,    // WQE索引
    out_wqes    // WQE内存地址指针
    */
    /*
    ctrl_seg: Control Segment（控制段）
    类型: ibgda_ctrl_seg_t，实际上是 struct mlx5_wqe_ctrl_seg 的别名
    大小: 16字节（sizeof(int4)）
    作用: 包含WQE的控制信息（操作码、QP编号、数据段数量等）
    对齐: __attribute__((__aligned__(8)))，8字节对齐
    */
    ibgda_ctrl_seg_t ctrl_seg;
    /*
    raddr_seg: Remote Address Segment（远程地址段）
    类型: struct mlx5_wqe_raddr_seg
    大小: 16字节（sizeof(int4)）
    作用: 包含远程内存地址和rkey
    字段: raddr（64位地址）、rkey（32位密钥）、reserved（32位保留字段）
    */
    struct mlx5_wqe_raddr_seg raddr_seg;
    /*
    data_seg: Data Segment（数据段）
    类型: struct mlx5_wqe_data_seg
    大小: 16字节（sizeof(int4)）
    作用: 包含本地内存地址、lkey和传输字节数
    字段: byte_count（32位字节数）、lkey（32位密钥）、addr（64位地址）
    */
    struct mlx5_wqe_data_seg data_seg;

    /*
    以上三个段是用来声明WQE段结构体。每个段都是16字节（int4的大小），这是MLX5硬件的要求。
    WQE内存布局（64字节对齐）：
    ┌───────────────────────────────────┐
    │ Control Segment (16字节)           │ ← ctrl_seg_ptr
    │ Remote Address Segment (16字节)    │
    │ Data Segment (16字节)              │
    │ Reserved/Padding (16字节)          │ ← 对齐到64字节
    └───────────────────────────────────┘
    */

    // 指向WQE内存中Control Segment的位置，这是WQE的第一个段，位于WQE的起始位置（偏移0）。
    auto* ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    /*
    av_seg_ptr: Address Vector Segment指针
    注意: 虽然变量名是 av_seg_ptr，但实际上在RDMA Write操作中，这个位置是Remote Address Segment。
    这可能是一个历史遗留的命名（在某些操作中可能是Address Vector）

    下面的5行代码其实两行就能搞定:
    struct mlx5_wqe_raddr_seg* raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_data_seg* data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
    */
    void* av_seg_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    struct mlx5_wqe_raddr_seg* raddr_seg_ptr;
    struct mlx5_wqe_data_seg* data_seg_ptr;

    raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(av_seg_ptr));
    data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));

    /*
    HtoBE64: Host to Big-Endian 64位转换函数
    功能: 将64位整数从主机字节序转换为大端序（Big-Endian）
    原因: InfiniBand网络协议使用大端序，必须转换
    实现: 使用PTX汇编指令 prmt 进行字节序转换。
    Remote Address Segment结构:
        struct mlx5_wqe_raddr_seg {
            uint64_t raddr;    // 远程内存地址（64位，大端序）
            __be32 rkey;       // 远程内存键（32位，大端序）
            uint32_t reserved; // 保留字段（32位，必须为0）
        };  // 总共16字节
    */
    raddr_seg.raddr = HtoBE64(raddr);  // 8字节，64位
    raddr_seg.rkey = rkey;             // 4字节，大端序32位（Remote Address Segment的远程密钥字段）
    raddr_seg.reserved = 0;            // 4字节，保留字段。填充结构体，确保内存布局正确

    /*
    data_seg.byte_count: 告诉NIC要传输多少字节的数据
    data_seg.lkey:       本地内存键
    data_seg.addr:       本地内存地址。告诉NIC源数据在本地内存中的位置
    */
    data_seg.byte_count = HtoBE32(bytes);  // 4字节，大端序32位
    data_seg.lkey = lkey;                  // 4字节，大端序32位
    data_seg.addr = HtoBE64(laddr);        // 8字节，64位

    /*
    ibgda_ctrl_seg_t 的定义来自于nvshmem，而mlx5_wqe_ctrl_seg来自于:
    https://github.com/linux-rdma/rdma-core/blob/master/providers/mlx5/mlx5dv.h; 
    struct mlx5_wqe_ctrl_seg {
        __be32		opmod_idx_opcode;
        __be32		qpn_ds;
        uint8_t		signature;
        __be16		dci_stream_channel_id;
        uint8_t		fm_ce_se;
        __be32		imm;
    } __attribute__((__packed__)) __attribute__((__aligned__(4)));
    typedef struct mlx5_wqe_ctrl_seg __attribute__((__aligned__(8))) ibgda_ctrl_seg_t;
    */
    /*
    C++11的初始化列表语法。将结构体的所有字段初始化为0。等价于: memset(&ctrl_seg, 0, sizeof(ctrl_seg))
    原因: 确保所有字段都有明确的初始值，避免未初始化的内存。
    */
    ctrl_seg = {0};  // 将结构体初始化为全0
    /*
    qp->qpn: Queue Pair Number，QP的唯一标识符（24位）。用于告诉NIC这个WQE属于哪个QP。
    qp->qpn << 8: 左移8位。QP编号存储在字段的高24位（bit 8-31）。
    | 3: 按位或操作，告诉NIC这个WQE包含多少个段，即设置Data Segments。表示有3个数据段，存储在字段的低8位（bit 0-7）。
    */
    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 3);
    /*
    MLX5_WQE_CTRL_CQ_UPDATE: 完成队列更新标志常量。指示NIC在处理完这个WQE后，更新Completion Queue（完成队列）。
                             值通常是 0x1（具体值取决于MLX5驱动定义）。
    ctrl_seg.fm_ce_se: Control Segment的标志字段。控制WQE完成后的行为。fm: Flow Meter（流计量器）标志。
                       ce: Completion Event（完成事件）标志。se: Solicited Event（请求事件）标志。
    为什么需要这个标志: 完成通知: 当WQE处理完成后，NIC会在Completion Queue中写入一个完成条目
                     轮询机制: GPU可以通过轮询Completion Queue来检查WQE是否完成
    */
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    /*
    ctrl_seg.opmod_idx_opcode: Control Segment的操作码和索引字段（32位，大端序）。
    wqe_idx: WQE索引（输入参数）。uint16_t（16位无符号整数）。
    MLX5_OPCODE_RDMA_WRITE: RDMA写操作码常量。通常是 0x05（具体值取决于MLX5驱动定义），指示这是一个RDMA Write操作。
    opmod_idx_opcode字段（32位）：
    ┌───────────────────────────────────────────────┐
    │ Op Mod (8位) │ WQE Index (16位) │ Opcode (8位) │
    │   bit 24-31  │    bit 8-23     │   bit 0-7    │
    │      0       │   wqe_idx << 8  │ RDMA_WRITE   │
    └───────────────────────────────────────────────┘
    */
    ctrl_seg.opmod_idx_opcode = HtoBE32((wqe_idx << 8) | MLX5_OPCODE_RDMA_WRITE);

    // 3个段都是16字节
    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");
    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == 16, "sizeof(*data_seg_ptr) == 16");
    /*
    为什么需要 st_na_relaxed 而不是普通写入：
        st_na_relaxed: Non-Aligned, Relaxed memory ordering。
        绕过L1缓存: 直接写入全局内存，不经过L1缓存。
        立即可见: 确保NIC能立即看到写入的数据。
        普通写入: 可能被缓存在L1中，NIC无法看到。
    */
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
}

__device__ static __forceinline__ void ibgda_write_empty_recv_wqe(void* out_wqe) {
    auto* data_seg_ptr = reinterpret_cast<struct mlx5_wqe_data_seg*>(out_wqe);
    struct mlx5_wqe_data_seg data_seg;

    // Make the first segment in the WQE invalid, then the entire list will be invalid
    data_seg.byte_count = 0;
    data_seg.lkey = HtoBE64(MLX5_INVALID_LKEY);
    data_seg.addr = 0;

    EP_STATIC_ASSERT(sizeof(mlx5_wqe_data_seg) == sizeof(int4), "Invalid data type length");
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
}

template <bool kAlwaysDoPostSend = false>
__device__ static __forceinline__ void nvshmemi_ibgda_put_nbi_warp(
    uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe, int qp_id, int lane_id, int message_idx) {
    /*
    uint64_t req_rptr,      远程目标地址（64位）
    uint64_t req_lptr,      本地源地址（64位）
    size_t bytes,           要传输的字节数
    int dst_pe,             目标PE（Processing Element）ID，即目标rank。注意: 这个PE可以是GPU，也可以是节点
    int qp_id,              Queue Pair ID，在internode_ll.cu中是expert_local_idx
    int lane_id,            warp内的线程ID（0-31）
    int message_idx         消息索引，用于批处理控制
    */
    
    // Get lkey and rkey, store them into lanes
    uint32_t num_wqes = 0;          // WQE数量计数器，初始化为0。记录需要多少个WQE来完成这次传输。如果数据跨越多个内存chunk，可能需要多个WQE
    __be32 my_lkey = 0;             // 本地内存键（Local Key），访问密钥，大端序32位。用于RDMA操作验证本地内存访问权限
    uint64_t my_laddr = 0;          // 本地内存的实际地址（可能因chunk边界而调整）
    __be32 my_rkey = 0;             // 远程内存键（Remote Key），访问密钥，大端序32位。用于RDMA操作验证远程内存访问权限
    uint64_t my_raddr = 0;          // 远程内存的实际地址（转换为NIC可访问的地址）
    uint64_t my_chunk_size = 0;     // 当前chunk可传输的最大字节数（受本地和远程chunk大小限制）

    /*
    获取指向目标PE的RC（Reliable Connection）Queue Pair的指针。
    在 internode_ll.cu 中，qp_id 是 expert_local_idx。不同专家使用不同的QP，实现通信负载在多条QP上分散，提高并行度。
    在 internode.cu 中，qp_id 是 channel_id 或 (channel_id + num_channels);
    qp 的类型是: nvshmemi_ibgda_device_qp_t *
    */
    auto qp = ibgda_get_rc(dst_pe, qp_id);

    // Decide how many messages (theoretically 3 for maximum)
    auto remaining_bytes = bytes;    // 剩余待传输字节数，初始化为总字节数
    while (remaining_bytes > 0) {    // 循环直到所有字节都分配完
        /* 
        只有特定线程负责获取当前 WQE 的密钥信息，并在后面通过 __shfl_sync 将获取的信息广播给warp内所有线程。
        每个线程都需要知道自己需要处理的chunk信息。

        注意: 每个线程在循环的不同迭代中获取自己负责的chunk的lkey/rkey信息。
        */
        if (lane_id == num_wqes) {
            // 获取lkey和rkey，并计算当前chunk的最大传输大小，也就是当前chunk的剩余大小，不能超过chunk边界。
            /*
            每个WQE只能使用一对lkey/rkey。如果需要传输的数据跨越多个chunk，需要使用多个WQE。
            WQE结构限制: 每个WQE的Data Segment只能包含：
                一个lkey（本地内存密钥）
                一个rkey（远程内存密钥）
                一个连续的地址范围
            密钥的唯一性: 每个chunk有自己独立的lkey/rkey，不能混用。
            硬件要求: RDMA硬件要求一次操作必须在同一个已注册的内存区域内。
            */
            my_chunk_size = min(remaining_bytes,
                // 返回值: 当前chunk可传输的最大字节数（取本地和远程chunk剩余大小的最小值）
                /* 每个线程在循环的不同迭代中获取自己负责的chunk的lkey/rkey信息。
                QP是点对点连接：
                每个QP绑定到特定的NIC设备（通过qp->dev_idx）
                QP建立的是：本地NIC dev_idx ↔ 远程NIC dev_idx的点对点连接
                通信必须使用对应的NIC设备对。这样可以保证通信一致性，避免NIC设备错乱，也使得密钥匹配更方便。
                */
                ibgda_get_lkey_and_rkey(
                    my_laddr = req_lptr,    // 输入：本地地址。req_lptr: 用在循环中，帮助计算每个线程负责的数据的本地地址。
                    &my_lkey,               // 输出：本地内存键。
                    req_rptr,               // 输入：用在循环中，帮助计算每个线程负责的数据的远程地址。
                    dst_pe,                 // 输入：目标PE
                    &my_raddr,              // 输出：当前线程要实际写入到远程PE的内存的地址
                    &my_rkey,               // 输出：远程内存键
                    qp->dev_idx             // 输入：设备索引。指的是当前PE和远程PE的 NIC 设置索引。每个 NIC 设备都有唯一的索引，用于标识它在 PCIe 总线上的位置。
                )
            );
        }

        // Move one more message
        // 每个线程都需要知道自己需要处理的chunk信息。
        auto chunk_size = __shfl_sync(0xffffffff, my_chunk_size, static_cast<int>(num_wqes));
        remaining_bytes -= chunk_size;  // 减去已分配的chunk大小
        // 每次循环后，req_lptr和req_rptr都向前移动chunk_size字节，确保下一个WQE处理下一个chunk的数据。
        req_lptr += chunk_size;         // 更新本地地址指针
        req_rptr += chunk_size;         // 更新远程地址指针
        /*
        如果离开while循环后num_wqes > 1，则必然有跨chunk的情况。
        这是因为 min 要么选择remaining_bytes要么选择当前chunk可传输的最大字节数。
        如果选择remaining_bytes，那么remaining_bytes -= chunk_size就会让remaining_bytes为0，循环完成，num_wqes是1；
        如果选择当前chunk可传输的最大字节数，那么remaining_bytes -= chunk_size执行后 remaining_bytes>0，循环继续，num_wqes就会大于1。
        */
        ++num_wqes;                     // WQE计数器加1
    }
    EP_DEVICE_ASSERT(num_wqes <= 32);   // 最多32个WQE（warp大小限制）

    // Process WQE
    uint64_t base_wqe_idx = 0;  // 基础WQE索引
    if (lane_id == 0)
        // 原子地预留num_wqes个连续的WQE槽位。用于后续的WQE分配。
        // base_wqe_idx: 预留的WQE索引范围的起始位置。
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes);
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);
    if (lane_id < num_wqes) {
        /*
        每个线程根据 lane_id 计算自己负责的WQE索引。每个线程负责写入一个 WQE。
        一个 WQE 处理一个 chunk 的数据。
        wqe_idx: 当前线程负责的WQE索引。这个索引是指qp的tx_wq中的WQE索引。
        */
        auto wqe_idx = base_wqe_idx + lane_id;
        /*
        根据WQE索引计算WQE在发送队列中的实际内存地址。
        qp->tx_wq.wqe: WQE发送队列的内存基地址（void*类型）
        qp->tx_wq.nwqes: WQE发送队列的大小（数量）
        然后环形缓冲区取模得到。
        */
        auto wqe_ptr = ibgda_get_wqe_ptr(qp, wqe_idx);
        /*
        qp,              // Queue Pair指针
        my_laddr,        // 本地内存地址（当前chunk）
        my_lkey,         // 本地内存键
        my_raddr,        // 远程内存地址（当前chunk）
        my_rkey,         // 远程内存键
        my_chunk_size,   // 传输字节数
        wqe_idx,         // WQE索引
        &wqe_ptr         // WQE内存地址指针
        */
        ibgda_write_rdma_write_wqe(qp, my_laddr, my_lkey, my_raddr, my_rkey, my_chunk_size, wqe_idx, &wqe_ptr);
    }
    __syncwarp();  // 等待所有线程完成 WQE 写入本地 SQ（Send Queue），和运行成功后写入 CQ 的回调。

    // Submit
    if (lane_id == 0)
        ibgda_submit_requests<kAlwaysDoPostSend>(qp, base_wqe_idx, num_wqes, message_idx);
    __syncwarp();
}

__device__ static __forceinline__ void ibgda_write_amo_add_wqe(nvshmemi_ibgda_device_qp_t* qp,
                                                               const int& value,
                                                               uint64_t laddr,
                                                               __be32 lkey,
                                                               uint64_t raddr,
                                                               __be32 rkey,
                                                               uint16_t wqe_idx,
                                                               void** out_wqes) {
    ibgda_ctrl_seg_t ctrl_seg = {0};
    struct mlx5_wqe_raddr_seg raddr_seg;
    struct mlx5_wqe_atomic_seg atomic_seg_1;
    struct mlx5_wqe_data_seg data_seg;

    auto ctrl_seg_ptr = reinterpret_cast<ibgda_ctrl_seg_t*>(out_wqes[0]);
    auto raddr_seg_ptr = reinterpret_cast<mlx5_wqe_raddr_seg*>(reinterpret_cast<uintptr_t>(ctrl_seg_ptr) + sizeof(*ctrl_seg_ptr));
    auto atomic_seg_ptr = reinterpret_cast<mlx5_wqe_atomic_seg*>(reinterpret_cast<uintptr_t>(raddr_seg_ptr) + sizeof(*raddr_seg_ptr));
    auto data_seg_ptr = reinterpret_cast<mlx5_wqe_data_seg*>(reinterpret_cast<uintptr_t>(atomic_seg_ptr) + sizeof(*atomic_seg_ptr));

    raddr_seg.raddr = HtoBE64(raddr);
    raddr_seg.rkey = rkey;
    raddr_seg.reserved = 0;

    // NOTES: `0x08000000` means `IBGDA_4_BYTE_EXT_AMO_OPMOD`
    ctrl_seg.opmod_idx_opcode = HtoBE32(MLX5_OPCODE_ATOMIC_MASKED_FA | (wqe_idx << 8) | 0x08000000);
    auto atomic_32_masked_fa_seg = reinterpret_cast<ibgda_atomic_32_masked_fa_seg_t*>(&atomic_seg_1);
    atomic_32_masked_fa_seg->add_data = HtoBE32(value);
    atomic_32_masked_fa_seg->field_boundary = 0;

    ctrl_seg.qpn_ds = HtoBE32((qp->qpn << 8) | 4);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

    data_seg.byte_count = HtoBE32(sizeof(int));
    data_seg.lkey = lkey;
    data_seg.addr = HtoBE64(laddr);

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == sizeof(int4), "Invalid vectorization");
    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == sizeof(int4), "Invalid vectorization");
    EP_STATIC_ASSERT(sizeof(*atomic_seg_ptr) == sizeof(int4), "Invalid vectorization");
    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == sizeof(int4), "Invalid vectorization");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<int4*>(&ctrl_seg));
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<int4*>(&raddr_seg));
    st_na_relaxed(reinterpret_cast<int4*>(atomic_seg_ptr), *reinterpret_cast<int4*>(&atomic_seg_1));
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<int4*>(&data_seg));
}

__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(
    void* rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    if (is_local_copy) {
        atomicAdd(static_cast<unsigned long long*>(rptr), value);
    } else {
        nvshmemi_ibgda_device_qp_t* qp = ibgda_get_rc(pe, qp_id);

        __be32 rkey;
        uint64_t raddr;
        ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), pe, &raddr, &rkey, qp->dev_idx);

        uint64_t my_wqe_idx = ibgda_reserve_wqe_slots(qp, 1);
        void* wqe_ptrs = ibgda_get_wqe_ptr(qp, my_wqe_idx);

        ibgda_write_amo_add_wqe(qp, value, reinterpret_cast<uint64_t>(qp->ibuf.buf), qp->ibuf.lkey, raddr, rkey, my_wqe_idx, &wqe_ptrs);

        ibgda_submit_requests<true>(qp, my_wqe_idx, 1);
    }
}

__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // Local rank, no need for mapping
    if (rank == dst_rank)
        return ptr;
    /*
    peer_heap_base_p2p
    数据类型: void **（指针数组）
    含义: 数组，每个元素是指向nvshmem集群中的各个rank通过 P2P（Peer-to-Peer）访问的 PE 堆基地址的指针，用于 GPU 间直接访问。
         peer_heap_base_p2p表示的只是当前rank对整个多机多卡集群上的各个rank的是否可以进行p2p的信息，
         如果对应dst_rank返回的peer_base是0，就说明当前rank与dst_rank不能p2p访问，也就是不在同一个node上。
         如果对应dst_rank返回的peer_base不是0，就说明可以p2p访问，在同一个node上。记录的就是这个dst_rank的heap_base地址。
    注意: 在每个rank中的nvshmemi_device_state_d.peer_heap_base_p2p记录的信息是不一样的。
    作用:
        - 存储可通过 P2P 访问的 PE 的堆基地址
        - 用于 P2P 访问：当两个 GPU 支持 P2P 时，直接使用该地址进行访问
        - 在动态 VMM 模式下，可能使用虚拟地址映射，而非实际物理地址
    数组结构:
        - 元素数量：npes。npes：参与 NVSHMEM 通信的 PE 总数，包含多机多卡的所有PE。
        - 每个元素：void *（8 字节，64 位系统），表示该PE的P2P可访问堆基地址。
        - 总大小：npes * sizeof(void *) = npes * 8 字节
    数据排列方式:
        - 连续数组，索引对应 PE ID
        - peer_heap_base_p2p[i] 存储 PE i 的 P2P 可访问堆基地址
        - 如果 PE i 不支持 P2P，该元素可能为 0
        - 本地 PE 的地址：peer_heap_base_p2p[mype] = heap_base

    补充:
        nvshmemi_device_state_d.peer_heap_base_p2p[dst_pe] 和 nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]
        指向的都是rank dst_pe在整个nvshmem的PGAS体系下的对称内存的起始地址。
        区别: peer_heap_base_p2p[dst_pe] 表示当前 rank 能通过 GPU P2P（如 NVLink/PCIe peer access）直接访问的目标 rank 的 heap 基址（不可 P2P 时为 0）；
             peer_heap_base_remote[dst_pe] 则是用于 RDMA/IBGDA 等跨节点远程访问的远端基址，供构造远端 raddr 使用。
    */
    auto peer_base = __ldg(reinterpret_cast<uint64_t*>(nvshmemi_device_state_d.peer_heap_base_p2p) + dst_rank);

    // RDMA connected
    if (peer_base == 0)
        return 0;

    // NVLink P2P is enabled
    /*
    heap_base
    数据类型： void *（指针）
    含义： 指向当前 PE（Processing Element）的对称堆（symmetric heap）基地址。
          就是PGAS的PE之间通过对称内存（Symmetric Memory） 进行通信和数据共享，这类内存从位于GPU内存中的“对称堆（Symmetric Heap）”
    作用：
        - 作为本地堆的起始地址，用于计算堆内偏移
        - 在设备端用于地址转换：将本地地址转换为远程 PE 的对应地址
        - 用于判断地址是否在堆范围内
    分配方式：
        - 通过 cudaMalloc、cuMemAddressReserve 或共享内存等方式分配
        - 大小由 heap_size 决定，通常对齐到内存粒度（如 2MB）

    地址ptr相对于当前rank的heap_base的偏移量：ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base)
    加上peer_base，最终返回的就是ptr对应的在rank dst_rank的heap_base中的地址。
    */
    return peer_base + (ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base));
}

// This is a simplified version of NVSHMEM's `ibgda_poll_cq`.
// Note that this implementation does not guarantee thread safety,
// so we must ensure that no other threads are concurrently using the same QP.
/*
这是 NVSHMEM 的 ibgda_poll_cq 的简化版本。
注意：此实现不保证线程安全，因此必须确保没有其他线程并发使用同一个 QP。
*/
__device__ static __forceinline__ void ibgda_poll_cq(nvshmemi_ibgda_device_cq_t* cq, uint64_t idx) {
    /*
    nvshmemi_ibgda_device_cq_t* cq,  // 指向完成队列（Completion Queue）的指针
    uint64_t idx                     // 需要等待完成的最大 WQE 索引（实际维护的是 index + 1）
    */

    /*
    cqe64：类型为 mlx5_cqe64*，指向 64 字节 CQE 的数组的指针。
    cq->cqe：完成队列条目（Completion Queue Entry）数组的基地址
    mlx5_cqe64：Mellanox MLX5 的 64 字节 CQE 结构体
    将 cq->cqe 转换为 mlx5_cqe64*，用于访问硬件 CQE

    cq结构体（简化）
    typedef struct {
        void* cqe;             // CQE数组基地址
        uint32_t ncqes;        // CQE数量
        uint64_t* cons_idx;    // 消费者索引指针（软件维护）。已轮询完成的最大 WQE 索引 + 1。
    } nvshmemi_ibgda_device_cq_t;
    */
    const auto cqe64 = static_cast<mlx5_cqe64*>(cq->cqe);
    /*
    ncqes = Number of Completion Queue Entries（完成队列条目数量）
    这是 CQ 数组的大小，即可以存储多少个 CQE，是 CQ 的容量（固定值）
    用于溢出安全比较，因为 CQ 是环形缓冲区
    通常为 2 的幂：256、512、1024 等
    */
    const uint32_t ncqes = cq->ncqes;
    // CTA 级别内存屏障。acquire后不到前。防止在读取 CQ 状态前，内存操作被重排序。确保后续读取能看到最新的硬件状态
    memory_fence_cta();
    // *cq->cons_idx：软件维护的消费者索引（已轮询完成的最大 WQE 索引 + 1），如果该值大于等于idx，说明idx对应的WQE已经完成消费。
    if (*cq->cons_idx >= idx)
        return;
    // NOTES: this while loop is part of do-while below.
    // `wqe_counter` is the HW consumer index. However, we always maintain `index + 1`.
    // To be able to compare with the index, we need to use `wqe_counter + 1`.
    // Because `wqe_counter` is `uint16_t`, it may be overflow. Still, we know for
    // sure that if `idx - wqe_counter - 1 < ncqes`, `wqe_counter + 1 is less than
    // idx, and thus we need to wait. We don't need to wait when `idx == wqe_counter + 1`
    // That's why we use `- 2` here to make this case overflow.
    /*
    注意：这个 while 循环是下面 do-while 的一部分。
    wqe_counter 是硬件消费者索引。但我们总是维护 index + 1。
    为了能够与索引进行比较，我们需要使用 wqe_counter + 1。
    因为 wqe_counter 是 uint16_t，可能会溢出。
    但我们知道，如果 idx - wqe_counter - 1 < ncqes，则 wqe_counter + 1 小于 idx，因此需要等待。
    当 idx == wqe_counter + 1 时不需要等待。此时 idx - wqe_counter - 1 = 0， idx - wqe_counter - 2 = -1
    这就是为什么我们在这里使用 - 2 来使这种情况溢出。此时就是负数（-1），但是uint16_t是无符号的，
    所以负数（-1）就会变成65535，反而是uint16_t能表示的最大的正数，此时就肯定大于等于uint16_t格式的ncqes。
    
    不能直接用 idx < wqe_counter 的原因:
        溢出问题：wqe_counter 是 16 位，会从 65535 回绕到 0
        类型不匹配：idx 是 64 位，wqe_counter 是 16 位
        错误判断：溢出后直接比较会得出错误结论
    
    场景1：正常情况（无溢出）
        wqe_counter = 100  (已消费索引 0-99)
        idx = 50           (需要等待索引 49)
        直接比较: idx < wqe_counter → 50 < 100 → true ✓
        结论: 索引49已完成，不需要等待 ✓
    
    场景2：溢出情况（关键问题）
        假设 wqe_counter 从 65535 溢出到 0
        wqe_counter = 0    (实际上已消费了索引 0-65534，溢出后重新开始)
        idx = 100          (需要等待索引 99)
        直接比较: idx < wqe_counter → 100 < 0 → false ✗
        结论: 错误！实际上索引99已经完成了，但判断为未完成
    
    场景3：另一个溢出情况
        wqe_counter = 50   (溢出后的值，实际已消费了 65535 + 50 = 65685 个WQE)
        idx = 100          (需要等待索引 99)
        直接比较: idx < wqe_counter → 100 < 50 → false ✗
        结论: 错误！实际上索引99早就完成了，但判断为未完成
    */
    uint16_t wqe_counter;
    do {
        // &cqe64->wqe_counter = 已消费的最大WQE索引 + 1
        // 使用PTX指令进行字节序转换。将主机字节序转换为大端序（硬件使用大端序）
        wqe_counter = HtoBE16(ld_na_relaxed(&cqe64->wqe_counter));
    } while ((static_cast<uint16_t>(static_cast<uint16_t>(idx) - wqe_counter - static_cast<uint16_t>(2)) < ncqes));
    *cq->cons_idx = idx;  // 软件维护消费者索引，已轮询完成的最大 WQE 索引 + 1。

    // Prevent reordering of this function and later instructions
    // 防止函数返回后的指令被重排序到索引更新之前。确保其他线程能看到更新后的 cons_idx
    memory_fence_cta();
}

// Wait until wqe `idx - 1` is completed.
/*
等待指定目标 PE 的指定 QP 上所有已提交的 RDMA 操作完成。用于确保在清理缓冲区或进行同步操作前，所有未完成的 RDMA 写操作已完成。
*/
__device__ static __forceinline__ void nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    // qp：类型为 nvshmemi_ibgda_device_qp_t*，指向目标 PE 的指定 QP
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    // state：类型为 nvshmemi_ibgda_device_state_t*，指向 IBGDA 全局状态
    auto state = ibgda_get_state();
    /*
    prod_idx：当前PE需要等待完成当前PE的QP的最大 WQE 索引。
    use_async_postsend：是否使用异步提交模式。
    异步模式。特点: 高吞吐量，NIC 批量处理。延迟较高，适合训练阶段。
        qp->tx_wq.prod_idx 是指向已提交给 NIC 的 WQE 索引的指针。NIC 自动轮询并更新该索引。该值表示“已提交给 NIC 的最大 WQE 索引”。
        ld_na_relaxed：
            使用 PTX ld.relaxed.gpu.global.L1::no_allocate 指令
            非对齐、relaxed 内存顺序、不缓存到 L1
            性能优化：避免缓存污染，适合一次性读取
    同步模式。特点: 低延迟，GPU 立即触发。需要显式门铃，适合推理阶段。
        qp->mvars.tx_wq.ready_head 是已准备好提交的 WQE 索引。GPU 显式控制提交，通过门铃通知 NIC。读取该值表示“已准备好提交的最大 WQE 索引”。
    
    // QP结构体中的指针
    qp->tx_wq.prod_idx  // 指向 mvars->tx_wq.prod_idx 或内部 prod_idx

    // 管理变量中的值
    qp->mvars.tx_wq.prod_idx     // 已提交给NIC的WQE索引（异步模式使用）
    qp->mvars.tx_wq.ready_head   // 已准备好提交的WQE索引（同步模式使用）

    */
    uint64_t prod_idx = state->use_async_postsend ? ld_na_relaxed(qp->tx_wq.prod_idx) : ld_na_relaxed(&qp->mvars.tx_wq.ready_head);
    
    /*
    ibgda_poll_cq 作用: 轮询 CQ，直到所有索引小于 prod_idx 的 WQE 都已完成。
    qp->tx_wq.cq: 完成队列（Completion Queue）指针。
    prod_idx: 需要等待完成的最大 WQE 索引。
    */
    ibgda_poll_cq(qp->tx_wq.cq, prod_idx);
}

}  // namespace deep_ep
