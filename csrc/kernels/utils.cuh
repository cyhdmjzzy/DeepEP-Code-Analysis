#pragma once

#include "exception.cuh"

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                     \
    {                                                                                                                                 \
        constexpr int kLoopStride = 32 * (UNROLL_FACTOR);                                                                             \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                          \
        auto __src = (SRC);                                                                                                           \
        auto __dst = (DST);                                                                                                           \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                      \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32); \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);  \
        }                                                                                                                             \
        {                                                                                                                             \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                  \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);                                                           \
                }                                                                                                                     \
            }                                                                                                                         \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * 32 < (N)) {                                                                                           \
                    ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);                                                            \
                }                                                                                                                     \
            }                                                                                                                         \
        }                                                                                                                             \
    }

namespace deep_ep {

template <int kBytes>
struct VecInt {};
template <>
struct VecInt<1> {
    using vec_t = int8_t;
};
template <>
struct VecInt<2> {
    using vec_t = int16_t;
};
template <>
struct VecInt<4> {
    using vec_t = int;
};
template <>
struct VecInt<8> {
    using vec_t = int64_t;
};
template <>
struct VecInt<16> {
    using vec_t = int4;
};

template <typename FuncT>  // FuncT 通常是 lambda 或函数对象。
struct PatternVisitor {
    FuncT func;  // 存储传入的函数对象

    /*
    __device__ __host__：可在 GPU 和 CPU 上使用。
    explicit：禁止隐式转换。
    func(std::forward<FuncT>(func)): 给参数func传入一个右值引用，避免不必要的拷贝。
    左值和右值的参考文档: https://osvicyu5w5.feishu.cn/wiki/SqUnwOf6YitRpvkobKKcWpSznPg

    转发引用（只有在模板参数中，T&& 才有特殊含义）
    */
    __device__ __host__ explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}

    /*
    对操作符“[]”的重载：
    operator[](const uint32_t& i) { return func(i); }
    返回值：func(i)
    参数：const uint32_t& i
    作用：调用func函数，传入参数i，返回结果。
    */
    __device__ __host__ auto operator[](const uint32_t& i) { return func(i); }
};

__device__ __forceinline__ void trap() {
    asm("trap;");
}

__device__ __forceinline__ void memory_fence() {
    // 
    /*
    fence 指令会依据《内存一致性模型》的描述，在该线程发起的内存访问操作（包括 ld、st、atom 及 red 指令）之间建立排序关系。
    其中，作用域限定符指定了可能观察到本操作排序效果的线程集合。
    fence.acq_rel 是一种轻量级屏障指令，足以满足大多数程序的内存同步需求。当 fence.acq_rel 指令结合额外内存操作使用时，会实现同步效果，
    具体遵循《内存一致性模型》中 “获取 - 释放模式” 的相关定义。若指令中省略了可选的 .sem 限定符，则默认采用 .acq_rel 属性。
    */
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ void memory_fence_gpu() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void memory_fence_cta() {
    // acquire后不到前
    asm volatile("fence.acq_rel.cta;" ::: "memory");
}

__device__ __forceinline__ void st_relaxed_sys_global(const int* ptr, int val) {
    asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

/* 可以直接使用NVLink P2P进行远程传输，也就是同节点内的rank之间可以直接进行数据传输。 */
__device__ __forceinline__ void st_release_sys_global(const int* ptr, int val) {
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_release_cta(const int* ptr, int val) {
    /*
    st：store（存储）操作
    .release：release内存序，保证在此存储之前的所有内存操作不会被重排到此存储之后
    .cta：CTA作用域，保证操作对整个CTA内所有线程可见
    .s32：32位有符号整数
    [%0]：存储到地址%0
    */
    asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_acquire_global(const int* ptr) {
    int ret;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
}

__device__ __forceinline__ int ld_acquire_cta(const int* ptr) {
    int ret;
    /*
    ld：load（加载）操作
    .acquire：acquire内存序，保证在此加载之后的所有内存操作不会被重排到此加载之前
    .cta：CTA（线程块）作用域，保证操作对整个CTA内所有线程可见
    .s32：32位有符号整数
    [%1]：从地址%1加载
    */
    asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t* ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t* ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
    int ret;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float ld_volatile_global(const float* ptr) {
    float ret;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const int64_t* ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const uint64_t* ptr) {
    int64_t ret;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
/*
使用 ld.global.nc.L1::no_allocate：
    ↓
明确指示：不要在 L1 中分配缓存行
    ↓
绕过 L1，直接从 L2 或全局内存读取
    ↓
避免读取到 L1 中的脏数据
    ↓
读取结果正确 ✅ 但是从 L2 或全局内存中读取，速度慢
*/
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA.[width].CONSTANT` in SASS
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t* ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}

template <>
__device__ __forceinline__ uint8_t ld_nc_global(const uint8_t* ptr) {
    uint16_t ret;
    // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned 16-bit)
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
}

template <>
__device__ __forceinline__ int ld_nc_global(const int* ptr) {
    int ret;
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ int64_t ld_nc_global(const int64_t* ptr) {
    int64_t ret;
    asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ float ld_nc_global(const float* ptr) {
    float ret;
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ int2 ld_nc_global(const int2* ptr) {
    int2 ret;
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
}

template <>
__device__ __forceinline__ int4 ld_nc_global(const int4* ptr) {
    int4 ret;
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

/*
na: Non-Aligned, 非对齐。
*/
__device__ __forceinline__ void st_na_relaxed(const uint8_t* ptr, uint8_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(static_cast<uint16_t>(val)));
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t* ptr, uint16_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_relaxed(const int* ptr, int val) {
    /*
    no_allocate: Do not allocate data to cache. This priority is suitable for streaming data.
    */
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_relaxed(const int4* ptr, int4 val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__ __forceinline__ void st_na_release(const int* ptr, int val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_release(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_release(const uint64_t* ptr, uint64_t val) {
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t* ptr, const dtype_t& value) {
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
__device__ __forceinline__ void st_na_global(const int* ptr, const int& value) {
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int64_t* ptr, const int64_t& value) {
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const float* ptr, const float& value) {
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int4* ptr, const int4& value) {
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

__device__ __forceinline__ float log2f_approx(const float& x) {
    float ret;
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

__device__ __forceinline__ float exp2f_approx(const float& x) {
    float ret;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

__forceinline__ __device__ int get_lane_id() {
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

/*
elect_one_sync 的参考文档: https://osvicyu5w5.feishu.cn/wiki/GwnEwR9JdiHhLok5OJ0cIV2UnGg
Hopper架构GPU中的核心作用：选择warp中的一个线程，并且同步warp中的所有线程。
*/
__device__ __forceinline__ uint32_t elect_one_sync() {
#ifndef DISABLE_SM90_FEATURES
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
#else
    return get_lane_id() == 0;
#endif
}

// TMA PTX instructions
#ifndef DISABLE_SM90_FEATURES

__device__ __forceinline__ void fence_barrier_init() {
    /*
    PTX内存屏障指令，确保mbarrier初始化操作在cluster全局可见。
    指令详解：
    fence：内存屏障。
    .mbarrier_init：针对mbarrier初始化操作的特定屏障。
    .release：释放语义。保证该屏障前的所有写操作（特别是mbarrier.init）在该屏障对其他线程可见之前完成。  “release前不到后”， “acquire后不到前”
    .cluster：作用域为整个cluster，确保cluster内所有SM都能看到一致的初始化状态。

    为什么mbarrier初始化屏障需要.cluster作用域？
        在Hopper架构中，Cluster是同步和通信的基本单元：
        一个Cluster包含多个SM（具体数量依GPU型号而定）
        Cluster内的SM可以直接访问彼此的共享内存（通过shared::cluster）
        TMA的cp.async.bulk操作是集群级的：一个SM发起的TMA拷贝，数据可能从另一个SM的L2缓存或全局内存传输过来。
    
    批量异步拷贝操作的源地址和目标地址可位于共享内存或全局内存中，支持以下传输方向：
    1. 从全局内存读取数据到共享内存；
    2. 从共享内存写入数据到全局内存；
    3. 在同一集群（cluster）内，从共享内存拷贝到另一线程块的分布式共享内存（Distributed Shared Memory，Hopper架构开始有的）。
    在第三种模式中，需要使用.cluster作用域，确保cluster内所有SM都能看到一致的初始化状态。
    如果TMA拷贝的源和目标地址设计涉及跨SM的DSM，或者TMA操作本身是集群级别的，那么执行TMA拷贝的硬件单元（可能位于另一个SM上） 需要能看到发起方CTA设置的mbarrier状态。这正需要 fence.mbarrier_init.release.cluster 来确保这种跨SM的可见性。
    而且PTX官方文档也是这么写的。
    */
    asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
    /*
    __cvta_generic_to_shared函数将该指针的通用地址转换为共享内存空间地址, 因为这个ptx指令中需要共享内存空间的地址。
    硬件背景：GPU的MMU将虚拟地址空间划分为不同区域（全局、共享、局部等）。PTX指令操作共享内存时，需要明确的空间标识符（如.shared）。此转换确保地址被正确解释。
    类似的还有__cvta_shared_to_generic，将共享内存空间地址转换为全局内存空间地址。
    __cvta_generic_to_global，将全局内存空间地址转换为共享内存空间地址。
    为什么不让global内存和shared内存都用通用的generic地址呢？
    答: 
    一、通用地址（Generic Address）的本质与局限
    CUDA 中的通用地址（Generic Address） 是一种 “统一寻址” 抽象，理论上可指向任意地址空间（global/shared/local/const/param 等），
    但硬件层面并不直接支持通用地址的原生访问 —— 所有通用地址的内存操作，最终都需要硬件或编译器完成地址空间解析，再映射到对应物理存储的访问逻辑。
    若强制让 global/shared 完全依赖通用地址，会带来核心问题：
    1. 性能损耗：地址解析的额外开销
        硬件层面，global 内存（全局内存）和 shared 内存（共享内存）的访问路径完全不同：
        shared 内存是片上高速存储，通过 ld.shared/st.shared 指令直接访问，地址仅需 32 位（甚至更小），无地址翻译延迟；
        global 内存需经过 MMU（内存管理单元）的虚拟地址→物理地址翻译，且地址宽度为 64 位（支持大内存寻址），访问指令为 ld.global/st.global。
        若仅用通用地址访问，每次内存操作都需先解析地址所属的空间（硬件需额外电路判断地址范围），再分支到对应访问逻辑，
        这会增加指令周期，抵消 shared 内存的高速优势（shared 内存延迟通常比 global 低 1~2 个数量级，额外解析开销会显著放大延迟）。
    2. 指令设计的专用性：PTX 指令的空间绑定
        PTX 作为 CUDA 的中间指令集，其内存访问指令强绑定地址空间（如 ld.shared.u32/ld.global.u32），目的是让编译器 / 硬件精准优化访问逻辑：
        shared 内存指令可利用片上存储的并行访问特性（如 bank 冲突优化）；
        global 内存指令可触发缓存策略（L1/L2 缓存）、合并访问优化等。
        若全部使用通用地址，PTX 指令无法区分空间属性，编译器无法针对性优化，硬件也无法调度专用访问通路，导致内存访问效率大幅下降。
    二、为什么需要地址空间转换，而非全用通用地址？
    1. 硬件层面：地址空间的物理隔离
        shared 内存是每个 CTA（线程块）私有的片上存储，地址空间是 “局部” 的（仅在 CTA 内有效），地址宽度仅需 32 位即可覆盖（多数架构下 shared 内存最大为 256KB，32 位地址足够）；
        global 内存是全局存储，地址空间是 “全局” 的，需 64 位地址支持 TB 级甚至 PB 级寻址。
        通用地址为了兼容所有空间，必须采用 64 位宽度，若用通用地址存储 shared 内存地址，会浪费 32 位空间（如数据结构中存储大量 shared 地址时，64 位指针比 32 位地址占用翻倍），违背性能优化的核心目标。
    2. 软件层面：优化与兼容性需求
        性能优化：如前文文档所述，shared/local/const 空间的地址范围小于 32 位，可将其转换为 32 位整数存储，减小数据结构体积（如链表、数组中的地址字段），降低内存占用和访存带宽；
        指令互操作：部分 PTX 指令（如 ld.shared）仅接受对应空间的专用地址，不支持通用地址，必须通过 __cvta_generic_to_shared 等函数完成转换，才能保证指令合法执行；
        兼容性：老架构（如 sm_52 及之前）对通用地址的支持有限，专用地址空间可保证代码在不同架构下的稳定性和可预测性。
    */
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    /*
    PTX内联汇编：执行mbarrier.init指令。
    作用：在指定内存地址初始化一个mbarrier，硬件将内部计数器清零，并设置“期待到达数”为arrive_count。
    指令详解：
        mbarrier.init：初始化一个mbarrier对象。
        .shared::cta：mbarrier位于共享内存，作用域为整个CTA（线程块）。
        .b64：操作64位内存位置（mbarrier对象大小）。
        [%1]：操作数1，mbarrier对象的内存地址（mbar_int_ptr）。
        %0：操作数0，期望的到达计数（arrive_count，此处为1）。
    
    mbarrier.init.shared::cta.b64 [shMem], 12;
    mbarrier硬件对象在物理上包含了两个独立计数器，而不是一个：
    到达计数器	arrival_count	记录有多少个“到达”事件已发生	每次执行mbarrier.arrive（包括expect_tx变体）或TMA完成时加1
    事务字节计数器	transaction_byte_count	记录期待/已完成的字节总数	mbarrier.arrive.expect_tx增加期待值；TMA完成时增加完成值
    */
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar_ptr) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mbar_int_ptr));
}

/* 
一个mbarrier只能对应一个stage的等待状态。并不是这里讨论的是多级流水线模式就意味着一个mbarrier可以对应多个stage的等待状态。
这里的多级流水线模式是指这一个mbarrier对应的是phase中的哪一位比特位的stage的相位。

注意: 实际上，根本就没有必要把多个stage的相位都放在一个uint32_t类型的变量phase中，这样也节约不了多少内存，还不如每个stage设置一个uint32_t类型的相位变量phase，
      这样更清晰，更易于理解。在internode.cu的combine函数中的 “uint32_t tma_phase[kNumStages] = {0};” 就是这样的。

*/
template <bool kWithMultiStages = false>
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase, int stage_idx = 0) {
    /*
    __cvta_generic_to_shared()：将通用地址转换为共享内存空间地址
    mbarrier对象：位于共享内存的一个64位数据结构。它的核心是一个由硬件自动更新的到达计数器。
        这个计数器每个mbarrier只有一个。它记录“有多少个预期的到达事件已经发生”（例如，发起了多少次TMA拷贝）。
        mbarrier硬件指令需要共享内存的专用地址格式。 转换为uint32_t供PTX内联汇编使用。
    kWithMultiStages模式下，多个stage通常对应多个mbarrier对象。
    */
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    /*
    单级模式 (kWithMultiStages=false)：直接使用phase值
    多级流水线模式 (kWithMultiStages=true)：从phase中提取特定流水线阶段的比特
    例如3级流水线: phase=0b101, stage_idx=1 → wait=(0b101>>1)&1=0
    phase中的每一个比特位都是对应一个stage的相位等待状态。
    // 三级流水线示例
    phase bits: [stage2][stage1][stage0]
    // 阶段0等待相位比特0
    // 阶段1等待相位比特1  
    // 阶段2等待相位比特2
    每个流水线阶段独立控制，允许重叠执行：
        阶段0处理第N批数据时
        阶段1可同时处理第N-1批数据
        最大化硬件利用率
    
    wait 就是单级模式下期望的奇偶值（Expected Parity），或者多stage情况下的一个stage的期望的奇偶值，它是一个软件预期的奇偶状态（0或1）。
    
    一个比特位（奇偶性）之所以能区分，是因为我们将连续发生的、需要区分的“批次”，通过约定，强制安排成交替的奇偶性。
    前提：调用函数时必须保证，对于同一个mbarrier，同一奇偶值的等待阶段，只发起一轮TMA操作。即，在phase=0时发起第N批拷贝并等待，完成后翻转为phase=1，再发起第N+1批拷贝。这样，当奇偶值再次回到0时，它必然对应第N+2批拷贝，而不会是第N批。
    作用：这个比特位防止了“同一奇偶状态下，不同批次操作”之间的混淆。它解决了“完成信号到底是第N批的还是第N+2批的”这类跨越多个批次的ABA问题。对于紧邻的两批（N和N+1），它们的奇偶值本身就不同（0和1），所以不需要防ABA。
    
    这里讨论的多级，更可能是指 “多个并行的TMA加载流” 。例如，一个线程块需要从全局内存的不同不连续区域加载多个数据块到共享内存的不同位置，然后统一计算。
    举例说明（三级流水线：Load -> Compute -> Store）：
    例子1: 多satge同时发起（如果硬件支持），适用于数据和任务之间没有依赖关系的情况。。比如下面有12个份数据在3个satge上用TMA并行传输。
    假设有连续的数据块 Data1, Data2, Data3,... Data12
    时间线:
        时刻 T0: Stage0(Load Data1)  | Stage1(Compute Data2)  | Stage2(Store Data3)
        时刻 T1: Stage0(Load Data4)  | Stage1(Compute Data5)  | Stage2(Store Data6)
        时刻 T2: Stage0(Load Data7)  | Stage1(Compute Data8)  | Stage2(Store Data9)
        时刻 T3: Stage0(Load Data10) | Stage1(Compute Data11) | Stage2(Store Data12)
    
    例子2: 多satge交错发起以隐藏延迟。适用于数据和任务之间有依赖关系的情况。
    比如如果每一份数据都要经过三级流水线：Load -> Compute -> Store，那么就需要交错发起，以隐藏延迟。
    假设有连续的数据块 Data1, Data2, Data3,...
    时间线:
        时刻 T0: Stage0(Load Data1)
        时刻 T1: Stage0(Load Data2) | Stage1(Compute Data1)
        时刻 T2: Stage0(Load Data3) | Stage1(Compute Data2) | Stage2(Store Data1)
        时刻 T3: Stage0(Load Data4) | Stage1(Compute Data3) | Stage2(Store Data2)
        时刻 TN: ...
    */
    const auto& wait = kWithMultiStages ? (phase >> stage_idx) & 1 : phase;
    /*
    +--------+-------------------+--------+-------------------+-----------------------+----------+
    | Phase  | Arrive_Count      | Lock   | Transaction_Count | Expected_Arrive_Count | Reserved |
    | (1bit) | (20bit) [int20_t] | (1bit) | (21bit) [int21_t] | (20bit) [int20_t]     | (1bit)   |
    +--------+-------------------+--------+-------------------+-----------------------+----------+
    | bit 63 | bit 62~43         | bit 42 | bit 41~21         | bit 20~1              | bit 0    |
    +--------+-------------------+--------+-------------------+-----------------------+----------+
    mbarrier在数据表示上表达为一个64bit的整数类型数据，其内部各个域的定义如上图所示，整体分为六个域，分别是:
        最低位的第0比特的保留域，
        第1比特到第20比特的Expected_Arrive_Count域，即期望arrive的数量。初始化的时候设置一次，存入的是正数参数的相反数，也就是负数，用补码表示的，类型是int20_t。
            虽然在使用mbarrier_init初始化的时候已经设置了Expected_Arrive_Count的值，但是当每次mbarrier的Phase翻转时，
            就需要把Arrive_Count重新设置为Expected_Arrive_Count，所以Expected_Arrive_Count需要在初始化mbarrier的时候记录起来。
            有的mbarrier.arrive指令，比如mbarrier.arrive.b64 %r2, [addr], cnt;可以设置arrive—count大于1，可能导致Arrive_Count从负数变为大于0的数，此时Lock就会变为1，报错。
        第21比特到第41比特的Transaction_Count域。
            mbarrier初始化的时候，Transaction_Count是0，当使用mbarrier.expect_tx指令设置期望的Transaction_Count时，就会把Transaction_Count设置为相反数。
            而mbarrier::complete_tx::bytes指令就会把mbarrier的Transaction_Count加上对应的值，直到Transaction_Count变为0，就满足了Transaction的条件。
            如果mbarrier::complete_tx::bytes指定的值累加到Transaction_Count大于0，则TODO，
            我猜根据tx-count的范围判断当Transaction_Count加上一个值之后变号了，也不会导致lock报错，因为事务变量就是比较灵活，得靠用户自己实现可以变为0。
        第42比特位置的Lock域，即错误锁标志位，为0表示没有错误，为1表示有错误。
        第43比特到第62比特的Arrive_Count域，即当前到达的数目。
        第63比特位置的Phase域，防止ABA问题。
    
    The valid range of each of the counts is as shown below:
        Count name	                    Minimum value	Maximum value
        Expected arrival count	        1	            2^20 - 1
        Pending arrival count	        0	            2^20 - 1
        tx-count(Transaction_Count)     -(2^20 - 1)	    2^20 - 1

    关于等待的问题，只要出现63位的Phase的值与传入的参数wait不一样，就等到了，就完成了，结果“真（1）”就会写入谓词寄存器P1，bra DONE就会跳转到分支标签DONE，这个汇编循环结构就退出了。
    因为参数wait就是来查看相位状态是wait的状态是否完成的，如果现在63位的Phase相位状态不是wait对应的状态，那么就说明wait对应的状态已经完成了。
    当mbarrier.try_wait.parity等到了之后，也就是Arrive_Count和Transaction_Count都变成0了，63位的Phase的值就会翻转。
    所以，来调用这个函数时，只有当第63比特位置的Phase域和传进来的参数wait相等的时候才有意义，否则一调用这个函数就会等到了，就退出了。

    mbarrier.try_wait.parity：关键指令
        .parity：使用奇偶校验机制（与之前讨论的phase ^= 1相关！）
        .shared::cta：作用于CTA（线程块）级别的共享内存屏障
        .b64：操作64位mbarrier对象
    参数：
        [%0]：mbarrier对象地址（mbar_int_ptr）
        %1：期望的相位值（wait）
        %2：超时计数器（0x989680 = 10,000,000周期 ≈ 计算后的超时时间）
    逻辑：循环尝试，直到屏障达到指定相位才继续

    mbarrier.try_wait.parity 官方文档:
    mbarrier.try_wait.parity{.sem.scope}{.shared{::cta}}.b64  waitComplete, [addr], phaseParity {, suspendTimeHint};
    
    详解下面的asm volatile(...):
    1、汇编结构：一个带谓词跳转的循环
        {
            .reg .pred       P1;          // 声明一个谓词寄存器P1
            LAB_WAIT:                     // 标签：循环开始点
            mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;  // 关键等待指令
            @P1 bra DONE;                // 如果P1为真（等待成功），跳转到DONE
            bra     LAB_WAIT;            // 否则，跳回LAB_WAIT继续循环
            DONE:                         // 标签：等待完成，退出循环
        }
        { ... }：PTX指令作用域。在此范围内声明的寄存器（如.reg .pred P1）是局部的。
        .reg .pred P1;：声明一个谓词寄存器。
        .reg：声明寄存器。
        .pred：寄存器类型为谓词（1位布尔值，用于条件执行）。
        P1：寄存器名称。mbarrier.try_wait.parity指令的结果（成功/失败）将写入这里。
        LAB_WAIT: 和 DONE:：标签，用于bra（分支/跳转）指令的目标。
        @P1 bra DONE;：条件跳转。
        @P1：指令修饰符，表示“当谓词寄存器P1为真时”才执行后面的bra指令。
        bra DONE：无条件跳转到DONE标签。
        bra LAB_WAIT;：无条件跳转回LAB_WAIT，形成主动轮询循环。这与CPU上可能引起线程挂起的锁不同，GPU线程持续查询，延迟更低。

    2、核心指令: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;
        (1) 指令修饰符 (Modifiers)
        mbarrier.try_wait：指令根操作，尝试等待屏障。
        .parity：关键修饰符。指定本次等待使用奇偶相位校验机制。它要求指令检查mbarrier硬件状态中Phase域（bit 63）的值。
        .shared::cta：地址空间与作用域。
        .shared：操作对象（[%0]）位于共享内存。
        ::cta：此mbarrier的作用域为整个CTA（线程块）。CTA内的所有线程都看到同一个屏障状态。
        .b64：操作数大小。表示对[%0]地址的操作是64位的，这与mbarrier对象的大小完全一致。

        (2) 操作数 (Operands)
        指令格式为：指令 结果寄存器, [地址], 立即数1, 立即数2
        P1：结果寄存器。指令执行后，P1被设置为1（真）表示等待成功，0（假）表示条件未满足。
        [%0]：mbarrier对象地址。对应mbar_int_ptr。
        []：表示间接寻址，取该地址中的值（即64位mbarrier状态）。
        %0：内联汇编的占位符，对应后面的第一个输入操作数r(mbar_int_ptr)。
        %1：期望的奇偶相位值 (Expected Parity)。对应wait变量。指令会将其与mbarrier状态中的Phase域（bit 63）进行比较。
        %2：到达计数阈值 (Arrival Count Threshold)。对应立即数0x989680（十进制10,000,000）。这是理解你代码特殊性的关键。
    
    3、操作数约束与传入: ::"r"(mbar_int_ptr), "r"(wait), "r"(0x989680)
        :: 分隔开汇编模板和操作数约束列表。
        "r"：约束，表示该值需要放入一个32位通用整数寄存器。
        传入的三个值按顺序绑定到 %0, %1, %2。
        给suspendTimeHint设置一个极大的超时等待值0x989680，也就是1000万纳秒，1000万个时钟周期。
        虽然这里没有显式的 : "memory" 破坏描述符，但mbarrier指令本身隐含了内存排序和同步语义，编译器会保守地假设内存被修改。
    
    
    mbarrier.try_wait.parity的suspendTimeHint参数设置为: "r"(0x989680)，超时时间设置为1000万个时钟周期
    mbarrier.try_wait.parity 指令的执行是原子的、单次的，它不会在指令内部进行“多次对比”。其硬件执行流程可以更精确地描述如下：
    1. 原子快照与检查：指令执行时，硬件会原子性地读取mbarrier的完整64位状态，并一次性检查所有释放条件（基于我们讨论的结构图）：
    - 当前相位 Phase (bit 63) 是否等于 传入的期望相位 %1 (wait)？
    - 当前到达计数 Arrive Count (bits 62-43) 是否大于等于 预期的到达计数 Expected Arrive Count (bits 19-0)？
    - 当前事务计数 Transaction Count (bits 41-21) 是否大于等于 累计的期望字节数？
    2. 瞬时决策：
    - 如果所有条件同时满足：硬件会原子性地完成操作（例如翻转Phase位），并立即将谓词寄存器 P1 设置为 1。指令执行结束，线程继续运行下一条指令（@P1 bra DONE）。这个过程没有挂起。
    - 如果任一条件不满足：硬件不会进行任何修改。此时，根据指令的“潜在阻塞”属性，执行线程会立即进入挂起状态，而不是进行轮询。这就是你所说的“第一次对比不满足之后就挂起”。
    所以，“条件不满足时”指的正是：在指令执行的那个瞬间，对mbarrier状态的原子快照检查发现条件未满足。一旦不满足，线程立即挂起，不存在“挂起过程中进行多次对比”。

    */
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::"r"(mbar_int_ptr),
        "r"(wait),
        "r"(0x989680));
    /*
    相位更新：
    等待完成后翻转相位标志位
    单级：phase ^= 1（在0/1间切换）
    多级：phase ^= (1 << stage_idx)（翻转指定流水线阶段的比特）
    这实现了相位交替，为下一轮TMA拷贝做准备
    */
    phase ^= kWithMultiStages ? (1 << stage_idx) : 1;
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    /*
    关键指令：mbarrier.arrive.expect_tx。
    作用：通知mbarrier，期待一个特定大小的传输事务。这个大小的值会存储在硬件状态机中。
    硬件状态机：当软件向mbar_ptr指向的共享内存地址执行特定的PTX指令时，GPU的内存子系统/同步单元会识别这些指令，
              并操作对应的硬件计数器，而不是直接修改共享内存的值。
    
    这个函数本身不累加事务字节计数器，只是设置事务字节检测条件。还更新Arrive_Count域，使其自增1。

    执行流程：
        指令执行时，mbarrier的期待事务计数器会**增加（实际上是相反数减少）**num_bytes。
        当TMA拷贝完成时，它会向mbarrier报告完成的事务字节数。
        当报告的完成字节数 >= 期待的事务字节数，且达到arrive_count（初始化时设为1）时，mbarrier才满足打开条件。
        为什么需要？ 这允许将一次逻辑拷贝拆分为多个小事务，或者合并多个小拷贝为一个屏障等待，提供了灵活性。
    */
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(num_bytes), "r"(mbar_int_ptr));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_ptr) {
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::"r"(mbar_int_ptr));
}

__device__ __forceinline__ void tma_store_fence() {
    /*
    ​​fence.proxy.async.shared::cta 分为:
    fence, 表示同步指令;
    proxy, 涉及​​代理内存​​（非直接内存访问，如异步拷贝、Tensor Core 操作）;
    async, 表示异步拷贝
    shared::cta, 表示异步拷贝的数据范围是cta内的shared memory。
    所以这个指令表示: 每次进行tma操作前，TMA引擎对cta内的shared memory写入操作可见，
        这样能够确保通过这个同步之后，接下来的TMA操作能读取到正确的共享内存数据。
        至于说在通过这个同步之后到发布TMA操作之前共享内存还有别的更新，那TMA就不一定看得到了。
    mbarrier使用SM上的共享内存作为存储后端，为了提高其操作效率，硬件层面提供cache机构对其加速，只在初始化和销毁barrier的时候cache内容才会写回（write back）共享内存，其他操作均可以发生在cache内，并不向共享内存写回。
    我有个问题是：在执行fence.proxy.async.shared::cta之后，又有了往cta的共享内存写了数据new_data，那么接下来执行新的TMA读取共享内存到全局内存操作时是否可见new_data?
    根据我对大家使用fence.proxy.async.shared::cta和cp_async_bulk的观察，好像是都没有在二者之间往共享内存写数据，如果写了，我猜测可能是有可能TMA读的旧数据，有可能是新数据，所以不这么用就行。因为fence.proxy.async.shared::cta是为了保证TMA对最新的共享内存的可见性，既然cuda设计出这样的指令，那么这个指令就只能保证当时的可见性。
    
    官方文档最权威: 等待共享内存写入操作对 TMA 引擎可见。
    */
    asm volatile("fence.proxy.async.shared::cta;");
}

/*
下面这两个宏定义是理解Hopper架构TMA（Tensor Memory Accelerator）L2缓存策略控制的关键。它们是编码了特定控制信息的64位“魔数”，是L2缓存策略控制字。
kEvictFirst 和 kEvictNormal 是用于 cp.async.bulk 指令中 L2::cache_hint 修饰符的操作数。它们告诉GPU的L2缓存子系统，如何处理本次批量加载的数据。

L2缓存是GPU级别的，共享内存是SM级别的，和L1缓存是统一（unified）的。

kEvictNormal (0x1000000000000000): 这是默认策略。数据加载到L2缓存后，参与正常的缓存淘汰算法（如LRU）。如果缓存空间不足，它可能被后续访问的其他数据覆盖。
    适用于的场景：需要重用的数据：如果加载的数据块在后续计算中会被多个内核或多次访问，那么使用 kEvictNormal（或更积极的保留策略）可以让它更久地驻留在L2中，减少重复从全局内存加载的昂贵延迟。
kEvictFirst  (0x12f0000000000000): 这是“优先逐出”策略。这个数据被标记为“低优先级”，当L2缓存需要空间时，它会被优先考虑淘汰，为其他可能更重要的数据（如后续计算即将频繁访问的数据）让出空间。
    适用于的场景：一次性的中间数据：假设你加载一块数据到共享内存，处理完后就不会再重复访问它。这些数据在L2缓存中保留的价值很低。使用 kEvictFirst 标记，可以在使用后尽快从L2中清除，避免它们“污染”缓存，挤占那些需要被反复访问的数据（如权重参数）的空间。

在 tma_load_1d 函数中，evict_first 参数默认为 true，意味着作者默认假设：通过TMA加载到共享内存的数据，被消费（计算）完后，就不再需要了。 因此，优先从L2中逐出它们是优化整体缓存利用率的明智选择。
*/
constexpr uint64_t kEvictFirst = 0x12f0000000000000;
constexpr uint64_t kEvictNormal = 0x1000000000000000;

__device__ __forceinline__ void tma_load_1d(
    const void* smem_ptr, const void* gmem_ptr, uint64_t* mbar_ptr, int num_bytes, bool evict_first = true) {
    /* 将两个指针转换为PTX指令可用的共享内存地址格式。 因为TMAc操作需要共享内存空间的地址。 */
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    // 根据参数选择缓存提示枚举值。  evict_first：L2缓存提示策略
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    /*
    TMA load指令（从global memory到shared memory）
    核心PTX指令：cp.async.bulk，Hopper架构的批量异步拷贝指令。
    数据复制方向：global -> shared::cluster
    cp.async.bulk.dst.src.completion_mechanism{.multicast}{.level::cache_hint}
                        [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}

    .dst =                  { .shared::cluster }
    .src =                  { .global }
    .completion_mechanism = { .mbarrier::complete_tx::bytes }  传输完size个字节之后，就向关联的mbarrier报告。
    .level::cache_hint =    { .L2::cache_hint }
    .multicast =            { .multicast::cluster  }

    指令修饰符详解：
        shared::cluster：拷贝的目的地是共享内存。共享内存集群级别操作。
        global：拷贝的源数据在全局内存。
        mbarrier::complete_tx::bytes：拷贝完成时，按字节数向关联的mbarrier报告。
        L2::cache_hint：指定L2缓存行为提示。
    操作数：
        %0 (smem_int_ptr)：共享内存目标地址。
        %1 (gmem_ptr)：全局内存源地址（"l"约束表示64位立即数或寄存器）。
        %2 (num_bytes)：拷贝字节数。
        %3 (mbar_int_ptr)：关联的mbarrier地址。
        %4 (cache_hint)：缓存提示。
        : "memory"：编译器内存破坏描述，告知编译器内存内容已改变，防止错误优化。
    
    "h" = .u16 reg
    "r" = .u32 reg
    "l" = .u64 reg
    "f" = .f32 reg
    "d" = .f64 reg

    注意: .completion_mechanism修饰符指定指令变体支持的完成机制。不同变体支持的完成机制汇总如下表：
    --------------------------------------------------------------------------------------
    | .completion-mechanism | .dst	            | .src	        | Completion mechanism   |
    |-----------------------|-------------------|---------------|------------------------|
    | .mbarrier::...	    | .shared::cta      | .global	    | mbarrier based         |
    |                       | .shared::cluster  | .global	    |                        |
    |                       | .shared::cluster  | .shared::cta  |                        |
    ------------------------|-------------------|---------------|------------------------|
    | .bulk_group	        | .global	        | .shared::cta  | Bulk async-group based |
    --------------------------------------------------------------------------------------

    为什么tma_load_1d是将全局内存的数据写到.shared::cluster，但是tma_store_1d却是把.shared::cta的数据写入到全局内存？
    实际上tma_load_1d也可以将全局内存的数据写到.shared::cta。为什么不是这样的呢？
    */
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(smem_int_ptr),
        "l"(gmem_ptr),
        "r"(num_bytes),
        "r"(mbar_int_ptr),
        "l"(cache_hint)
        : "memory");
}

__device__ __forceinline__ void tma_store_1d(const void* smem_ptr, const void* gmem_ptr, int num_bytes, bool evict_first = true) {
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    /* 
    TMA store指令（从shared memory到global memory）
    数据复制方向：shared::cta -> global
    对于把共享内存的数据写入到全局内存，PTX官方文档只给了 cp.async.bulk.global.shared::cta.bulk_group 这条命令。
    cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [dstMem], [srcMem], size, cache-policy;
    */
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::"l"(gmem_ptr),
                 "r"(smem_int_ptr),
                 "r"(num_bytes),
                 "l"(cache_hint)
                 : "memory");
    /*
    这里提交bulk_group，不仅仅只是针对tma_store_1d中用了cp.async.bulk，还针对tma_load_1d中也用了cp.async.bulk。
    但是tma_load_1d中也用了cp.async.bulk是通过mbarrier同步的，？？？TODO：需要进一步分析。
    */
    asm volatile("cp.async.bulk.commit_group;");
}

/*
cp.async.bulk.wait_group 指令会使执行线程进入等待状态，直到当前执行线程最近的批量异步组中仅剩余 N 个或更少的未完成组，
且当前执行线程提交的所有比这N个未完成组更先前批量异步组均已完成。例如，当 N 为 0 时，执行线程会等待所有先前批量异步组完成。操作数 N 为整数常量。

默认情况下，cp.async.bulk.wait_group 指令会使执行线程等待指定批量异步组中所有批量异步操作完成。批量异步操作包括以下步骤：
1. （可选）读取张量映射（tensormap）；
2. 从源位置读取数据；
3. 写入各自的目标位置；
4. 使写入操作**对执行线程可见**。
但是！
可选修饰符 .read 表示等待 只 需持续到指定批量异步组中**所有**批量异步操作完成前两步即可：
1. 读取张量映射（tensormap）；
2. 从源位置读取数据。并且也是“TMA引擎读取完了数据”这一信息**对执行线程可见**。也就是说执行线程看到的TMA引擎读取源数据之后的TMA引擎的内存视图中是可以反映出TMA引擎已经读取了源数据的。
.read 可以避免后一次bulk_group任务必须等待前一次bulk_group任务彻底完成，从而提高效率。
因为当前一次bulk_group任务已经完成了对源位置的数据读取之后，源位置的内存空间就可以给后一次bulk_group任务使用了，尤常出现在源位置是共享内存中的给前后两个bulk_group任务共用的情况。
后一次bulk_group任务没必要等写入目的地的完成。
这样，就可以实现前后两次bulk_group任务的交叠执行，从而提高效率。
*/
template <int N>
__device__ __forceinline__ void tma_store_wait() {
    /*
    cp.async.bulk.wait_group.read %0;
    %0: 这是「占位符」（也叫 “操作数占位符”），对应后面输出列表里的第 0 个变量（这里就是N）。
    "n": n表示操作数是编译期常量（立即数），编译器会直接把N的值嵌入汇编指令（比如N=16，汇编会变成cp.async.bulk.wait_group.read 16;），不能用变量
    */
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

#endif

template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
    token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t>
/*
__shfl_sync(mask, value, src_lane) 的 value 只能参与 32 位寄存器级别的传输，即一次只能 shuffle 一个 int（32 位） 大小的标量。
为了每次传输更多的数据，需要将数据打包成 int 数组，然后每次传输一个 int 数组。
而 broadcast 就实现了这个。
典型应用: auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
*/

__device__ __forceinline__ dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");  // sizeof(dtype_t) % sizeof(int) = 2
    auto send_int_values = reinterpret_cast<int*>(&ptr);
    int recv_int_values[sizeof(dtype_t) / sizeof(int)];
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);
    // 下面最左边的*是解引用，表示获取它右边的指针的指向的内容
    return *reinterpret_cast<dtype_t*>(recv_int_values);
}

// Lane ID（线程道ID） 是CUDA中warp内部的线程索引，范围是 [0, 31]。
__forceinline__ __device__ int get_lane_id() {
    int lane_id;  // int 就是 s32 = signed 32-bit 有符号32位整数
    /*
    CUDA 内联汇编完全继承 GCC 风格，完整格式是4 个段，用3 个冒号分隔：
    asm [volatile] ("汇编指令模板" 
        : 输出操作数列表  // 第1段：数据从汇编流向C变量（可选）
        : 输入操作数列表  // 第2段：数据从C变量流向汇编（可选）
        : 破坏列表       // 第3段：告诉编译器哪些资源被修改（可选）
    );
    每个段都可以为空，但分隔符（冒号）必须保留；
    连续的冒号（比如::）就表示「前面的段是空的」。

    mov.s32: mov 数据移动指令。.s32是修饰符，表示操作32位有符号整数（s32 = signed 32-bit）
    %0: 这是「占位符」（也叫 “操作数占位符”），对应后面输出列表里的第 0 个变量（这里就是lane_id）。
    %laneid: GPU 硬件的特殊寄存器（只读），存储当前线程的 Warp 内索引（0~31），是硬件直接提供的 “天然值”。
        类似的特殊寄存器参见: https://osvicyu5w5.feishu.cn/wiki/VSeWwdMNEi0bpakhFy8cVSyTnue#share-Q7PedTYLzoauXJxVjPucbwA1nNc
    "=r": 约束符（Constraint），告诉编译器两件事：
        =: 表示这是输出操作数（数据从汇编指令流向 C/C++ 变量）；
        r: 表示编译器要把这个变量（lane_id）分配到 GPU 的通用寄存器（General Purpose Register，GPR）中（GPU 执行汇编指令时，只能直接操作寄存器，不能直接操作内存）；
    */
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;

__forceinline__ __device__ float fast_pow2(int x) {
    // We can ensure `-126 <= x and x <= 127`
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

__forceinline__ __device__ int fast_log2_ceil(float x) {
    auto bits_x = *reinterpret_cast<uint32_t*>(&x);
    auto exp_x = (bits_x >> 23) & 0xff;
    auto man_bits = bits_x & ((1 << 23) - 1);
    return exp_x - 127 + (man_bits != 0);
}

__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
    if (round_scale) {
        auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);
        scale = fast_pow2(-exp_scale_inv);
        scale_inv = fast_pow2(exp_scale_inv);
    } else {
        scale_inv = amax * kFinfoAmaxInvE4M3;
        scale = kFinfoAmaxE4M3 / amax;
    }
}

template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
__forceinline__ __device__ out_dtype_t extract_required_scale_format(float value) {
    if constexpr (kIsUE8M0) {
        return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
    } else {
        return value;
    }
}

/**
 * @brief GPU device function implementing a distributed barrier synchronization
 * across multiple ranks
 *
 * This function implements a barrier synchronization mechanism that allows
 * multiple GPU ranks to synchronize with each other using shared memory
 * pointers. Each rank signals its completion and waits for all other ranks to
 * complete before proceeding.
 *
 * @tparam kNumRanks Number of ranks participating in the barrier (compile-time
 * constant)
 * @tparam kSyncOnly If true, only performs synchronization without memory fence
 * operations
 *
 * @param barrier_signal_ptrs 2D array of pointers where
 * barrier_signal_ptrs[i][j] points to the signal location for rank i to
 * communicate with rank j
 * @param rank The current rank ID of this GPU process
 *
 * Algorithm:
 * 1. Optional memory fence to ensure visibility of prior memory operations
 * 2. Each thread signals its own rank's completion and decrements other ranks'
 * counters
 * 3. Wait until all other ranks have signaled completion (all values <= 0)
 * 4. Timeout detection to prevent infinite waiting in case of failures

 为什么使用模板参数？
    编译时优化：允许编译器进行循环展开和常量优化
    性能：避免运行时分支判断
    类型安全：确保编译时就知道PE数量

  kNumRanks：当前节点参与barrier的PE数量（编译时常量），不会超过 NUM_MAX_NVL_PEERS。
  kSyncOnly：是否只进行同步，不执行内存屏障（默认false）

  rank：nvl_rank
 */
template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, the memory operations by other threads in the block must be visible to the `sys` scope
    if constexpr (not kSyncOnly) {
        /* 
        为什么需要内存屏障？
        GPU内存模型允许乱序执行
        需要确保barrier信号的内存操作对其他PE可见，就是barrier_signal_ptrs的数据对其他PE可见
        防止编译器优化破坏内存访问顺序
        
        memory_fence(); 内存屏障：确保所有之前的内存操作对同一个节点内的所有GPU内的所有线程可见。
        ”fence.acq_rel.sys“ 中的 sys 表示 system scope
        System scope 影响同一节点内的所有GPU的内存系统，跨越GPU边界，不跨越节点边界。

        在下面的链接中搜索：.sys  其中第二个搜索结果中有一段话：
        If the threads are on different devices, the .sys scope must be used.
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
        */
        memory_fence();  // 内存屏障：确保所有之前的内存操作对同一个节点内的所有GPU的所有线程可见。
        __syncthreads();  // 确保内存屏障在所有线程上执行完毕
    }

    // Add self-ranks, sub other ranks
    if (thread_id < kNumRanks) {
        /*
        系统级原子操作：影响整个GPU内存系统
        原子性：操作不可被中断
        可见性：结果立即对所有PE可见

        barrier_signal_ptrs是void**类型的。
        barrier_signal_ptrs[rank]是void*类型的，通过cudaIpcOpenMemHandle 和 
        cudaMemcpy(barrier_signal_ptrs_gpu_, barrier_signal_ptrs_, ...)  
        指向rank rank的barrier_signal_ptrs_gpu_的地址，而rank rank的barrier_signal_ptrs_gpu_是指向连续8个int类型的指针。
        因此，barrier_signal_ptrs[rank] + thread_id表示的是rank rank的barrier_signal_ptrs_gpu_的第thread_id个int类型的指针。
        而 barrier_signal_ptrs[thread_id] + rank 表示的是rank thread_id的barrier_signal_ptrs_gpu_的第rank个int类型的指针。
        
        为什么需要下面这么复杂的同步？
        1、因为GPU内存模型的复杂性。GPU内存模型允许：
        1. 乱序执行
        2. 写缓冲
        3. 缓存一致性延迟
        4. 多级内存层次
        2、多线程竞争条件。
        不管在所有PE中只使用PE 0的一个内存地址（即一个整数值的内存地址）作为全局所有PE的所有线程的同步内存，
        还是在每个PE中都设置一个长度为kNumRanks的共享内存作为当前PE所有线程的同步内存，
        都会设计多线程竞争的情况，就是说都可能出现某一个内存地址同时被多个线程竞争写入。
        比如，如果全局所有PE的线程都只使用一个内存地址作为同步内存，那么显然就会出现所有线程都会竞争写入。
        如果是在每个PE中都设置一个共享内存作为当前PE所有线程的同步内存，
        那么就会有当前PE的threads_per_PE个线程对这一个内存地址竞争写入。
        所以，对于需要再当前节点进行所有PE同步的情况，一个节点有kNumRanks个PE，
        那么就得有kNumRanks * kNumRanks个内存地址进行竞争写入，这样的竞争写入最多只会有当前PE的一个线程和另外的一个线程进行竞争写入，
        当前PE是原子性地增加FINISHED_SUM_TAG，而另一个PE中对应当前PE的线程是原子性地减少FINISHED_SUM_TAG。
        于是，这样的竞争性大大减少，通过并行，可以很快完成所有PE的同步。

        // 不合适方案1：单一全局地址
        int global_signal = 0;
        // 所有PE的所有线程都竞争写入同一个地址
        // 竞争：kNumRanks * threads_per_PE 个线程竞争1个地址

        // 不合适方案2：每PE一个地址  
        int signals[kNumRanks];
        // 每个PE的所有线程竞争写入kNumRanks个地址
        // 竞争：threads_per_PE 个线程竞争kNumRanks个地址

        // 正确方案：kNumRanks * kNumRanks 个地址
        int signals[kNumRanks][kNumRanks];
        // 每个地址最多只有2个线程竞争（一个+，一个-）
        // 竞争：最多2个线程竞争1个地址

        如果一个rank中有 M 个block的 N 个线程执行barrier_block()，那么就会出现 2 * M * N 个线程竞争一个地址的情况。
        但是一般使用barrier_block函数时，一个rank中只有一个block的所有线程执行barrier_block()
        */
        atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
    }
    EP_DEVICE_ASSERT(kNumRanks <= blockDim.x);

    // Polling phase: wait for all other ranks to signal completion
    // Check timeout
    auto start_time = clock64();
    while (true) {
        // Read the signal value for this thread's corresponding rank
        // 读取rank rank的rank级barrier在rank thread_id的信号值。
        auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
        
        /*
        之所以可以使用__all_sync来同时判断当前PE上记录的其他同节点的所有PE的同步状态，
        是因为一个节点的PE数量kNumRanks最大不可能超过NUM_MAX_NVL_PEERS，
        而NUM_MAX_NVL_PEERS是8，而0xffffffff表示一个线程束中的32个线程，
        而且需要使用ld_volatile_global来读取同步状态值的时候，当前线程的id肯定是小于kNumRanks的，
        而当前线程所在的线程束的32个线程肯定就包含这kNumRanks个线程。

        atomicAdd_system执行的时候，当前rank上的线程是增加了FINISHED_SUM_TAG。
        现在如果满足了value <= 0，则说明别的rank上的线程已经对当前rank rank的barrier_signal_ptrs[rank] + thread_id减去了FINISHED_SUM_TAG。
        而且，同一个warp中的所有线程的指令同时（指的是指令的时钟周期）到达这里，如果都满足，说明当前rank的barrier_signal_ptrs上的kNumRanks个值同时都<=0，
        也就说明了当前rank的barrier_signal_ptrs上的kNumRanks个信号值都被不同的kNumRanks个rank进行了修改，这kNumRanks个不同的rank都到达了barrier。
        */
        if (__all_sync(0xffffffff, value <= 0))
            break;

        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);
            /* 
            停止执行：触发GPU异常
            调试友好：可以在调试器中查看状态
            */
            trap();
        }
    }
    // 如果当前线程能跳出了上面的while循环到达这里，说明
    __syncthreads();
}

__forceinline__ __device__ int atomic_cas_cta_acquire(int* addr, int x, int y) {
    /*
    PTX：atom.acquire.cta.shared::cta.cas.b32。
    语义：原子 Compare-And-Swap；若 *addr == x 则 *addr = y，返回操作前的 *addr（旧值）。
    acquire：该原子操作具有 acquire 语义——本指令之后的读/写不能被重排到本指令之前；保证「拿到锁之后」的临界区访问不会越过「拿锁」这条指令。
    cta：作用域为 CTA（block），与 __shared__ 锁只在同一 block 内使用一致。
    shared::cta：操作的是 shared 地址空间、cta 作用域。
    在锁中的角色：acquire_lock 里用 atomic_cas_cta_acquire(mutex, 0, 1) 尝试把 0 改成 1；成功（返回 0）的那次 CAS 就是「在锁变量上的一次 acquire」。
    之后该 warp 在临界区里读 tail/window 等，保证能看到上一次在同一个锁上做 release 的 warp 在 release 之前的所有写。
    */
    int ret;
    asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;" : "=r"(ret) : "l"(addr), "r"(x), "r"(y) : "memory");
    return ret;
}

__forceinline__ __device__ int atomic_exch_cta_release(int* addr, int x) {
    /*
    PTX：atom.release.cta.shared::cta.exch.b32。
    语义：原子 Exchange；把 *addr 写成 x，返回操作前的 *addr。
    release：该原子操作具有 release 语义——本指令之前的读/写不能被重排到本指令之后；保证「放锁之前」的临界区写（tail、window）对「之后在同一个锁上 acquire 的 warp」可见。
    在锁中的角色：release_lock 里用 atomic_exch_cta_release(mutex, 0) 把锁写成 0；这次写就是「在锁变量上的一次 release」。
    与下一个在同一个 mutex 上成功 CAS(0→1) 的 warp 形成 synchronizes-with。
    */
    int ret;
    asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(x) : "memory");
    return ret;
}

__forceinline__ __device__ void acquire_lock(int* mutex) {
    // To make later memory operations valid, we must use `acquire` for memory semantics
    /*
    mutex: 互斥锁
    逻辑：不断用 atomic_cas_cta_acquire(mutex, 0, 1) 尝试把 0 改成 1；返回 0 表示本次 CAS 成功（本 warp 把 0 改成了 1），退出 while；返回非 0 表示别人已持锁，继续自旋。
    注释：用 acquire 是为了让「拿锁之后」的临界区读/写不会被重排到「拿锁」之前，从而能正确看到上一个持锁者在临界区里的写。
    也就是 "acquire后不到前"。

    为什么window需要昂贵的锁:
    复杂状态管理：32位位图表示32个slot的释放状态
    并发竞争激烈：多个warp的相同lane可能同时更新同一个window（同一目标rank）
    操作类型：位操作（置位、位移、计数前导1等）
    并发模式：多写者竞争同一内存位置
    操作序列需要原子性：读tail，计算offset，读window，计算从bit 0 起连续多少个1，存储推进tail，右移window并存储。

    自旋锁的必要性:
    1、互斥保证：确保只有一个warp的lane能同时操作特定的window
    2、状态一致性：防止多个warp同时修改位图导致数据损坏
    3、公平性考虑：虽然是自旋锁，但通过临时的锁释放给其他warp机会
    */
    while (atomic_cas_cta_acquire(mutex, 0, 1) != 0)
        ;
}

__forceinline__ __device__ void release_lock(int* mutex) {
    // To make previous memory operations visible to other threads, we must use `release` for memory semantics
    /*
    逻辑：用 atomic_exch_cta_release(mutex, 0) 把锁写回 0。
    注释：用 release 是为了让「放锁之前」的临界区写对之后 acquire 的 warp 可见。
    也就是 "release前不到后"。
    */
    atomic_exch_cta_release(mutex, 0);
}

// Operation functors
template <typename T>
struct ReduceSum {
    __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct ReduceMax {
    // const：表示该操作不修改对象状态
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct ReduceMin {
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};
template <typename T>
struct ReduceAnd {
    __device__ T operator()(T a, T b) const { return a & b; }
};
template <typename T>
struct ReduceOr {
    __device__ T operator()(T a, T b) const { return a | b; }
};

// Unified reduction function
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
    EP_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or kNumLanesPerGroup == 4 or
                         kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
    constexpr uint32_t mask = 0xffffffff;
    /*
    T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
    这里没用第四个参数。有的地方也把laneMask叫lane_offset，offset的名称也的确体现了lane的间隔偏移量。
    为什么使用XOR而不是直接索引：
        配对对称性：XOR运算确保配对是双向的（A配对B，B也配对A）
        归约效率：创建完美的配对网络，便于树状归约
        硬件优化：GPU硬件对XOR模式的shuffle有特殊优化
    */
    if constexpr (kIntergroupReduce) {  
        // 组间归约模式， 从小间隔到大间隔：1 → 2 → 4 → 8 → 16
        if constexpr (kNumLanesPerGroup <= 1)
            value = op(value, __shfl_xor_sync(mask, value, 1));
        if constexpr (kNumLanesPerGroup <= 2)
            value = op(value, __shfl_xor_sync(mask, value, 2));
        if constexpr (kNumLanesPerGroup <= 4)
            value = op(value, __shfl_xor_sync(mask, value, 4));
        if constexpr (kNumLanesPerGroup <= 8)
            value = op(value, __shfl_xor_sync(mask, value, 8));
        if constexpr (kNumLanesPerGroup <= 16)
            value = op(value, __shfl_xor_sync(mask, value, 16));
    } else {  
        /*
        普通warp内归约（默认），从大间隔到小间隔：16 → 8 → 4 → 2 → 1
            适用场景：标准warp内归约，每个线程独立计算
            执行顺序：从粗粒度到细粒度，先处理大组间的交互
            使用案例：warp_reduce_max<8>(value) - 8线程组内最大值归约
            优势：适合大部分并行归约需求，性能稳定
        以 kNumLanesPerGroup = 8，lane_offset = 4 为例:
            lane 0 → target_lane = 0 XOR 4 = 4
            lane 1 → target_lane = 1 XOR 4 = 5
            lane 2 → target_lane = 2 XOR 4 = 6
            lane 3 → target_lane = 3 XOR 4 = 7
            lane 4 → target_lane = 4 XOR 4 = 0
            lane 5 → target_lane = 5 XOR 4 = 1
            lane 6 → target_lane = 6 XOR 4 = 2
            lane 7 → target_lane = 7 XOR 4 = 3
        如果 kNumLanesPerGroup = 15，lane_offset = 4 那么还有:
            lane 8 → target_lane = 8 XOR 4 = 12
            lane 9 → target_lane = 9 XOR 4 = 13
            lane 10 → target_lane = 10 XOR 4 = 14
            lane 11 → target_lane = 11 XOR 4 = 15
            lane 12 → target_lane = 12 XOR 4 = 8
            lane 13 → target_lane = 13 XOR 4 = 9
            lane 14 → target_lane = 14 XOR 4 = 10
            lane 15 → target_lane = 15 XOR 4 = 11
        可见配对对称性。
        */
        if constexpr (kNumLanesPerGroup >= 32)
            value = op(value, __shfl_xor_sync(mask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16)
            value = op(value, __shfl_xor_sync(mask, value, 8));
        if constexpr (kNumLanesPerGroup >= 8)
            value = op(value, __shfl_xor_sync(mask, value, 4));
        if constexpr (kNumLanesPerGroup >= 4)
            value = op(value, __shfl_xor_sync(mask, value, 2));
        if constexpr (kNumLanesPerGroup >= 2)
            value = op(value, __shfl_xor_sync(mask, value, 1));
    }
    return value;
}

// Convenience aliases
template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    // ReduceMax<T>{}：创建该结构体的临时对象（函数对象）
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_and(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceAnd<T>{});
}

template <int kNumLanesPerGroup = 32, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_or(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceOr<T>{});
}

}  // namespace deep_ep
