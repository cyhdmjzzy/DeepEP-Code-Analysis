# 安装 NVSHMEM

## 重要通知

**本项目既非由 NVIDIA 赞助，也未获得其支持。**

**NVIDIA NVSHMEM 的使用受 [NVSHMEM 软件许可协议](https://docs.nvidia.com/nvshmem/api/sla.html) 条款的约束。**

## 先决条件

硬件要求：
   - 同一节点内的 GPU 需要通过 NVLink 连接
   - 不同节点间的 GPU 需要通过 RDMA 设备连接，参见 [GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)
   - 支持 InfiniBand GPUDirect Async (IBGDA)，参见 [IBGDA 概述](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)
   - 更多详细要求，请参见 [NVSHMEM 硬件规格](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#hardware-requirements)， [NVSHMEM 3.3.0 文档](https://docs.nvidia.com/nvshmem/api/introduction.html)

软件要求：
   - NVSHMEM v3.3.9 或更高版本

## 安装步骤

### 1. 安装 NVSHMEM 二进制文件

NVSHMEM 3.3.9 二进制文件有多种格式：
   - 适用于 [x86_64](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.3.9_cuda12-archive.tar.xz) 和 [aarch64](https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-sbsa/libnvshmem-linux-sbsa-3.3.9_cuda12-archive.tar.xz) 的压缩包
   - RPM 和 deb 包：可在 [NVHSMEM 安装程序页面](https://developer.nvidia.com/nvshmem-downloads?target_os=Linux) 上找到相关说明
   - conda-forge 提供的 Conda 包
   - PyPI 提供的 pip 轮包：`pip install nvidia-nvshmem-cu12`
DeepEP 与上游的 NVSHMEM 3.3.9 及更高版本兼容。


### 2. 启用 NVSHMEM IBGDA 支持

NVSHMEM 支持两种具有不同要求的模式。可使用以下任一方法启用 IBGDA 支持。

#### 2.1 配置 NVIDIA 驱动

此配置启用传统的 IBGDA 支持。

修改 `/etc/modprobe.d/nvidia.conf`：

```bash
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
```

更新内核配置：

```bash
sudo update-initramfs -u
sudo reboot
```

#### 2.2 安装 GDRCopy 并加载 gdrdrv 内核模块

此配置通过 CPU 辅助的异步发送后操作启用 IBGDA。有关 CPU 辅助的 IBGDA 的更多信息，可参见 [此博客](https://developer.nvidia.com/blog/enhancing-application-portability-and-compatibility-across-new-platforms-using-nvidia-magnum-io-nvshmem-3-0/#cpu-assisted_infiniband_gpu_direct_async%C2%A0)。
这种方式会带来轻微的性能损失，但在无法修改驱动注册表项时可以使用。

下载 GDRCopy
GDRCopy 有预构建的 deb 和 rpm 包，可在 [此处](https://developer.download.nvidia.com/compute/redist/gdrcopy/) 获取，也可在 [GDRCopy GitHub 仓库](https://github.com/NVIDIA/gdrcopy) 上获取源代码。

按照 [GDRCopy GitHub 仓库](https://github.com/NVIDIA/gdrcopy?tab=readme-ov-file#build-and-installation) 上的说明安装 GDRCopy。

## 安装后配置

当不通过 rpm 或 deb 包安装 NVSHMEM 时，请在 shell 配置中设置以下环境变量：

```bash
export NVSHMEM_DIR=/path/to/your/dir/to/install  # 用于 DeepEP 安装
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
```

## 验证

```bash
nvshmem-info -a # 应显示 nvshmem 的详细信息
```