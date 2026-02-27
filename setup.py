import os
import subprocess
import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

"""
在 Linux/Unix 系统中，动态库通常有版本号后缀：
/opt/nvshmem/lib/
├── libnvshmem_host.so          # 符号链接 → libnvshmem_host.so.2
├── libnvshmem_host.so.2        # 符号链接 → libnvshmem_host.so.2.4.5
└── libnvshmem_host.so.2.4.5    # 实际文件
版本号的含义：
完整版本: libnvshmem_host.so.2.4.5 (major.minor.patch)
主版本: libnvshmem_host.so.2 (soname - 共享库名)
无版本: libnvshmem_host.so (链接器名)
"""
# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


if __name__ == '__main__':
    disable_nvshmem = False
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    nvshmem_host_lib = 'libnvshmem_host.so'
    if nvshmem_dir is None:
        try:
            nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
            nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
            import nvidia.nvshmem as nvshmem  # noqa: F401
        except (ModuleNotFoundError, AttributeError, IndexError):
            print(
                'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n'
            )
            disable_nvshmem = True
    else:
        disable_nvshmem = False

    if not disable_nvshmem:
        assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable', '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    sources = ['csrc/deep_ep.cpp', 'csrc/kernels/runtime.cu', 'csrc/kernels/layout.cu', 'csrc/kernels/intranode.cu']
    include_dirs = ['csrc/']
    library_dirs = []
    nvcc_dlink = []
    extra_link_args = ['-lcuda']

    # NVSHMEM flags
    if disable_nvshmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
    else:
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])
        include_dirs.extend([f'{nvshmem_dir}/include'])
        library_dirs.extend([f'{nvshmem_dir}/lib'])

        """
        nvcc_dlink每个参数的含义：
        -dlink：启用设备代码链接（Device Link）
        CUDA的RDC（Relocatable Device Code）模式需要这个
        将多个 .cu 文件中的设备代码（__device__, __global__ 函数）链接在一起
        -L/opt/nvshmem/lib：添加库搜索路径
        告诉链接器在这个目录下找库文件
        类似于 library_dirs，但这是给CUDA设备代码链接器用的
        -lnvshmem_device：链接NVSHMEM设备库
        链接 libnvshmem_device.a 静态库
        这个库包含在GPU上运行的NVSHMEM函数
        """
        nvcc_dlink.extend(['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem_device'])

        """
        extra_link_args每个参数的含义：
        -l:libnvshmem_host.so：链接特定版本的NVSHMEM主机库
        -l: 表示精确匹配文件名（而不是 -l 的自动添加lib前缀）
        这个库包含在CPU上运行的NVSHMEM管理代码
        -l:libnvshmem_device.a：链接NVSHMEM设备静态库
        .a 是静态库，会被完整嵌入到最终的 .so 文件中
        -Wl,-rpath,/opt/nvshmem/lib：设置运行时库搜索路径
        -Wl, 表示将后面的参数传递给链接器（ld）
        -rpath 设置运行时动态库搜索路径
        非常重要：没有这个，运行时会找不到 libnvshmem_host.so
        """
        extra_link_args.extend([f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a', f'-Wl,-rpath,{nvshmem_dir}/lib'])

    if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
        # Prefer A100
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')

        # Disable some SM90 features: FP8, launch methods, and TMA
        cxx_flags.append('-DDISABLE_SM90_FEATURES')
        nvcc_flags.append('-DDISABLE_SM90_FEATURES')

        # Disable internode and low-latency kernels
        assert disable_nvshmem
    else:
        # Prefer H800 series
        os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

        # CUDA 12 flags
        nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Disable LD/ST tricks, as some CUDA version does not support `.L1::no_allocate`
    if os.environ['TORCH_CUDA_ARCH_LIST'].strip() != '9.0':
        assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
        os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print(f' > NVSHMEM path: {nvshmem_dir}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    """
    使用python setup.py install实际安装的内容：
        Python源代码：deep_ep/*.py → site-packages/deep_ep/
        编译好的扩展：deep_ep_cpp.so（Linux）或 .pyd（Windows）→ site-packages/
        不包含：.cu、.cpp、.hpp 等源文件

    安装后的文件布局
    /Users/zhangziyang/anaconda3/envs/env_smpl/lib/python3.10/site-packages/
    ├── deep_ep/                    # Python包（来自include=['deep_ep']）
    │   ├── __init__.py             # ← 从deep_ep/__init__.py复制
    │   ├── buffer.py               # ← 从deep_ep/buffer.py复制
    │   └── utils.py                # ← 从deep_ep/utils.py复制
    ├── deep_ep_cpp.so              # C++扩展（从sources编译链接而来）
    └── deep_ep-1.1.0+xxx.egg-info/ # 元数据

    # 不会安装：
    # - csrc/ 目录下的任何文件 ✗
    # - tests/ 目录 ✗
    # - setup.py ✗


    编译链接后生成：deep_ep_cpp.so  # 或 deep_ep_cpp.pyd (Windows)
    这个 .so 文件包含：
        csrc/deep_ep.cpp 编译的代码（包括所有 #include 的头文件）
        csrc/kernels/*.cu 编译的所有CUDA kernels
        通过 PYBIND11_MODULE 导出的Python接口

    test代码也能调用：
        import deep_ep_cpp  # C++扩展（编译好的.so）

    """
    setuptools.setup(
        name='deep_ep',  # 定义使用python setup.py install时，python模块的名称为deep_ep。这是包在PyPI（Python Package Index）上的唯一标识符
        version='1.1.0' + revision,  # 定义包的版本号，格式为1.1.0加上Git短哈希值
        packages=setuptools.find_packages(
            include=['deep_ep']  # 自动发现并包含需要打包的Python模块，这里包括deep_ep目录下的所有Python模块
        ),
        ext_modules=[
            CUDAExtension(
                name='deep_ep_cpp',  # 定义编译后的C++扩展模块名称
                include_dirs=include_dirs, # 指定C++/CUDA编译时的头文件搜索路径
                library_dirs=library_dirs, # 指定链接时的库文件搜索路径，链接器需要找到libnvshmem_host.so等动态库

                # 为什么不把所有.cpp/.cu都加到sources？
                #   只编译实现文件（.cpp、.cu），不编译声明文件（.hpp、.cuh），因为：
                #   避免重复编译：头文件会被多个源文件 #include，如果编译头文件会导致符号重复
                #   编译模型：C++的设计就是：头文件声明，源文件实现
                #   条件编译：某些文件只在特定条件下需要（如 internode.cu 只在启用NVSHMEM时编译）
                sources=sources,  # 指定需要编译的源文件列表

                # 指定C++/CUDA编译时的额外编译选项，这些参数优化性能、控制编译行为、启用/禁用特性
                # 'cxx'：C++编译器参数（如-O3优化，-Wno-*禁用警告）
                # 'nvcc'：NVIDIA CUDA编译器参数（如-O3, -rdc=true启用可重定位设备代码）
                # 'nvcc_dlink'：设备链接参数（如-dlink, -lnvshmem_device）
                extra_compile_args=extra_compile_args,

                # 指定额外的链接器参数，确保编译的扩展能正确链接所需的外部库
                # -l:{nvshmem_host_lib}：链接NVSHMEM主机库
                # -l:libnvshmem_device.a：链接NVSHMEM设备静态库
                # -Wl,-rpath,{nvshmem_dir}/lib：设置运行时库搜索路径，避免运行时找不到.so文件
                extra_link_args=extra_link_args
            )
        ],
        # 自定义构建扩展模块的命令类
        # 为什么设置：
        #   BuildExtension是PyTorch提供的扩展构建类
        #   替代setuptools的默认build_ext命令
        #   提供了CUDA特有的构建逻辑：
        #   自动检测CUDA架构
        #   处理TORCH_CUDA_ARCH_LIST环境变量
        #   支持混合C++/CUDA编译
        #   处理设备代码链接（device code linking）
        #   使CUDA扩展的构建更加可靠和智能
        cmdclass={
            'build_ext': BuildExtension
        }
    )
"""
// csrc/deep_ep.cpp (在sources中)
#include "deep_ep.hpp"      // ← 引入csrc/deep_ep.hpp
#include "kernels/api.cuh"  // ← 引入csrc/kernels/api.cuh

编译器处理 #include 时会：
找到 deep_ep.hpp（在 include_dirs=['csrc/'] 中搜索）
把整个文件内容复制粘贴到这里
然后一起编译。所以sources 之外的 .hpp/.cuh 文件中的代码也会被编译，
只是是通过 #include 间接包含进来的。
"""
    setuptools.setup(name='deep_ep',
                     version='1.2.1' + revision,
                     packages=setuptools.find_packages(include=['deep_ep']),
                     ext_modules=[
                         CUDAExtension(name='deep_ep_cpp',
                                       include_dirs=include_dirs,
                                       library_dirs=library_dirs,
                                       sources=sources,
                                       extra_compile_args=extra_compile_args,
                                       extra_link_args=extra_link_args)
                     ],
                     cmdclass={'build_ext': BuildExtension})
