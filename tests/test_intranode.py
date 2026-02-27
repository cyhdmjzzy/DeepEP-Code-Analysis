import argparse
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, calc_diff, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
import test_low_latency


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int, buffer: deep_ep.Buffer,
              group: dist.ProcessGroup):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden  # 4096,  7168
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank  # 每个元素设置为rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None
    """
    PyTorch Tensor有两个重要概念：
    逻辑形状（shape）：你看到的维度
    内存布局（stride）：数据在内存中的存储顺序

    转置只是改变了视图, 不改变内存布局。
    原始张量（行主序, Row-major）。stride: (56, 1)  # 跨越56个元素到下一行, 跨越1个元素到下一列
    转置后, scales.T: (56, 4096)。stride: (1, 56)  # 只是改变了访问方式

    scales_T_contig = scales.T.contiguous() # 重新分配内存, 按照新的形状连续存储。stride: (4096, 1)。再次转置回来后, 但内存布局已经改变了！
    这里改变内存布局是因为：
        CUDA 内存合并访问（Coalesced Access）：
        同一个warp（32个线程）访问连续内存 → 快 ✓
        访问非连续内存 → 慢 ✗

    Stride 定义了"从一个元素到另一个元素需要跨越多少个内存位置"。
    对于 2D 张量 tensor[i][j]：内存地址 = base_address + i * stride[0] + j * stride[1]
    """
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
    # 这里所有scores都是正数（abs() + 1）, 所以在这个测试代码中实际上不会有 -1。
    # 但是在真实场景中, scores = gate_network(hidden_states)可能会产生-1, 比如padding部分的token, 所以需要后面再rank_idx中标记为-1
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    # 将每个token分配到的expert的 expert ID 映射到 rank ID（确定该 expert 在哪个 GPU/进程上）
    # (num_experts // num_ranks)表示每个rank中有多少个expert
    # 每个token的每个副本要发送到的rank ID
    rank_idx = topk_idx // (num_experts // num_ranks)  # rank_idx.shape: (num_tokens, num_topk) 每个token选择的topk个expert的rank编号。
    rank_idx = rank_idx.to(torch.int64)
    """
    Python中 -1//32 = -1, 但是在某些语言中可能是 -1//32 = 0, 导致错误！所以这里使用topk_idx判断无效的expert的位置, 然后设置rank_idx为-1

    在 MoE 系统中, 可能有一些 token 没有分配到足够的 experts（padding）, 这里是将无效的 token-expert 对应的 rank_idx 也标记为无效（-1）
    跨平台兼容性：不同系统/语言对负数整除的行为可能不同。
    动态 Sparse MoE：某些 token 可能不需要所有 top-k 个 experts；
        # 假设 num_topk = 8, 但某个 token 只需要 5 个 experts
        topk_idx[token_i] = [45, 123, 67, 89, 234, -1, -1, -1]
    为了平衡负载, 可能限制某些 expert 的使用：
        假设每个expert最多处理100个tokens, 而第128个expert处理的token数已经满了, 
        就将选择 expert 128 的 token 标记为无效（-1）
    训练时随机 dropout 某些 experts；
    Top-k 选择后的过滤。某些实现中, 分数太低的 expert 会被过滤, 比如TopK的expert的score都很低；

    """
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks) # 对 rank_idx 每一行（每个 token）进行去重 + 频率排序 + 原地修改。

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)  # 每个rank的num_tokens_per_expert都进行一个全局累加, 得到每个expert被多少个tokens选择。

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        # (rank_idx == i)是布尔Tensor。max 实际上是检查该行是否有任何 True。[0]的作用：取返回值的第一个元素（values）
        # token_sel[j] = True：token j 需要与 rank i 通信。否则不需要通信。
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item() # 需要参与与 rank i 通信的token数
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1] # 需要参与与 rank i 通信的token的索引排在前面, 不需要参与的排在后面。
        tokens[:count] = torch.sort(tokens[:count])[0]  # 对需要参与与 rank i 通信的token的索引进行排序, 排序后, 不需要参与的排在后面也不用排序。
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda') # 需要参与与 rank i 通信的token的索引在token_idx_in_rank[i]中标记为对应的位置, 按[0, 1, 2, 3, ..., count-1]标记, 没有标记的就是-1。
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0  # is_token_in_rank[j][i] = True：token j 需要与 rank i 通信。否则不需要通信。
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)  # 每个rank的num_tokens_per_rank都进行一个全局累加, 得到每个rank需要处理的token数量。

    """
    ref_num_tokens_per_rank：每个rank需要处理的token数量。 shape: (num_ranks)
    ref_num_tokens_per_expert：每个expert被多少个tokens选择。 shape: (num_experts)
    ref_is_token_in_rank：布尔矩阵, 标记每个token是否需要与每个rank通信。 shape: (num_tokens, num_ranks)
    """
    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    # torch.allclose() 的作用：检查两个张量是否"足够接近"（考虑浮点数精度误差）
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    # 返回对buffer.get_dispatch_layout(topk_idx, num_experts)进行基准测试的平均时间、最小时间和最大时间。
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    group.barrier()  # 同步屏障。同步所有进程, 确保所有进程都执行完前面的代码。
    time.sleep(1)  # 确保GPU状态完全稳定。为了保证测试的可靠性, 需要等待一段时间, 让GPU完成所有操作。

    # Config
    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    for previous_mode in (False, True):
        for async_mode in (False, True):
            for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x, x_e4m3)):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            flush=True,
                            end='')
                    dispatch_args = {
                        'x': current_x,   # shape: (num_tokens, hidden)
                        'num_tokens_per_rank': num_tokens_per_rank,  # shape: (num_ranks)
                        'is_token_in_rank': is_token_in_rank,  # shape: (num_tokens, num_ranks)
                        'num_tokens_per_expert': num_tokens_per_expert,  # shape: (num_experts)
                        'config': config,  # shape: (num_sms, nvl_chunk_size, nvl_buffer_size)
                        'async_finish': async_mode  # bool
                    }
                    # 如果不用topk_idx, 实际上就不能在dispatch时候使用expert模型计算token的输出了, 
                    # 因为只知道每个token发往哪个rank, 但是token到达rank之后还是不知道具体各个token发往哪些expert, 于是expert也无法获得token数据进行计算了。
                    # 而这种方式只是用于测试环境中, 实际生产环境中还是会使用topk_idx。
                    if with_topk:
                        dispatch_args.update({
                            'topk_idx': topk_idx,
                            'topk_weights': topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        })
                    if previous_mode:
                        # 注意：整个DeepEP中，默认流是compute_stream！不信你全局搜：at::cuda::getCurrentCUDAStream() 看看返回的stream是如何命名的（是"compute_stream"）。
                        dispatch_args.update({'previous_event': buffer.capture()})
                    # recv_x.shape: (num_recv_tokens, hidden)  dispatch中, 从别的rank中接收到的token数据。
                    # recv_topk_idx.shape: (num_recv_tokens, num_topk)  当前rank接收到的每个token选择的topk个expert的编号。
                    # recv_topk_weights.shape: (num_recv_tokens, num_topk) 当前rank接收到的每个token选择的topk个expert的权重。
                    # recv_num_tokens_per_expert_list.shape: (num_local_experts) 当前rank中的每个expert的token数。实际上每个rank的expert数量是一样的。
                    # handle:
                    # event.shape: (num_recv_tokens, num_topk)
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle, event = buffer.dispatch(**dispatch_args)
                    # 每次执行完buffer.dispatch，都会在dispatch函数内部的最后切换到compute_stream，也就是默认流。
                    # event 是记录在 comm_stream 中的事件，只在异步模式下才有。
                    #
                    # 注意, async_mode 表示的异步还是同步, 指的是buffer.dispatch内部关于compute_stream是否等待comm_stream完成。
                    # 如果async_mode是异步, 说明buffer.dispatch里面只是向comm_stream提交了一些CUDA任务, 但是并没有实际执行完这些CUDA任务。
                    # 这些任务都在comm_stream中排布, 包括：intranode::cached_notify_dispatch、intranode::notify_dispatch和intranode::dispatch。
                    #
                    # 当前流是compute_stream，需要等待comm_stream中的事件event记录之前的任务完成。因为只有comm_stream中的任务都执行完了，才能保证通信写入的数据都写好了。
                    event.current_stream_wait() if async_mode else ()
                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    rank_prefix_matrix = handle[0]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                        0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if current_x is not x_pure_rand:
                        check_data(recv_x, rank_prefix_matrix)
                    recv_topk_weights_clone = None
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) |
                                ((recv_topk_idx >= 0) &
                                 (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):  # i 是当前rank的全局expert编号
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()
                        if current_x is not x_pure_rand:
                            # 只在recv_topk_weights的无效expert位置上填充最大权重。
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(
                                dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, rank_prefix_matrix)

                    # Test `num_worst_tokens != 0`
                    if with_topk:
                        num_worst_tokens = num_tokens * num_ranks
                        dispatch_args.update({'num_worst_tokens': num_worst_tokens})
                        recv_worst_x, recv_worst_topk_idx, recv_worst_topk_weights, empty_list, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_worst_x = per_token_cast_back(*recv_worst_x) if isinstance(recv_worst_x, tuple) else recv_worst_x
                        assert len(empty_list) == 0
                        assert num_worst_tokens == recv_worst_x.size(0)
                        assert num_worst_tokens == recv_worst_topk_idx.size(0)
                        assert num_worst_tokens == recv_worst_topk_weights.size(0)
                        assert torch.equal(recv_x, recv_worst_x[:recv_x.size(0)])
                        assert torch.equal(recv_topk_idx, recv_worst_topk_idx[:recv_x.size(0)])
                        assert torch.equal(recv_topk_weights_clone, recv_worst_topk_weights[:recv_x.size(0)])
                        assert torch.all(recv_worst_topk_idx[recv_x.size(0):] == -1).item()

                    # Test cached dispatch (must without top-k staffs)
                    if not with_topk:
                        dispatch_args = {'x': current_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                        if previous_mode:
                            dispatch_args.update({'previous_event': buffer.capture()})
                        recv_x, _, _, _, _, event = buffer.dispatch(**dispatch_args)
                        event.current_stream_wait() if async_mode else ()
                        recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x
                        if current_x is not x_pure_rand:
                            check_data(recv_x, rank_prefix_matrix)

                    # Test combine
                    combine_args = {'x': recv_x, 'handle': handle, 'config': config, 'async_finish': async_mode}
                    if with_topk:
                        combine_args.update({'topk_weights': recv_topk_weights})
                    if previous_mode:
                        combine_args.update({'previous_event': buffer.capture()})
                    combined_x, combined_topk_weights, event = buffer.combine(**combine_args)
                    event.current_stream_wait() if async_mode else ()
                    check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x
                                                                       is x_pure_rand) else (combined_topk_weights /
                                                                                             is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        for nvl_chunk_size in tuple(range(4, 33, 2)) + (0, ):
            if nvl_chunk_size > 0:
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            else:
                # Test default config as well
                deep_ep.Buffer.set_num_sms(num_sms)
                config = deep_ep.Buffer.get_dispatch_config(num_ranks)
            tune_args = {'x': current_x, 'handle': handle, 'config': config}
            t = bench(lambda: buffer.dispatch(**tune_args))[0]  # noqa: B023
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)
            if local_rank == 0:
                print(
                    f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                    f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',
                    flush=True)
        if local_rank == 0:
            print(
                f'[tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
                flush=True)
            print('', flush=True)

        # Gather the best config from rank 0 and the first test setting
        if best_dispatch_results is None:
            best_dispatch_results = torch.tensor([best_results[0], best_results[1]], dtype=torch.int32, device='cuda')
            all_best_fp8_results_list = [torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size)

    dispatch_args = {
        'x': x,
        'num_tokens_per_rank': num_tokens_per_rank,
        'is_token_in_rank': is_token_in_rank,
        'num_tokens_per_expert': num_tokens_per_expert,
        'config': dispatch_config if dispatch_config is not None else config
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0, ):
        if nvl_chunk_size > 0:
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        else:
            # Test default config as well
            deep_ep.Buffer.set_num_sms(num_sms)
            config = deep_ep.Buffer.get_combine_config(num_ranks)
        tune_args = {'x': recv_x, 'handle': handle, 'config': config}
        t = bench(lambda: buffer.combine(**tune_args))[0]  # noqa: B023
        if local_rank == 0:
            print(
                f'[tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                f'{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',
                flush=True)
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)

    if local_rank == 0:
        print(
            f'[tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}: {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
            flush=True)
        print('', flush=True)


# spawn 会自动做以下事情：
#   自动添加第一个参数 local_rank（进程ID, 从 0 到 nprocs-1）
#   然后才添加 args 元组中的参数
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    test_ll_compatibility, num_rdma_bytes = False, 0
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(ll_num_tokens, ll_hidden, num_ranks, ll_num_experts)

    buffer = deep_ep.Buffer(group,
                            int(2e9),
                            num_rdma_bytes,
                            low_latency_mode=test_ll_compatibility,
                            num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1),
                            explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl,
                            use_fabric=args.use_fabric)
    torch.manual_seed(rank)

    for i in (24, ):  # 24是要使用的流多处理器（SM）数量
        test_main(args, i, local_rank, num_ranks, rank, buffer, group)
        if local_rank == 0:
            print('', flush=True)

    # Test compatibility with low latency functions
    if test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test intranode EP kernels')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts (default: 256)')
    parser.add_argument('--allow-mnnvl', action="store_true", help='Enable MNNVL support')
    parser.add_argument('--use-fabric', action="store_true", help='Enable fabric mode')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
