import time
import torch
import numpy as np
import pandas as pd

import contextlib

from dgl import DGLError
from torch.cuda import (
    profiler,
    synchronize,
    memory_allocated,
    max_memory_allocated,
    reset_peak_memory_stats,
)


def check_equal(first, second):
    if first is None or second is None:
        print("cannot guarantee correctness because of OOM")
    else:
        try:
            np.testing.assert_allclose(
                first.cpu().detach().numpy(), second.cpu().detach().numpy(), rtol=1e-3
            )
        except AssertionError as e:
            print("correctness check failed!")
            print(e)
            return False
        print("correctness check passed!")


def bench(
    net,
    net_params,
    tag="",
    nvprof=False,
    memory=False,
    repeat=1000,
    log=None,
    nvtx_flag=False,
):
    try:
        # warm up
        for i in range(5):
            net(*net_params)
        synchronize()
        memory_offset = memory_allocated()
        reset_peak_memory_stats()
        if nvtx_flag:
            import nvtx

            cm0 = torch.cuda.profiler.profile()
            cm1 = torch.autograd.profiler.emit_nvtx()
            cm2_0 = nvtx.annotate("forward", color="purple")
            with cm0:
                with cm1:
                    with cm2_0:
                        logits = net(*net_params)
            synchronize()

        if nvprof:
            profiler.start()
        start_time = time.time()
        for i in range(repeat):
            logits = net(*net_params)
        synchronize()
        if nvprof:
            profiler.stop()
        elapsed_time = (time.time() - start_time) / repeat * 1000
        print("{} elapsed time: {} ms/infer".format(tag, elapsed_time))
        log.at[tag, "time"] = elapsed_time
        if memory:
            max_mem_consumption = (max_memory_allocated() - memory_offset) / 1048576
            print("intermediate data memory usage: {} MB".format(max_mem_consumption))
            log.at[tag, "mem"] = max_mem_consumption
    except (RuntimeError, DGLError) as e:
        print("{} OOM".format(tag))
        print(e)
        return None
    except BaseException as e:
        print(e)
        raise
    return logits


def bench_with_bck_prop(
    net,
    net_params,
    tag="",
    nvprof=False,
    memory=False,
    repeat=1000,
    log=None,
    nvtx_flag=True,
):
    try:
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, total_steps=5 + repeat, max_lr=1e-3
        # )
        # warm up
        for i in range(5):
            optimizer.zero_grad()
            logits = net(*net_params)
            # if type(logits) is torch.Tensor:
            #     grad_logits = torch.rand_like(logits, requires_grad=False)
            #     logits.backward(grad_logits)
            # else:
            #     # grad_logits is a dict
            #     grad_logits = {
            #         k: torch.rand_like(v, requires_grad=False)
            #         for k, v in logits.items()
            #     }
            #     for k, v in grad_logits.items():
            #         logits[k].backward(v)
            if type(logits) is torch.Tensor:
                labels = torch.randint(0, logits.shape[-1], logits.shape[:-1]).cuda()
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
            else:
                labels = {
                    k: torch.randint(0, v.shape[-1], v.shape[:-1]).cuda()
                    for k, v in logits.items()
                }
                loss = dict()
                for k, v in labels.items():
                    loss[k] = torch.nn.functional.cross_entropy(logits[k], labels[k])
                for k, v in labels.items():
                    loss[k].backward(retain_graph=True)
            optimizer.step()
            # scheduler.step()
        synchronize()
        memory_offset = memory_allocated()
        reset_peak_memory_stats()
        if nvtx_flag:
            import nvtx

            cm0 = torch.cuda.profiler.profile()
            cm1 = torch.autograd.profiler.emit_nvtx()
            cm2_0 = nvtx.annotate("forward", color="purple")
            cm2_1 = nvtx.annotate("backward", color="green")
            cm2_2 = nvtx.annotate("optimizer_step", color="green")
            optimizer.zero_grad()
            with cm0:
                with cm1:
                    with cm2_0:
                        logits = net(*net_params)
                    # synchronize()
                    # fw_end_time = time.time()
                    if type(logits) is torch.Tensor:
                        loss = torch.nn.functional.cross_entropy(logits, labels)
                    else:
                        loss = dict()
                        for k, v in labels.items():
                            loss[k] = torch.nn.functional.cross_entropy(
                                logits[k], labels[k]
                            )
                    with cm2_1:
                        # if type(logits) is torch.Tensor:
                        #     logits.backward(grad_logits)
                        # else:
                        #     for k, v in grad_logits.items():
                        #         logits[k].backward(v)
                        if type(logits) is torch.Tensor:
                            loss.backward()
                        else:
                            for k, v in labels.items():
                                loss[k].backward(retain_graph=True)
                    with cm2_2:
                        optimizer.step()
                        # scheduler.step()
            synchronize()

        if nvprof:
            profiler.start()
        start_time = time.time()
        for i in range(repeat):
            optimizer.zero_grad()

            logits = net(*net_params)
            # synchronize()
            # fw_end_time = time.time()
            if type(logits) is torch.Tensor:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            else:
                loss = dict()
                for k, v in labels.items():
                    loss[k] = torch.nn.functional.cross_entropy(logits[k], labels[k])

                # if type(logits) is torch.Tensor:
                #     logits.backward(grad_logits)
                # else:
                #     for k, v in grad_logits.items():
                #         logits[k].backward(v)
            if type(logits) is torch.Tensor:
                loss.backward()
            else:
                for k, v in labels.items():
                    loss[k].backward(retain_graph=True)
            optimizer.step()
            # scheduler.step()
        synchronize()
        end_time = time.time()
        if nvprof:
            profiler.stop()
        elapsed_time = (end_time - start_time) / repeat * 1000
        # bw_elapsed_time = (end_time - fw_end_time) / repeat * 1000
        print("{} elapsed time: {} ms/training".format(tag, elapsed_time))
        # print("{} elapsed time: {} ms/backward".format(tag, bw_elapsed_time))
        log.at[tag, "train_time"] = elapsed_time
        # log.at[tag, "bw_time"] = bw_elapsed_time

        if memory:
            max_mem_consumption = (max_memory_allocated() - memory_offset) / 1048576
            print("intermediate data memory usage: {} MB".format(max_mem_consumption))
            log.at[tag, "train_mem"] = max_mem_consumption
    except (RuntimeError, DGLError) as e:
        print("{} OOM".format(tag))
        print(e)
        return None
    except BaseException as e:
        print(e)
        raise
    return logits


def init_log(tags, metrics):
    index = pd.MultiIndex.from_product([tags, metrics], names=["tag", "metric"])
    return pd.Series(np.zeros((len(tags) * len(metrics),)), index=index)


def empty_cache(func):
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        func(*args, **kwargs)

    return wrapper
