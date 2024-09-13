import torch
import triton
import triton.language as tl
from utility.stop_watch import stop_watch


@triton.jit
def add_kernel(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    output_ptr: torch.Tensor,
    n_elements: int,
    BLOCK_SIZE: int,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


@stop_watch
def torch_add(x: torch.Tensor, y: torch.Tensor):
    return x + y


@stop_watch
def triton_add(x: torch.Tensor, y: torch.Tensor):
    return add(x, y)

# 一つだけ実行
if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = torch_add(x, y)
    output_triton = triton_add(x, y)
    print(output_torch)
    print(output_triton)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(output_torch - output_triton))}"
    )

# ベンチマーク

# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         x_names=["size"],  # Argument names to use as an x-axis for the plot.
#         x_vals=[
#             2**i for i in range(12, 28, 1)
#         ],  # Different possible values for `x_name`.
#         x_log=True,  # x axis is logarithmic.
#         line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
#         line_vals=["triton", "torch"],  # Possible values for `line_arg`.
#         line_names=["Triton", "Torch"],  # Label name for the lines.
#         styles=[("blue", "-"), ("green", "-")],  # Line styles.
#         ylabel="GB/s",  # Label name for the y-axis.
#         plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
#         args={},  # Values for function arguments not in `x_names` and `y_name`.
#     )
# )
# def benchmark(size, provider):
#     x = torch.rand(size, device="cuda", dtype=torch.float32)
#     y = torch.rand(size, device="cuda", dtype=torch.float32)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == "torch":
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
#     if provider == "triton":
#         ms, min_ms, max_ms = triton.testing.do_bench(
#             lambda: add(x, y), quantiles=quantiles
#         )

#     def gbps(ms):
#         return 3 * x.numel() * x.element_size() / ms * 1e-06

#     return gbps(ms), gbps(max_ms), gbps(min_ms)


# if __name__ == "__main__":
#     benchmark.run(print_data=True)
