from std.time import perf_counter_ns

from max.gpu import global_idx
from max.gpu.host import DeviceContext

# What it costs to hand the GPU a kernel, and to wait for it. A kernel that does next to nothing goes to the GPU over
# and over: 57 at a time, as many as a token takes in kernels.mojo, then 32, each lot followed by a synchronize.

comptime ROUNDS = 1000


def touch(x: Pointer[Float32, MutAnyOrigin]):
    if global_idx.x == 0:
        x[unsafe_offset=0] += 1


def main() raises:
    var ctx = DeviceContext()
    var buffer = ctx.enqueue_create_buffer[DType.float32](1)
    var x = buffer.unsafe_ptr().as_unsafe_any_origin()
    # A first launch compiles the kernel for the GPU: it goes before the clock starts.
    ctx.enqueue_function[touch](x, grid_dim=1, block_dim=32)
    ctx.synchronize()
    for kernels in [57, 32]:
        var handing = 0
        var start = perf_counter_ns()
        for _ in range(ROUNDS):
            var queued = perf_counter_ns()
            for _ in range(kernels):
                ctx.enqueue_function[touch](x, grid_dim=1, block_dim=32)
            handing += Int(perf_counter_ns() - queued)
            ctx.synchronize()
        var total = Int(perf_counter_ns() - start)
        print(kernels, "kernels, then synchronize:", total // ROUNDS // 1000, "µs")
        print(
            "  handing them to the GPU:", handing // ROUNDS // 1000, "µs,", handing // ROUNDS // kernels, "ns a kernel"
        )
