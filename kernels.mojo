from std.math import ceildiv, erf, exp, sqrt
from std.memory import unsafe_stack_allocation
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from max.gpu import block_dim, block_idx, global_idx, thread_idx
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives import warp
from max.gpu.sync import barrier

# Our GPT writing a token at a time on the GPU, as Mojo kernels: episode one's blocks, with LayerNorm, GELU and a
# learned embedding per position. mojo_model.py hands it the weights and calls it from Python, a token at a time.
# The kernels share the work out by warps, groups of 32 threads that run in step, which Apple calls SIMD groups.

comptime Floats = Pointer[Float32, MutAnyOrigin]
comptime MAX_CONTEXT = 1024
comptime MAX_HEAD = 128
comptime THREADS = 256
comptime WARPS = THREADS // WARP_SIZE


def embed(x: Floats, tokens: Floats, positions: Floats, token: Int32, pos: Int32, width: Int32):
    # Thread c writes channel c: the token's embedding plus its position's.
    var c = global_idx.x
    var C = Int(width)
    if c < C:
        x[unsafe_offset=c] = tokens[unsafe_offset=Int(token) * C + c] + positions[unsafe_offset=Int(pos) * C + c]


def linear[
    norm: Bool, gelu: Bool, add: Bool
](y: Floats, w: Floats, b: Floats, x: Floats, gain: Floats, shift: Floats, rows: Int32, cols: Int32):
    # y = W x + b, with W stored as PyTorch's Linear stores it: a row of `cols` weights for each of the `rows` outputs.
    # A warp for each output: its 32 lanes read the row side by side, 4 numbers each at a time, so that together
    # they read 128 neighbouring numbers, and warp.sum adds up their 32 dot products.
    var r = global_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE
    var n = Int(cols)
    if r < Int(rows):
        # With norm, x goes through a LayerNorm first, with its gain and shift. Each warp works out x's mean and
        # standard deviation for itself, two more reads of a few hundred numbers, which saves a kernel.
        var mean: Float32 = 0
        var std: Float32 = 1
        comptime if norm:
            var total: Float32 = 0
            for i in range(lane, n, WARP_SIZE):
                total += x[unsafe_offset=i]
            mean = warp.sum(total) / Float32(n)
            var squares: Float32 = 0
            for i in range(lane, n, WARP_SIZE):
                squares += (x[unsafe_offset=i] - mean) ** 2
            std = sqrt(warp.sum(squares) / Float32(n) + 1e-5)
        var part: Float32 = 0
        for c in range(lane * 4, n, WARP_SIZE * 4):
            var xs = x.unsafe_load[width=4](c)
            comptime if norm:
                xs = (xs - mean) / std * gain.unsafe_load[width=4](c) + shift.unsafe_load[width=4](c)
            part += (w.unsafe_load[width=4](r * n + c) * xs).reduce_add()
        var acc = warp.sum(part) + b[unsafe_offset=r]
        # norm, gelu and add are parameters, known when the kernel compiles: each mix of them is its own kernel.
        comptime if gelu:
            acc = 0.5 * acc * (1 + erf(acc / sqrt(Float32(2))))
        if lane == 0:
            comptime if add:
                y[unsafe_offset=r] += acc
            else:
                y[unsafe_offset=r] = acc


def attention(y: Floats, cache: Floats, pos: Int32, head_size: Int32, width: Int32):
    # A block of THREADS threads for each head: WARPS warps, which share out the positions, a position at a time.
    # The cache has a row for each position, its query, its key and its value, side by side: 3 C numbers.
    var head = block_idx.x
    var lane = thread_idx.x % WARP_SIZE
    var group = thread_idx.x // WARP_SIZE
    var hs = Int(head_size)
    var C = Int(width)
    var R = 3 * C
    var n = Int(pos) + 1
    # Memory the block's threads share: the scores, and each warp's share of the sums.
    var scores = unsafe_stack_allocation[MAX_CONTEXT, Float32, address_space=.SHARED]()
    var shares = unsafe_stack_allocation[WARPS * MAX_HEAD, Float32, address_space=.SHARED]()
    # A score for each position: the warp's lanes split the dot product of the query and that position's key.
    for t in range(group, n, WARPS):
        var part: Float32 = 0
        for i in range(lane, hs, WARP_SIZE):
            part += cache[unsafe_offset=Int(pos) * R + head * hs + i] * cache[unsafe_offset=t * R + C + head * hs + i]
        var dot = warp.sum(part)
        if lane == 0:
            scores[unsafe_offset=t] = dot / sqrt(Float32(hs))
    barrier()
    # The softmax: the largest score, each warp's then the block's, and then the exponentials and their total.
    var top: Float32 = -3.0e38
    for t in range(thread_idx.x, n, THREADS):
        top = max(top, scores[unsafe_offset=t])
    top = warp.max(top)
    if lane == 0:
        shares[unsafe_offset=group] = top
    barrier()
    for g in range(WARPS):
        top = max(top, shares[unsafe_offset=g])
    barrier()
    var total: Float32 = 0
    for t in range(thread_idx.x, n, THREADS):
        var p = exp(scores[unsafe_offset=t] - top)
        scores[unsafe_offset=t] = p
        total += p
    total = warp.sum(total)
    if lane == 0:
        shares[unsafe_offset=group] = total
    barrier()
    total = 0
    for g in range(WARPS):
        total += shares[unsafe_offset=g]
    barrier()
    # The output: each warp adds up the values of its positions, its lanes side by side across the channels, and
    # then a thread for each channel adds up the warps' shares.
    for d in range(lane, hs, WARP_SIZE):
        var acc: Float32 = 0
        for t in range(group, n, WARPS):
            acc += scores[unsafe_offset=t] * cache[unsafe_offset=t * R + 2 * C + head * hs + d]
        shares[unsafe_offset=group * MAX_HEAD + d] = acc
    barrier()
    if thread_idx.x < hs:
        var acc: Float32 = 0
        for g in range(WARPS):
            acc += shares[unsafe_offset=g * MAX_HEAD + thread_idx.x]
        y[unsafe_offset=head * hs + thread_idx.x] = acc / total


def ptr(mut buffer: DeviceBuffer[DType.float32]) -> Floats:
    # A pointer into a GPU buffer, to hand to the kernels.
    return buffer.unsafe_ptr().as_unsafe_any_origin()


@fieldwise_init
struct GPT(Movable, Writable):
    var ctx: DeviceContext
    var weights: DeviceBuffer[DType.float32]
    var cache: DeviceBuffer[DType.float32]
    var x: DeviceBuffer[DType.float32]
    var y: DeviceBuffer[DType.float32]
    var hidden: DeviceBuffer[DType.float32]
    var logits: DeviceBuffer[DType.float32]
    var vocab: Int
    var context: Int
    var layers: Int
    var heads: Int
    var width: Int
    var read: Int

    @staticmethod
    def py_init(out self: GPT, args: PythonObject, kwargs: PythonObject) raises:
        # From Python: GPT(weights, vocab, context, layers, heads, width), the weights one NumPy array of float32.
        var weights = args[0]
        var vocab = Int(py=args[1])
        var context = Int(py=args[2])
        var layers = Int(py=args[3])
        var heads = Int(py=args[4])
        var width = Int(py=args[5])
        if context > MAX_CONTEXT or width // heads > MAX_HEAD or width % 4:
            raise Error("attention holds 1024 scores and heads of up to 128 channels, and linear reads 4 at a time")
        var ctx = DeviceContext()
        # The weights go to the GPU once, and stay there.
        var buffer = ctx.enqueue_create_buffer[DType.float32](Int(py=weights.size))
        ctx.enqueue_copy(dst_buf=buffer, src_ptr=weights.ctypes.data.unsafe_get_as_pointer[DType.float32]())
        self = Self(
            ctx=ctx,
            weights=buffer,
            cache=ctx.enqueue_create_buffer[DType.float32](layers * context * 3 * width),
            x=ctx.enqueue_create_buffer[DType.float32](width),
            y=ctx.enqueue_create_buffer[DType.float32](width),
            hidden=ctx.enqueue_create_buffer[DType.float32](4 * width),
            logits=ctx.enqueue_create_buffer[DType.float32](vocab),
            vocab=vocab,
            context=context,
            layers=layers,
            heads=heads,
            width=width,
            read=0,
        )
        self.ctx.synchronize()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("GPT(", self.layers, " blocks of ", self.width, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @staticmethod
    def step(
        py_self: PythonObject, token: PythonObject, pos: PythonObject, logits: PythonObject
    ) raises -> PythonObject:
        # From Python: gpt.step(token, pos, logits), which writes the next token's logits into a NumPy array.
        var self = py_self.downcast_value_ptr[Self]()
        self[].forward(Int(py=token), Int(py=pos), logits.ctypes.data.unsafe_get_as_pointer[DType.float32]())
        return PythonObject(None)

    def next(mut self, size: Int) -> Floats:
        # The next tensor in the weights, in mojo_model.py's order: forward reads them from the start each time.
        var start = ptr(self.weights).unsafe_offset(self.read)
        self.read += size
        return start

    def run_linear[
        norm: Bool, gelu: Bool, add: Bool
    ](mut self, y: Floats, w: Floats, b: Floats, x: Floats, gain: Floats, shift: Floats, rows: Int, cols: Int) raises:
        self.ctx.enqueue_function[linear[norm, gelu, add]](
            y,
            w,
            b,
            x,
            gain,
            shift,
            Int32(rows),
            Int32(cols),
            grid_dim=ceildiv(rows * WARP_SIZE, THREADS),
            block_dim=THREADS,
        )

    def run_layer[
        norm: Bool = False, gelu: Bool = False, add: Bool = False
    ](mut self, y: Floats, x: Floats, rows: Int, cols: Int) raises:
        # One of the block's Linear layers: if it has a LayerNorm before it, the norm's gain and shift, and then its
        # weights and its bias, are the next tensors. Without a norm, the kernel doesn't read gain and shift.
        var gain = ptr(self.weights)
        var shift = gain
        comptime if norm:
            gain = self.next(cols)
            shift = self.next(cols)
        var w = self.next(rows * cols)
        var b = self.next(rows)
        self.run_linear[norm, gelu, add](y, w, b, x, gain, shift, rows, cols)

    def forward(mut self, token: Int, pos: Int, logits: Floats) raises:
        var C = self.width
        var T = self.context
        var x = ptr(self.x)
        var y = ptr(self.y)
        var hidden = ptr(self.hidden)
        self.read = 0
        var tokens = self.next(self.vocab * C)
        var no_bias = self.next(self.vocab)
        var positions = self.next(T * C)
        self.ctx.enqueue_function[embed](
            x, tokens, positions, Int32(token), Int32(pos), Int32(C), grid_dim=ceildiv(C, 128), block_dim=128
        )
        for layer in range(self.layers):
            # This layer's cache: a row for each position, T of them.
            var cache = ptr(self.cache).unsafe_offset(layer * T * 3 * C)
            # The LayerNorm, and the query, key and value as one Linear, which writes this token's row of the cache.
            self.run_layer[norm=True](cache.unsafe_offset(pos * 3 * C), x, 3 * C, C)
            var hs = C // self.heads
            self.ctx.enqueue_function[attention](
                y, cache, Int32(pos), Int32(hs), Int32(C), grid_dim=self.heads, block_dim=THREADS
            )
            # The output projection, and below it the MLP's down projection, add into the residual stream, x.
            self.run_layer[add=True](x, y, C, C)
            self.run_layer[norm=True, gelu=True](hidden, x, 4 * C, C)
            self.run_layer[add=True](x, hidden, C, 4 * C)
        # The final LayerNorm, and the head: its weights are the token embeddings, with the zeros for its bias.
        var gain = self.next(C)
        var shift = self.next(C)
        self.run_linear[True, False, False](ptr(self.logits), tokens, no_bias, x, gain, shift, self.vocab, C)
        # Each enqueue so far only queued work for the GPU. The copy goes after it, and synchronize waits for all of it.
        self.ctx.enqueue_copy(dst_ptr=logits, src_buf=self.logits)
        self.ctx.synchronize()


@export
def PyInit_kernels() abi("C") -> PythonObject:
    # What Python runs when it imports this module: it makes the module, with our GPT type, its init and its step.
    try:
        var module = PythonModuleBuilder("kernels")
        _ = module.add_type[GPT]("GPT").def_py_init[GPT.py_init]().def_method[GPT.step]("step")
        return module.finalize()
    except e:
        abort(String("failed to make the kernels module: ", e))
