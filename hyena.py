"""Hyena (Poli et al., 2023): long convolutions and gates in attention's place in the block.

For each channel, the output at token t is a weighted sum of the tokens up to t, as with attention. Here the weight
on token s comes from a filter, by how far back s is, and from two gates read off the tokens themselves:

    y_t = x2_t * (sum over s <= t of h_(t-s) * x1_s * v_s  +  b * x1_t * v_t)

A sum over every token before, weighted by distance, is a convolution, and the fast Fourier transform computes one
for every t at once in O(T log T) operations, where attention's table of scores takes O(T^2). This is the paper's
operator of order 2, as its authors' code computes it: one long filter between two gates.
"""

import math

import torch
import torch.nn as nn


def fft_conv(u, h):
    # u is (..., C, T), h is (C, T): for each channel, y_t = sum over s <= t of h_(t-s) u_s, for every t at once.
    T = u.size(-1)
    # The FFT's convolution is circular: a filter running off the end wraps round to the start. With the inputs
    # padded with zeros to 2T, what wraps round lands past the T outputs we keep, so those are the causal sums.
    n = 2 * T
    # In float32, as the FFT has no bfloat16 version, and its sums of thousands of terms need the precision.
    u_f = torch.fft.rfft(u.float(), n=n)
    h_f = torch.fft.rfft(h.float(), n=n)
    # Convolving in time is multiplying in frequency: one product per frequency, instead of a sum per output.
    return torch.fft.irfft(u_f * h_f, n=n)[..., :T].type_as(u)


def direct_conv(u, h):
    # The same sums, written out in O(T^2): for the tests, and to read what fft_conv computes.
    y = torch.zeros_like(u)
    for t in range(u.size(-1)):
        # Token s is t - s back, so its weight is h_(t-s): the filter's first t + 1 taps, reversed.
        y[..., t] = (u[..., : t + 1] * h[..., : t + 1].flip(-1)).sum(-1)
    return y


class Sine(nn.Module):
    # sin(w x), with a frequency w learned for each hidden unit, starting at 14. A network of smooth activations
    # learns smooth functions first; with sines at a high frequency, a filter can have sharp features too.
    def __init__(self, width, freq=14.0):
        super().__init__()
        self.freq = nn.Parameter(torch.full((1, width), freq))

    def forward(self, x):
        return torch.sin(self.freq * x)


class HyenaFilter(nn.Module):
    # A filter as long as the window for each channel, made by a small network from each position: h_t =
    # window(t) * FFN(encoding(t)), the paper's equation 7. Its weights don't grow with the window: an implicit filter.
    def __init__(self, channels, length, bands=16, width=64):
        super().__init__()
        self.length = length
        t = torch.arange(length)
        # Each position as 33 numbers: where it is in the window, from 0 to 1, and a cosine and a sine of it at 16
        # frequencies, from 0 to 15 turns over the window. Fixed, as in the paper's equation (the authors' language
        # models let training nudge them too).
        turns = torch.linspace(1e-4, bands - 1, bands)
        angles = 2 * math.pi * t[:, None] / length * turns
        self.register_buffer("encoding", torch.cat([t[:, None] / (length - 1), angles.cos(), -angles.sin()], dim=1))
        # One sine module three times: its 64 frequencies are shared, as in the authors' code.
        sine = Sine(width)
        self.ffn = nn.Sequential(nn.Linear(2 * bands + 1, width), sine, nn.Linear(width, width), sine,
                                 nn.Linear(width, width), sine, nn.Linear(width, channels, bias=False))
        # Each channel's filter fades as exp(-rate t), t from 0 to 1 over the window. The fastest falls to 1% by 30%
        # of the window and the slowest is still at about 5% at its end: some channels look close, others far.
        self.register_buffer("t", t / (length - 1))
        self.register_buffer("rates", torch.linspace(math.log(100) / 1.5, math.log(100) / 0.3, channels))
        # The skip: each channel also passes its own token straight through, times b.
        self.bias = nn.Parameter(torch.randn(channels))

    def forward(self, T):
        # The first T taps of every channel's filter, (C, T): tap 0 weighs a token's own value, tap 1 the token before.
        h = self.ffn(self.encoding[:T]) * torch.exp(-self.t[:T, None] * self.rates)
        return h.T


class Hyena(nn.Module):
    # The order-2 Hyena operator, where a block had attention: each token projected to x1, x2 and v, then for each
    # channel a long convolution between two gates. As matrices, y = D(x2) (S(h) + b I) D(x1) v, where D(x) has x on
    # its diagonal and S(h) has h_(t-s) at row t, column s, for s <= t.
    def __init__(self, config, layer):
        super().__init__()
        C = config.n_embd
        self.layer = layer
        self.in_proj = nn.Linear(C, 3 * C, bias=config.bias)
        # A filter three tokens long on each of the 3C channels: a token's projections mixed with the two tokens'
        # before it, the same channel's. Padded by 2 at both ends; the first T outputs end at each token.
        self.short_conv = nn.Conv1d(3 * C, 3 * C, kernel_size=3, padding=2, groups=3 * C)
        self.filter = HyenaFilter(C, config.block_size)
        self.o_proj = nn.Linear(C, C, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cache=None):
        B, T, C = x.shape  # batch, tokens, channels
        p = self.in_proj(x).transpose(1, 2)  # (B, 3C, T): channels first, as convolutions take them
        before = None if cache is None else cache.hyena[self.layer]
        if before is not None:
            # Writing with a cache: the last two tokens' projections, for the short filter.
            p = torch.cat([before["p"], p], dim=-1)
        u = self.short_conv(p)[..., : p.size(-1)][..., -T:]
        x1, x2, v = u.split(C, dim=1)
        z = x1 * v  # the first gate: each token's values, scaled channel by channel by its x1
        if before is not None:
            z = torch.cat([before["z"], z], dim=-1)
        # The filter doesn't depend on the tokens: with a cache, it's made once, for the whole window.
        h = before["h"] if before is not None else self.filter(self.filter.length if cache is not None else T)
        if before is not None and T == 1:
            # One new token: the gated values of all the tokens so far, each weighted by the filter at its
            # distance, the newest by h_0. A sum over the cache, as attention's is.
            y = (z * h[:, : z.size(-1)].flip(-1)).sum(-1, keepdim=True)
        else:
            y = fft_conv(z, h[:, : z.size(-1)])[..., -T:]
        y = y + self.filter.bias[:, None] * z[..., -T:]  # the skip
        y = x2 * y  # the second gate
        if cache is not None:
            # One number a channel for each token, z, where attention keeps a key and a value.
            cache.hyena[self.layer] = {"p": p[..., -2:], "z": z, "h": h}
        return self.resid_dropout(self.o_proj(y.transpose(1, 2)))

    def streams(self, x):
        # x1, x2 and v for whole sequences, (B, C, T) each: the projections, after the short filter.
        u = self.short_conv(self.in_proj(x).transpose(1, 2))[..., : x.size(1)]
        return u.split(x.size(-1), dim=1)

    def matrices(self, x):
        # The T x T matrix that each channel applies to its values, for one sequence, (C, T, T). Row t, column s
        # is x2_t (h_(t-s) + b if s = t) x1_s, and 0 past the diagonal. Hyena never builds them: they're for looking
        # at, as attention's weights are, and for the tests.
        x1, x2, _ = (s[0] for s in self.streams(x[:1]))
        T = x.size(1)
        h = self.filter(T)
        t = torch.arange(T, device=x.device)
        lag = t[:, None] - t[None, :]  # how far back column s is from row t
        S = h[:, lag.clamp(min=0)] * (lag >= 0)  # S(h): the filter by distance, below the diagonal and on it
        S = S + torch.diag_embed(self.filter.bias[:, None].expand(-1, T))  # the skip, on the diagonal
        return x2[:, :, None] * S * x1[:, None, :]
