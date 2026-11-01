"""Hyena (Poli et al., 2023): long convolutions and gates in attention's place in the block.

For each channel, the output at token t is a weighted sum of the tokens up to t, as with attention. Here the weight
on token s comes from a filter, by how far back s is, and from two gates read off the tokens themselves:

    y_t = x2_t * (sum over s <= t of h_(t-s) * x1_s * v_s  +  b * x1_t * v_t)

A sum over every token before, weighted by distance, is a convolution, and the fast Fourier transform computes one
for every t at once in O(T log T) operations, where attention's table of scores takes O(T^2).
"""

import torch


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
