""" 
Hilbert curve, in any dimension, parallelised across the inputs.
This is vastly a rewritting of the code of Nicolas Chopin at https://www.github.com/nchopin/particles
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def invlogit(x):
    return 1. / (1. + jnp.exp(-x))


@jax.jit
def hilbert_sort(x):
    d = x.shape[1]
    if x.ndim == 1:
        return jnp.argsort(x, axis=0)
    elif d == 1:
        return jnp.argsort(x[:, 0], axis=0)

    scaled_x = (x - jnp.mean(x, axis=0, keepdims=True)) / jnp.std(x, axis=0, keepdims=True)
    xs = invlogit(scaled_x)
    maxint = math.floor(2 ** (62 / d))
    xint = jnp.floor(xs * maxint).astype(jnp.int64)
    Hilbert_to_int_spec = lambda z: Hilbert_to_int(z, maxint)
    hilbert_array = jax.vmap(Hilbert_to_int_spec)(xint)
    return jnp.argsort(hilbert_array)


@partial(jax.jit, static_argnums=(1,))
def Hilbert_to_int(coords, max_int):
    nD = coords.shape[0]
    coord_chunks = unpack_coords(coords, max_int)
    nChunks = coord_chunks.shape[0]
    mask = 2 ** nD - 1

    def body(carry, coord_chunks_j):
        start, end = carry
        i = gray_decode_travel(start, end, mask, coord_chunks_j)
        start, end = child_start_end(start, end, mask, i)
        return (start, end), i

    _, index_chunks = jax.lax.scan(body, initial_start_end(nChunks, nD), coord_chunks)
    return pack_index(index_chunks, nD)


@jax.jit
def initial_start_end(nChunks, nD):
    # This orients the largest cube so that
    # its start is the origin (0 corner), and
    # the first step is along the x axis, regardless of nD and nChunks:

    return 0, 2 ** ((-nChunks - 1) % nD)  # in Python 0 <=  a % b  < b.


@partial(jax.jit, static_argnums=(1,))
def pack_index(chunks, nD):
    p = 2 ** nD  # Turn digits mod 2**nD back into a single number:

    def body(x, y):
        return p * x + y, None

    # This is probably better done using jax.lax.reduce, but it seems batching is not supported.
    return jax.lax.scan(body, chunks[0], chunks[1:])[0]


# unpack_coords( list of nD coords ) --> list of coord chunks each nD bits.

@partial(jax.jit, static_argnums=(1,))
def unpack_coords(coords, max_int):
    nChunks = int(np.ceil(np.log2(max_int)))

    if nChunks < 1:
        nChunks = 1

    return transpose_bits(coords, nChunks)


@partial(jax.jit, static_argnums=(1,))
def transpose_bits(srcs, nDests):
    def inner_body(dest, inner_srcs):
        dest = dest * 2 + inner_srcs % 2
        return dest, inner_srcs // 2

    def outer_body(outer_srcs, _):
        dest, outer_srcs = jax.lax.scan(inner_body, 0, outer_srcs)
        return outer_srcs, dest

    _, dests = jax.lax.scan(outer_body, srcs, jnp.arange(nDests), reverse=True)

    return dests


# Gray encoder and decoder from http://en.wikipedia.org/wiki/Gray_code :
#
@jax.jit
def gray_encode(bn):
    return jnp.bitwise_xor(bn, bn // 2)


@jax.jit
def gray_decode(n):
    cond = lambda carry: carry[0] > 1

    def body(carry):
        _, n_, sh = carry
        div = n_ >> sh
        n_ = n_ ^ div
        sh = sh << 1

        return div, n_, sh

    _, n, _ = jax.lax.while_loop(cond, body, (2, n, 1))
    return n


# gray_encode_travel -- gray_encode given start and end using bit rotation.
#    Modified Gray code.  mask is 2**nbits - 1, the highest i value, so
#        gray_encode_travel( start, end, mask, 0 )    == start
#        gray_encode_travel( start, end, mask, mask ) == end
#        with a Gray-code-like walk in between.
#    This method takes the canonical Gray code, rotates the output word bits,
#    then xors ("^" in Python) with the start value.
#
@jax.jit
def gray_encode_travel(start, end, mask, i):
    travel_bit = start ^ end
    modulus = mask + 1  # == 2**nBits
    # travel_bit = 2**p, the bit we want to travel.
    # Canonical Gray code travels the top bit, 2**(nBits-1).
    # So we need to rotate by ( p - (nBits-1) ) == (p + 1) mod nBits.
    # We rotate by multiplying and dividing by powers of two:
    g = gray_encode(i) * (travel_bit * 2)
    return ((g | (g // modulus)) & mask) ^ start


@jax.jit
def gray_decode_travel(start, end, mask, g):
    travel_bit = start ^ end
    modulus = mask + 1  # == 2**nBits
    rg = (g ^ start) * (modulus // (travel_bit * 2))
    return gray_decode((rg | (rg // modulus)) & mask)


# child_start_end( parent_start, parent_end, mask, i ) -- Get start & end for child.
#    i is the parent's step number, between 0 and mask.
#    Say that parent( i ) =
#           gray_encode_travel( parent_start, parent_end, mask, i ).
#    And child_start(i) and child_end(i) are what child_start_end()
#    should return -- the corners the child should travel between
#    while the parent is in this quadrant or child cube.
#      o  child_start( 0 ) == parent( 0 )       (start in a corner)
#      o  child_end( mask ) == parent( mask )   (end in a corner)
#      o  child_end(i) - child_start(i+1) == parent(i+1) - parent(i)
#         (when parent bit flips, same bit of child flips the opposite way)
#    Those constraints still leave choices when nD (# of bits in mask) > 2.
#    Here is how we resolve them when nD == 3 (mask == 111 binary),
#    for parent_start = 000 and parent_end = 100 (canonical Gray code):
#         i   parent(i)    child_
#         0     000        000   start(0)    = parent(0)
#                          001   end(0)                   = parent(1)
#                 ^ (flip)   v
#         1     001        000   start(1)    = parent(0)
#                          010   end(1)                   = parent(3)
#                ^          v
#         2     011        000   start(2)    = parent(0)
#                          010   end(2)                   = parent(3)
#                 v          ^
#         3     010        011   start(3)    = parent(2)
#                          111   end(3)                   = parent(5)
#               ^          v
#         4     110        011   start(4)    = parent(2)
#                          111   end(4)                   = parent(5)
#                 ^          v
#         5     111        110   start(5)    = parent(4)
#                          100   end(5)                   = parent(7)
#                v          ^
#         6     101        110   start(6)    = parent(4)
#                          100   end(6)                   = parent(7)
#                 v          ^
#         7     100        101   start(7)    = parent(6)
#                          100   end(7)                   = parent(7)
#    This pattern relies on the fact that gray_encode_travel()
#    always flips the same bit on the first, third, fifth, ... and last flip.
#    The pattern works for any nD >= 1.
#
@jax.jit
def child_start_end(parent_start, parent_end, mask, i):
    start_i = jnp.maximum(0, (i - 1) & ~1)  # next lower even number, or 0
    end_i = jnp.minimum(mask, (i + 1) | 1)  # next higher odd number, or mask
    child_start = gray_encode_travel(parent_start, parent_end, mask, start_i)
    child_end = gray_encode_travel(parent_start, parent_end, mask, end_i)
    return child_start, child_end
