import numpy as np
from numba import jit, prange


@jit(nopython=True, nogil=True)
def interpolate_single(x, y, dydx, x_interp):
    idx = np.searchsorted(x, x_interp) - 1

    if idx < 0:
        return y[0]

    if idx > len(x) - 2:
        if idx > len(x) - 1:
            return y[-1]
        else:
            return y[-2]

    return y[idx] + (x_interp - x[idx]) * dydx[idx]


@jit(nopython=True, nogil=True)
def interpolation_kernel(v, r, vr, corner_positions, block_positions, x, y):
    dvrdr = np.diff(vr) / np.diff(r)

    if corner_positions[0] < 0:
        rJa = abs(corner_positions[0])
    else:
        rJa = 0

    if corner_positions[0] + len(x) > v.shape[0]:
        rJb = v.shape[0] - corner_positions[0]
    else:
        rJb = len(x)

    rJ = range(rJa, rJb)
    rj = range(max(corner_positions[0], 0), min(corner_positions[0] + len(x), v.shape[0]))

    if corner_positions[1] < 0:
        rKa = abs(corner_positions[1])
    else:
        rKa = 0

    if corner_positions[0] + len(y) > v.shape[1]:
        rKb = v.shape[1] - corner_positions[1]
    else:
        rKb = len(y)

    rK = range(rKa, rKb)
    rk = range(max(corner_positions[1], 0), min(corner_positions[1] + len(y), v.shape[1]))  #

    for j, J in zip(rj, rJ):
        for k, K in zip(rk, rK):
            r_interp = np.sqrt((x[J] - block_positions[0]) ** 2. + (y[K] - block_positions[1]) ** 2.)
            v[j, k] += interpolate_single(r, vr, dvrdr, r_interp)


@jit(nopython=True, nogil=True, parallel=True)
def interpolation_kernel_parallel(v, r, vr, corner_positions, block_positions, x, y):
    for i in prange(len(corner_positions)):
        interpolation_kernel(v, r, vr[i], corner_positions[i], block_positions[i], x, y)


def interpolate_radial_functions(array, r, values, positions, sampling):
    block_margin = int(r[-1] / min(sampling))
    block_size = 2 * block_margin + 1

    corner_positions = np.round(positions[:, :2] / sampling).astype(np.int) - block_margin
    block_positions = positions[:, :2] - sampling * corner_positions

    x = np.linspace(0., block_size * sampling[0], block_size, endpoint=False)
    y = np.linspace(0., block_size * sampling[1], block_size, endpoint=False)

    interpolation_kernel_parallel(array, r, values, corner_positions, block_positions, x, y)
