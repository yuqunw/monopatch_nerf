
def grid_sample_2d(mat, sample):
    # Nx2
    FD, C, RH, RW = mat.shape
    _, _, N, _ = sample.shape
    sample = (sample.view(FD, 1, -1, 2).clamp(-1, 1) + 1) / 2 * (RW -1)
    tl = sample.floor().long()
    l = tl[..., 0]
    t = tl[..., 1]
    w = sample - tl
    br = sample.ceil().long()
    r = br[..., 0]
    b = br[..., 1]
    wx = w[..., 0]
    wy = w[..., 1]
    tl = t * RW + l
    tr = t * RW + r
    bl = b * RW + l
    br = b * RW + r
    ftl = mat.view(FD, C, -1).gather(2, tl.expand(-1, C, -1))
    ftr = mat.view(FD, C, -1).gather(2, tr.expand(-1, C, -1))
    fbl = mat.view(FD, C, -1).gather(2, bl.expand(-1, C, -1))
    fbr = mat.view(FD, C, -1).gather(2, br.expand(-1, C, -1))

    # compute corners
    ft = ftl * (1 - wx) + ftr * wx
    fb = fbl * (1 - wx) + fbr * wx

    # FD x C
    return (ft * (1 - wy) + fb * wy).view(FD, -1, 1, N)