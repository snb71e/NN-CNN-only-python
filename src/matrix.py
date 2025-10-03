import numpy as np

def im2col(x, FH, FW, stride=1, pad=0):
    N, C, H, W = x.shape
    H_out = (H + 2*pad - FH) // stride + 1
    W_out = (W + 2*pad - FW) // stride + 1

    img = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    cols = np.zeros((N, C, FH, FW, H_out, W_out))

    for y in range(FH):
        y_max = y + stride*H_out
        for x in range(FW):
            x_max = x + stride*W_out
            cols[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    cols = cols.reshape(N, C*FH*FW, H_out*W_out)
    cols = cols.transpose(1, 0, 2).reshape(C*FH*FW, N*H_out*W_out)
    return cols, H_out, W_out

def col2im(cols, x_shape, FH, FW, stride=1, pad=0):
    N, C, H, W = x_shape
    H_out = (H + 2*pad - FH) // stride + 1
    W_out = (W + 2*pad - FW) // stride + 1

    cols = cols.reshape(C, FH*FW, N, H_out*W_out).transpose(2, 0, 1, 3)
    cols = cols.reshape(N, C, FH, FW, H_out, W_out)

    dx_padded = np.zeros((N, C, H + 2*pad, W + 2*pad))

    for y in range(FH):
        y_max = y + stride*H_out
        for x in range(FW):
            x_max = x + stride*W_out
            dx_padded[:, :, y:y_max:stride, x:x_max:stride] += cols[:, :, y, x, :, :]
    if pad == 0:
        return dx_padded
    return dx_padded[:, :, pad:H+pad, pad:W+pad]