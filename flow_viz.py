import numpy as np


UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    # num of transes for each color
    RY = 15
    YG = 6 
    GC = 4 
    CB = 11
    BM = 13
    MR = 6 

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel
    

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])

    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    # convert to pol coords
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi  # angle in [-1, 1]

    # map flow angle to color hue
    # trans angle 'a' from [-1, 1] to a floating point index 'fk' into the color wheel
    # [-1, 1] --> [0, 2] --> [0, 1] --> [0, 55]
    # 55 is ncols = sum of color transes
    fk = (a + 1) / 2 * (ncols - 1) + 1

    # given fk is float angle -->  determine left/right int indices (k0, k1) on the color wheel
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    # handle wrap-around for the circular color wheel.
    k1[k1 == ncols + 1] = 1
    # fractional part of fk for saturation interp
    f = fk - k0

    # loop once for each color channel (R, G, B)
    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        # lin interpolate between the two colors
        col = (1 - f) * col0 + f * col1

        # mag (rad) to color saturation
        idx = rad <= 1
        # flows with magnitude <= 1, interpolate between the pure hue and white.
        # as 'rad' approaches 0, the color approaches white, as 'rad' approaches 1, the color approaches the pure hue color wheel
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        
        # flows with magnitude > 1 (an edge case), just darken the color.
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75

        # set the color for the current channel and set nan to black.
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


# from https://github.com/gengshan-y/VCN
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def save_vis_flow_tofile(flow, output_path):
    """
    A utility function to compute and save the flow visualization to a file.
    """
    vis_flow = flow_to_image(flow)
    from PIL import Image
    img = Image.fromarray(vis_flow)
    img.save(output_path)
