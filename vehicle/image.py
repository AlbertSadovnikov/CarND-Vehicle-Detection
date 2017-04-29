import cv2


def downscale(image, window, window_height):
    dh = window_height
    x0 = max(window[0], 0)
    x1 = min(window[2], image.shape[1])
    y0 = max(window[1], 0)
    y1 = min(window[3], image.shape[0])
    return cv2.resize(image[y0:y1, x0:x1], (int(dh * (x1 - x0) / (y1 - y0)), dh), cv2.INTER_LANCZOS4)


def patch(image, patch_size, patch_step):
    xlist = range(0, image.shape[1] - patch_size[0] + 1, patch_step[0])
    ylist = range(0, image.shape[0] - patch_size[1] + 1, patch_step[1])
    for xs in xlist:
        for ys in ylist:
            yield image[ys:ys + patch_size[1], xs:xs + patch_size[0], :], xs, ys


def im_rect(px, py, patch_size, window, window_height):
    dh = window_height
    x0 = window[0]
    y0 = window[1]
    y1 = window[3]

    k = (y1 - y0) / dh
    prx0 = x0 + px * k
    pry0 = y0 + py * k
    prx1 = prx0 + patch_size[0] * k
    pry1 = pry0 + patch_size[1] * k

    return int(prx0), int(pry0), int(prx1), int(pry1)