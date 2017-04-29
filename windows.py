import cv2
import numpy as np
from vehicle.image import downscale, patch, im_rect
from vehicle.pipeline import PATCH_STEP
"""
This script is used for generating a set of perspective search windows

"""


def get_line(img, text):
    cv2.namedWindow('lines', cv2.WINDOW_NORMAL)
    points = []
    cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),  1, cv2.LINE_AA)

    def click(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.ellipse(img, (x, y), (3, 3), 0, 0, 360,  (0, 0, 255), 2)
            points.append((x, y))
            cv2.imshow('lines', img)

    cv2.setMouseCallback('lines', click)

    cv2.imshow('lines', img)

    cv2.waitKey(0)

    points = np.array(points)

    # fitting the line in form x = ay + b
    mlx, clx = np.linalg.lstsq(np.vstack([points[:, 1], np.ones(len(points))]).T, points[:, 0])[0]

    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    return mlx, clx, min_y, max_y


# given transforms, generate a set of windows
def generate_windows(r, t, m, d):
    # width of detection area in meters
    w = 24  # 5 lanes approximately
    h_min = -2  # meters  (adjust for hills)
    h_max = 2  # meters
    s = 1.15
    # for z in np.arange(5, 101, 5) + np.cumsum(np.arange(0, 20, 1)):
    for z in np.arange(0, 100, 2):
        twp = np.array([[-w / 2, -h_min, 5 + s ** z],
                        [-w / 2, -h_max, 5 + s ** z],
                         [w / 2, -h_min, 5 + s ** z],
                         [w / 2, -h_max, 5 + s ** z]])
        p, _ = cv2.projectPoints(twp, r, t, m, d)
        p = np.squeeze(p).astype(np.int32)

        # yield bbox
        bbox = np.hstack((np.min(p, axis=0), np.max(p, axis=0)))
        if np.any(bbox < 0):
            continue
        if bbox[3] - bbox[1] < 64:
            return
        yield np.hstack((np.min(p, axis=0), np.max(p, axis=0)))


if __name__ == '__main__':
    try:
        filename = 'test_images/straight_lines2.jpg'
        # load image
        image = cv2.imread(filename)
        # load calibration file
        data = np.load('data/calibration.npz')

        sl, il, ml, xl = get_line(image.copy(), 'Put points on the left line')
        sr, ir, mr, xr = get_line(image.copy(), 'Put points on the right line')

        y0, y1 = 470, 670
        x00, x10 = sl * y0 + il, sl * y1 + il
        x01, x11 = sr * y0 + ir, sr * y1 + ir
        image_points = np.array([[x00, y0], [x10, y1], [x01, y0], [x11, y1]], np.float32)
        world_points = np.array([[-1.85, 0, 20], [-1.85, 0, 2], [1.85, 0, 20], [1.85, 0, 2]])
        found, rvec, tvec = cv2.solvePnP(world_points, image_points, data['matrix'], data['distortion'])

        cv2.namedWindow('Sample', cv2.WINDOW_NORMAL)
        cv2.line(image, (int(x00), y0), (int(x10), y1), (0, 255, 0), 2)
        cv2.line(image, (int(x01), y0), (int(x11), y1), (0, 255, 0), 2)
        cv2.line(image, (int(x00), y0), (int(x01), y0), (0, 255, 0), 2)
        cv2.line(image, (int(x10), y1), (int(x11), y1), (0, 255, 0), 2)

        windows = np.array(list(generate_windows(rvec, tvec, data['matrix'], data['distortion'])))
        np.clip(windows[:, [0, 2]], 0, image.shape[1])
        np.clip(windows[:, [1, 3]], 0, image.shape[0])

        for p in windows:
            cv2.rectangle(image, tuple(p[:2]), tuple(p[2:]), (255, 0, 0), 1)

        cv2.imshow('Sample', image)
        cv2.imwrite('./output_images/windows.jpg', image)

        weightmap = np.ones((image.shape[:2]), np.float32)

        for w in windows:
            ds = downscale(image, w, 128)
            for p, px, py in patch(ds, (64, 64), PATCH_STEP):
                r = im_rect(px, py, (64, 64), w, 128)
                weightmap[r[1]:r[3], r[0]:r[2]] += 1

        cv2.imwrite('./output_images/weightmap.jpg', weightmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except IndexError as ex:
        print('Exception: ', ex)
    else:
        out_file_name = 'data/windows'
        np.savez(out_file_name, windows=windows, weightmap=weightmap)
        print('Saved windows in %s.npz' % out_file_name)
