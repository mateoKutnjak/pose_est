from scipy.ndimage import rotate as rot
import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_processed_image(image, obj_center, obj_joints, scale, angle, draw_bbox=True):
    plt.imshow(image)

    plt.scatter(obj_center[0], obj_center[1], c='r')

    if draw_bbox:
        x0 = int(obj_center[0] - 200 * scale / 2)
        y0 = int(obj_center[1] - 200 * scale / 2)
        x1 = int(obj_center[0] + 200 * scale / 2)
        y1 = int(obj_center[1] + 200 * scale / 2)

        plt.plot([x0, x1], [y0, y0], c='b')
        plt.plot([x0, x1], [y1, y1], c='b')
        plt.plot([x0, x0], [y0, y1], c='b')
        plt.plot([x1, x1], [y0, y1], c='b')

    plt.title('Scale = {}\nAngle = {}'.format(scale, angle))

    for joint in obj_joints:
        plt.scatter(joint[0], joint[1], c='g')

    plt.show()

def rotate(original_image, obj_center, obj_joints, angle):
    rotated_image = rot(original_image, angle)
    rotation_origin = (original_image.shape[0] // 2, original_image.shape[1] // 2)

    points = np.copy(obj_center)
    points = points.reshape(-1, 2)
    points = np.append(points, obj_joints, axis=0)
    points = np.append(points, np.ones(shape=(len(points), 1)), axis=1)

    rotated_points = rotate_points(points, rotation_origin[1], rotation_origin[0], angle, original_image.shape[0], original_image.shape[1])

    rotated_center = rotated_points[0]
    rotated_joints = rotated_points[1:, :]

    return rotated_image, rotated_center, rotated_joints
# TODO joint visibility

def rotate_points(points, cx, cy, angle, h, w):
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    calculated = np.dot(points, M.T)
    return calculated

def crop(original_image, obj_center, obj_joints, scale):
    orig_x0 = int(obj_center[0] - 200 * scale / 2)
    orig_y0 = int(obj_center[1] - 200 * scale / 2)
    orig_x1 = int(obj_center[0] + 200 * scale / 2)
    orig_y1 = int(obj_center[1] + 200 * scale / 2)

    if orig_x0 < 0 or orig_y0 < 0 or orig_x1 > original_image.shape[1] or orig_y1 > original_image.shape[0]:
        original_image, x0, y0, x1, y1 = pad_image(original_image, orig_x0, orig_y0, orig_x1, orig_y1)
    else:
        x0, y0, x1, y1 = orig_x0, orig_y0, orig_x1, orig_y1

    obj_center[0] -= orig_x0
    obj_center[1] -= orig_y0
    obj_joints[:, 0] -= orig_x0
    obj_joints[:, 1] -= orig_y0

    return original_image[y0:y1, x0:x1, :], obj_center, obj_joints

def pad_image(img, x0, y0, x1, y1):
    img = np.pad(img,
                 ((np.abs(np.minimum(0, y0)), np.maximum(y1 - img.shape[0], 0)),
                  (np.abs(np.minimum(0, x0)), np.maximum(x1 - img.shape[1], 0)),
                  (0,0)), mode="constant")

    x_adjust = np.abs(np.minimum(0, x0))
    y_adjust = np.abs(np.minimum(0, y0))

    x0 += x_adjust
    y0 += y_adjust
    x1 += x_adjust
    y1 += y_adjust
    return img, x0, y0, x1, y1

def generate_labelmap(image, point, sigma):
    ul = [int(point[0] - 3 * sigma), int(point[1] - 3 * sigma)]
    br = [int(point[0] + 3 * sigma + 1), int(point[1] + 3 * sigma + 1)]

    if ul[0] > image.shape[1] or ul[1] >= image.shape[0] or br[0] < 0 or br[1] < 0:
        return image

    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    g_x = max(0, -ul[0]), min(br[0], image.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], image.shape[0]) - ul[1]

    img_x = max(0, ul[0]), min(br[0], image.shape[1])
    img_y = max(0, ul[1]), min(br[1], image.shape[0])

    image[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return image

def plot_labelmaps(original_image, original_joints, labelmaps, labelmap_joints):
    import pdb
    pdb.set_trace()

    fig, ax = plt.subplots(nrows=4, ncols=5)

    ax[0, 0].imshow(original_image)
    ax[0, 0].scatter(original_joints[:, 0], original_joints[:, 1])

    for i in range(0, 4):
        for j in range(1, 5):
            ax[i, j].imshow(labelmaps[:, :, i*4 + j-1])
            ax[i, j].scatter(labelmap_joints[:, 0], labelmap_joints[:, 1], s=1, c='r')

    plt.show()

def scale_points(input_res, output_res, points):
    inres = np.array(list(input_res))
    outres = np.array(list(output_res))

    factor = np.divide(outres, inres)
    scaled_joints = np.multiply(points, factor)

    return scaled_joints


def resize(original_image, obj_center, obj_joints, shape):
    resized_image = cv2.resize(original_image, shape)

    resized_center = scale_points(original_image.shape[:-1], resized_image.shape[:-1], obj_center)
    resized_joints = scale_points(original_image.shape[:-1], resized_image.shape[:-1], obj_joints)

    return resized_image, resized_center, resized_joints