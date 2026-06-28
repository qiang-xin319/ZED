import sys
import cv2
import numpy as np
import stereoconfig


def preprocess(img1, img2):
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
    return img1, img2


def getRectifyTransform(height, width, config):
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_K, left_distortion, right_K, right_distortion,
        (width, height), R, T, alpha=0
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2


def draw_line(image1, image2):
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
    line_interval = 50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)),
                 (2 * width, line_interval * (k + 1)), (0, 255, 0), 2, cv2.LINE_AA)
    return output


def stereoMatchSGBM(left_image, right_image, down_scale=False):
    img_channels = 1 if left_image.ndim == 2 else 3
    blockSize = 8
    paraml = {'minDisparity': 0,
              'numDisparities': 192,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 50,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY}
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml.copy()
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    size = (left_image.shape[1], left_image.shape[0])
    if not down_scale:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]
        disparity_left_half = left_matcher.compute(
            left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(
            right_image_down, left_image_down)
        disparity_left = cv2.resize(
            disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(
            disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    trueDisp_left = disparity_left.astype(np.float32) / 16.0
    trueDisp_right = disparity_right.astype(np.float32) / 16.0
    return trueDisp_left, trueDisp_right


def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    return depthMap.astype(np.float32)


def show_color_disparity(disparity: np.ndarray) -> np.ndarray:
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    return disp_color


def show_color_depth(depth: np.ndarray) -> np.ndarray:
    depth_vis = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_color


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        point_3d = param[y, x]
        distance = np.linalg.norm(point_3d)
        print(f"点击位置: ({x}, {y})")
        print(f"3D 坐标: X={point_3d[0]:.2f} mm, Y={point_3d[1]:.2f} mm, Z={point_3d[2]:.2f} mm")
        print(f"距离: {distance:.2f} mm")


if __name__ == '__main__':
    # 加载图像
    iml = cv2.imread(r'F:\ZED2i-ranging\Binocular-ranging-main\L0.png', 1)
    imr = cv2.imread(r'F:\ZED2i-ranging\Binocular-ranging-main\R0.png', 1)
    if iml is None or imr is None:
        print("Error: 图像为空，请检查路径！")
        sys.exit(0)

    height, width = iml.shape[:2]

    # 加载相机参数
    config = stereoconfig.stereoCamera()

    # 校正映射与Q矩阵
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)

    # 绘制校正线图
    line_img = draw_line(iml_rectified, imr_rectified)
    cv2.imwrite('check_rectification.png', line_img)

    # 匹配和深度估计
    iml_, imr_ = preprocess(iml_rectified, imr_rectified)
    disp, _ = stereoMatchSGBM(iml_, imr_, False)
    cv2.imwrite('disparity_gray.png', disp * 4)

    # 计算深度图并转伪彩色图
    depth_map = getDepthMapWithQ(disp, Q)
    depth_color = show_color_depth(depth_map)
    disparity_color = show_color_disparity(disp)

    # 保存
    cv2.imwrite('depth_color.png', depth_color)
    cv2.imwrite('disparity_color.png', disparity_color)

    # 显示交互窗口
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    cv2.imshow("Disparity Map (Color)", disparity_color)
    cv2.imshow("Depth Map (Color)", depth_color)
    cv2.setMouseCallback("Depth Map (Color)", on_mouse_click, points_3d)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
