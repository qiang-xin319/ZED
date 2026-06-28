import sys
import cv2
import numpy as np
import stereoconfig


def preprocess(img1, img2):
    """灰度化 + 直方图均衡化"""
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
    return img1, img2


def getRectifyTransform(height, width, config):
    """获取立体校正映射矩阵"""
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        config.cam_matrix_left, config.distortion_l,
        config.cam_matrix_right, config.distortion_r,
        (width, height), config.R, config.T, alpha=0
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        config.cam_matrix_left, config.distortion_l, R1, P1,
        (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        config.cam_matrix_right, config.distortion_r, R2, P2,
        (width, height), cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, Q


def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    """执行立体图像校正"""
    return (
        cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR),
        cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)
    )


def draw_line(image1, image2, interval=50):
    """在图像上绘制对齐参考线"""
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[:image1.shape[0], :image1.shape[1]] = image1
    output[:image2.shape[0], image1.shape[1]:] = image2

    for k in range(0, height, interval):
        cv2.line(output, (0, k), (width, k), (0, 255, 0), 1)
    return output


def stereoMatchSGBM(left_image, right_image, down_scale=False):
    """生成视差图（SGBM）"""
    img_channels = 1 if left_image.ndim == 2 else 3
    blockSize = 9
    numDisparities = 192
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8 * img_channels * blockSize ** 2,
        P2=32 * img_channels * blockSize ** 2,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    if down_scale:
        left_image = cv2.pyrDown(left_image)
        right_image = cv2.pyrDown(right_image)

    disp = matcher.compute(left_image, right_image).astype(np.float32) / 16.0
    return disp


def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """使用重投影矩阵Q估算深度"""
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    depthMap[(depthMap <= 0) | (depthMap > 8000)] = 0  # 限制深度在合理范围
    return depthMap.astype(np.float32)


def show_color_disparity(disparity: np.ndarray) -> np.ndarray:
    """视差图转伪彩色图"""
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(disp_vis.astype(np.uint8), cv2.COLORMAP_JET)


def show_color_depth(depth: np.ndarray) -> np.ndarray:
    """深度图转伪彩色图"""
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)


def on_mouse_click(event, x, y, flags, param):
    """鼠标点击显示三维坐标和距离"""
    if event == cv2.EVENT_LBUTTONDOWN:
        point_3d = param[y, x]
        distance = np.linalg.norm(point_3d)
        print(f"[点击位置] ({x}, {y})")
        print(f"[三维坐标] X: {point_3d[0]:.2f} mm, Y: {point_3d[1]:.2f} mm, Z: {point_3d[2]:.2f} mm")
        print(f"[空间距离] {distance:.2f} mm")


def main():
    # 加载图像
    iml_path = r'F:\ZED2i-ranging\Binocular-ranging-main\L0.png'
    imr_path = r'F:\ZED2i-ranging\Binocular-ranging-main\R0.png'
    iml = cv2.imread(iml_path)
    imr = cv2.imread(imr_path)
    if iml is None or imr is None:
        print("❌ 图像读取失败，请检查路径！")
        sys.exit(1)

    height, width = iml.shape[:2]
    config = stereoconfig.stereoCamera()
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)

    # 校正图像
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    cv2.imwrite('rectified_line_view.png', draw_line(iml_rectified, imr_rectified))

    # 图像预处理
    iml_proc, imr_proc = preprocess(iml_rectified, imr_rectified)

    # 立体匹配
    disp = stereoMatchSGBM(iml_proc, imr_proc)
    cv2.imwrite('disparity_gray.png', disp * 4)

    # 深度估计
    depth_map = getDepthMapWithQ(disp, Q)
    disp_color = show_color_disparity(disp)
    depth_color = show_color_depth(depth_map)

    # 保存结果
    cv2.imwrite('disparity_color.png', disp_color)
    cv2.imwrite('depth_color.png', depth_color)

    # 交互窗口
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    cv2.namedWindow("Depth Map (Color)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Depth Map (Color)", on_mouse_click, points_3d)
    cv2.imshow("Disparity Map (Color)", disp_color)
    cv2.imshow("Depth Map (Color)", depth_color)
    print("🖱 点击深度图以测量3D坐标与距离...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
