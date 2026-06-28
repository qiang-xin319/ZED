import sys
import cv2
import numpy as np
import stereoconfig1
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ---------------------- 核心改进模块 ----------------------
def enhanced_preprocess(img1, img2):
    """增强型预处理流程"""

    def adaptive_gamma(img):
        gamma = np.log(img.mean() + 1e-6) / np.log(127.5)
        return cv2.LUT(img, np.array([(i / 255.0) ** (1 / gamma) * 255
                                      for i in range(256)], dtype=np.uint8))

    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = cv2.fastNlMeansDenoising(img1, h=7, templateWindowSize=7, searchWindowSize=21)
    img2 = cv2.fastNlMeansDenoising(img2, h=7, templateWindowSize=7, searchWindowSize=21)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)

    img1 = adaptive_gamma(img1)
    img2 = adaptive_gamma(img2)

    # 添加细节增强
    # img1 = cv2.detailEnhance(img1, sigma_s=10, sigma_r=0.15)
    # img2 = cv2.detailEnhance(img2, sigma_s=10, sigma_r=0.15)

    return img1, img2


def optimized_SGBM():
    """优化后的SGBM参数配置"""
    param = {
        'mode': cv2.STEREO_SGBM_MODE_HH,
        'minDisparity': 0,
        'numDisparities': 120,
        'blockSize': 7,
        'P1': 8 * 1 * 7 ** 2,
        'P2': 32 * 1 * 7 ** 2,
        'disp12MaxDiff': 1,
        'preFilterCap': 63,
        'uniquenessRatio': 8,
        'speckleWindowSize': 50,
        'speckleRange': 1
    }
    return cv2.StereoSGBM_create(**param)


def advanced_post_filtering(disp_left, img_left, Q):
    """多阶段后处理滤波 - 修复WLS需要右视差图的问题"""
    left_matcher = optimized_SGBM()
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    disp_right = right_matcher.compute(img_left, img_left)

    lmbda = 8000
    sigma_color = 1.5
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma_color)

    filtered_disp = wls_filter.filter(disp_left, img_left, disparity_map_right=disp_right)
    filtered_disp = filtered_disp.astype(np.float32) / 16.0

    filtered_disp = filtered_disp.astype(np.float32)
    img_left_f = img_left.astype(np.float32)
    filtered_disp = cv2.ximgproc.jointBilateralFilter(filtered_disp, img_left_f, 15, 5, 5)

    def hole_filling(disp):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
        fill_mask = (closed == 0).astype(np.uint8)
        return cv2.inpaint(closed, fill_mask, 5, cv2.INPAINT_NS)

    return hole_filling(filtered_disp)


# ---------------------- 核心流程函数 ----------------------
def get_depth_pipeline(img_left, img_right, config):
    """完整的深度计算流水线"""
    height, width = img_left.shape[:2]
    map1x, map1y, map2x, map2y, Q = get_rectify_transform(height, width, config)
    imgL_rect = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
    imgR_rect = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

    imgL_proc, imgR_proc = enhanced_preprocess(imgL_rect, imgR_rect)

    left_matcher = optimized_SGBM()
    disp_left = left_matcher.compute(imgL_proc, imgR_proc)

    true_disp = advanced_post_filtering(disp_left, imgL_proc, Q)

    depth_map = cv2.reprojectImageTo3D(true_disp, Q)[:, :, 2]
    depth_map = np.where((depth_map < 10) | (depth_map > 10000), 0, depth_map)
    depth_map = cv2.medianBlur(depth_map.astype(np.float32), 5)
    depth_map = cv2.bilateralFilter(depth_map, 9, 75, 75)

    return true_disp, depth_map, Q


# ---------------------- 辅助函数 ----------------------
def get_rectify_transform(height, width, config):
    """立体校正参数生成"""
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        config.cam_matrix_left, config.distortion_l,
        config.cam_matrix_right, config.distortion_r,
        (width, height), config.R, config.T,
        alpha=0, flags=cv2.CALIB_ZERO_DISPARITY
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        config.cam_matrix_left, config.distortion_l, R1, P1,
        (width, height), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        config.cam_matrix_right, config.distortion_r, R2, P2,
        (width, height), cv2.CV_32FC1
    )
    return map1x, map1y, map2x, map2y, Q


def on_mouse_click(event, x, y, flags, param):
    """交互式深度查询"""
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_map, Q = param
        z = depth_map[y, x]
        if z <= 0:
            print(f"({x}, {y}): 无效深度")
            return

        fx = Q[2, 3]
        cx = -Q[0, 3]
        cy = -Q[1, 3]

        x_coord = (x - cx) * z / fx
        y_coord = (y - cy) * z / fx

        print(f"世界坐标 (mm): X={x_coord:.1f}, Y={y_coord:.1f}, Z={z:.1f}")
        print(f"距离: {np.sqrt(x_coord ** 2 + y_coord ** 2 + z ** 2):.1f} mm")


# ---------------------- 主程序 ----------------------
if __name__ == '__main__':
    imgL = cv2.imread('L01.png')  # 修改为实际左图路径
    imgR = cv2.imread('R01.png')  # 修改为实际右图路径
    if imgL is None or imgR is None:
        print("错误：图像加载失败")
        sys.exit(1)

    config = stereoconfig1.stereoCamera()

    disparity, depth, Q = get_depth_pipeline(imgL, imgR, config)

    # ---------------------- Matplotlib 可视化 ----------------------
    disp_viz = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_viz = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.subplot(231), plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)), plt.title('原始左图')
    plt.subplot(232), plt.imshow(disp_viz, cmap='jet'), plt.title('优化视差图')
    plt.subplot(233), plt.imshow(depth_viz, cmap='jet'), plt.title('优化深度图')
    plt.subplot(234), plt.hist(disparity.flatten(), bins=100), plt.title('视差分布')
    plt.subplot(235), plt.hist(depth[depth > 0].flatten(), bins=100), plt.title('深度分布')
    plt.tight_layout()
    plt.savefig('enhanced_result.png')

    # ---------------------- 彩色视差图显示 ----------------------
    colored_disp = cv2.applyColorMap(disp_viz, cv2.COLORMAP_JET)
    cv2.imwrite("colored_disparity.png", colored_disp)  # 保存彩色视差图

    cv2.namedWindow("Disparity Map", cv2.WINDOW_NORMAL)
    cv2.imshow("Disparity Map", colored_disp)

    # ---------------------- 彩色深度图显示 ----------------------
    colored_depth = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
    cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
    cv2.imshow("Depth Map", colored_depth)
    cv2.setMouseCallback("Depth Map", on_mouse_click, (depth, Q))

    print("按任意键退出，按 Esc 键可立即关闭窗口")

    while True:
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

