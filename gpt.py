import sys
import cv2
import numpy as np
import stereoconfig1
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------ 图像预处理 ------------------
def enhanced_preprocess(img1, img2):
    def adaptive_gamma(img):
        gamma = np.log(img.mean() + 1e-6) / np.log(127.5)
        return cv2.LUT(img, np.array([(i / 255.0) ** (1 / gamma) * 255 for i in range(256)], dtype=np.uint8))

    def histogram_stretch(img):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    def enhance_pipeline(img):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.fastNlMeansDenoising(img, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = adaptive_gamma(img)
        img = histogram_stretch(img)
        return img

    return enhance_pipeline(img1), enhance_pipeline(img2)

# ------------------ 匹配器配置 ------------------
def optimized_SGBM():
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=192,
        blockSize=9,
        P1=8 * 1 * 9 ** 2,
        P2=32 * 1 * 9 ** 2,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=1,
        mode=cv2.STEREO_SGBM_MODE_HH
    )

# ------------------ 后处理滤波 ------------------
def advanced_post_filtering(disp_left, imgL_proc, imgR_proc, Q):
    left_matcher = optimized_SGBM()
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    disp_right = right_matcher.compute(imgR_proc, imgL_proc)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.5)

    filtered_disp = wls_filter.filter(disp_left, imgL_proc, disparity_map_right=disp_right)
    filtered_disp = filtered_disp.astype(np.float32) / 16.0

    # 滤波增强：边缘引导空洞补全
    def edge_guided_hole_filling(disp):
        mask = (disp <= 0).astype(np.uint8)
        edges = cv2.Canny((disp * 16).astype(np.uint8), 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        fill_mask = np.clip(mask + dilated, 0, 1).astype(np.uint8)
        return cv2.inpaint(disp, fill_mask, 5, cv2.INPAINT_NS)

    disp_filled = edge_guided_hole_filling(filtered_disp)
    disp_filled = cv2.medianBlur(disp_filled.astype(np.float32), 5)
    disp_filled = cv2.bilateralFilter(disp_filled, 9, 75, 75)

    return disp_filled

# ------------------ 深度计算流水线 ------------------
def get_depth_pipeline(imgL, imgR, config):
    height, width = imgL.shape[:2]
    map1x, map1y, map2x, map2y, Q = get_rectify_transform(height, width, config)

    imgL_rect = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    imgR_rect = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    imgL_proc, imgR_proc = enhanced_preprocess(imgL_rect, imgR_rect)
    matcher = optimized_SGBM()
    dispL = matcher.compute(imgL_proc, imgR_proc)

    dispL = advanced_post_filtering(dispL, imgL_proc, imgR_proc, Q)

    depth_map = cv2.reprojectImageTo3D(dispL, Q)[:, :, 2]
    depth_map = np.where((depth_map < 10) | (depth_map > 10000), 0, depth_map)

    depth_map = cv2.medianBlur(depth_map.astype(np.float32), 5)
    depth_map = cv2.bilateralFilter(depth_map, 9, 75, 75)
    return dispL, depth_map, Q

# ------------------ 校正变换 ------------------
def get_rectify_transform(height, width, config):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
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

# ------------------ 鼠标点击深度输出 ------------------
def on_mouse_click(event, x, y, flags, param):
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
        print(f"距离: {np.sqrt(x_coord**2 + y_coord**2 + z**2):.1f} mm")

# ------------------ 主程序入口 ------------------
if __name__ == '__main__':
    imgL = cv2.imread('F:/ZED2i-ranging/Stereo-Detection-main/img/left/left_1.jpg')
    imgR = cv2.imread('F:/ZED2i-ranging/Stereo-Detection-main/img/right/right_1.jpg')

    if imgL is None or imgR is None:
        print("图像加载失败")
        sys.exit(1)

    config = stereoconfig1.stereoCamera()
    disparity, depth, Q = get_depth_pipeline(imgL, imgR, config)

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

    # 彩色显示
    cv2.imshow("Disparity Map", cv2.applyColorMap(disp_viz, cv2.COLORMAP_JET))
    cv2.imshow("Depth Map", cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET))
    cv2.setMouseCallback("Depth Map", on_mouse_click, (depth, Q))

    print("按任意键退出，Esc 键关闭窗口")
    while True:
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()
