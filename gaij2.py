import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 解决中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 路径配置
left_img_path = "F:/ZED2i-ranging/Stereo-Detection-main/img/left/left_1.jpg"
right_img_path = "F:/ZED2i-ranging/Stereo-Detection-main/img/right/right_1.jpg"

# Q矩阵（请用你实际标定结果替换）
Q = np.float32([
    [1, 0, 0, -6.46397972e+02],
    [0, 1, 0, -3.61071846e+02],
    [0, 0, 0, 5.24513750e+02],
    [0, 0, -8.27215372e-03, 0]
])

# 图像增强函数（CLAHE + Gamma）
def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    gamma = 1.0 if mean_brightness > 127 else 1.5 - mean_brightness / 255
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(enhanced, lut)

# 加载图像
left_img_raw = cv2.imread(left_img_path)
right_img_raw = cv2.imread(right_img_path)
left_img = enhance_image(left_img_raw)
right_img = enhance_image(right_img_raw)

# 转灰度
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# SGBM 参数
min_disp = 0
num_disp = 16 * 9
block_size = 9

left_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

# 计算视差图
disparity_left = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
disparity_right = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0

# 滤波
filtered_disp = wls_filter.filter(disparity_left, gray_left, None, disparity_right)
filtered_disp = cv2.normalize(filtered_disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
filtered_disp = np.clip(filtered_disp, 0, 255).astype(np.uint8)

# 计算深度图
disparity = np.where(disparity_left <= 0, 0.1, disparity_left)  # 防止除以0
points_3D = cv2.reprojectImageTo3D(disparity, Q)
depth_map = points_3D[:, :, 2]
depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

# 显示用的深度图
depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_display = np.clip(depth_display, 0, 255).astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

# 显示
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(left_img_raw, cv2.COLOR_BGR2RGB))
plt.title("原始左图")

plt.subplot(1, 3, 2)
plt.imshow(filtered_disp, cmap='plasma')
plt.title("优化视差图")

plt.subplot(1, 3, 3)
plt.imshow(depth_colored)
plt.title("优化深度图")

plt.tight_layout()
plt.show()
