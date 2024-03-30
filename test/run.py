import cv2
import numpy as np
import stereoconfig_040_2   # Camera parameters
import open3d

# import pcl
# import pcl.pcl_visualization


def preprocess(img1, img2):
    # RGB->Gray
    if(img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # OpenCV:BGR
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Equalize histogram
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# Elimination distortion
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# Obtain the mapping transformation matrix and reprojection matrix for distortion correction and stereo correction
# @param：config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # Read Intrinsic/external parameters
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # Rectify
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, 
                                                    (width, height), R, T, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# Distortion & Stereo Rectification
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
    return rectifyed_img1, rectifyed_img2


# Checking Stereo Rectification----draw lines
def draw_line(image1, image2):
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # draw equal distance parallel lines
    line_interval = 50  
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    return output


# Disparity Calculation
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM configs
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 1,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }

    # SGBM Object 
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # Get Disparity map
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)

    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # Real Disparity（SGBM Disparity is ×16）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


# h×w×3 into N×3
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


# Change depth map into point cloud
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)

    # Color
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # Coordinate + Color
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)

    # Delete some inappropriate points
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]

    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack((remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return pointcloud_1


# Show point cloud
# def view_cloud(pointcloud):
#     cloud = pcl.PointCloud_PointXYZRGBA()
#     cloud.from_array(pointcloud)

#     try:
#         visual = pcl.pcl_visualization.CloudViewing()
#         visual.ShowColorACloud(cloud)
#         v = True
#         while v:
#             v = not (visual.WasStopped())
#     except:
#         pass


if __name__ == '__main__':

    i = 8
    string = 'Val'
    # Read image
    iml = cv2.imread('./ValLeft8.bmp')  
    imr = cv2.imread('./ValRight8.bmp') 
    height, width = iml.shape[0:2]

    print("width = %d \n"  % width)
    print("height = %d \n" % height)
    

    # Read camera parameters
    config = stereoconfig_040_2.stereoCamera()

    # Stereo Rectification
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # Obtain the mapping matrix used for distortion correction and stereo correction and the reprojection matrix used to calculate pixel space coordinates
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)

    print("Print Q!")
    print(Q)

    # Check Stereo Rectification
    line = draw_line(iml_rectified, imr_rectified)
    cv2.imwrite('./%sVerify%d.png' %(string,i), line)

    # undistortion
    iml = undistortion(iml, config.cam_matrix_left, config.distortion_l)
    imr = undistortion(imr, config.cam_matrix_right, config.distortion_r)

    # stereo matching
    iml_, imr_ = preprocess(iml, imr)  # preprocess with lights (optional)

    iml_rectified_l, imr_rectified_r = rectifyImage(iml_, imr_, map1x, map1y, map2x, map2y)

    disp, _ = stereoMatchSGBM(iml_rectified_l, imr_rectified_r, True) 
    cv2.imwrite('./%sDisparity%d.png' %(string,i), disp)

    

    # Get pixel 3d corrdinates (from left camera)
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # using stereo_config.py parameters

    #points_3d = points_3d

        # on click
    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Point (%d, %d) 3D coordinates (%f, %f, %f)' % (x, y, points_3d[y, x, 0], points_3d[y, x, 1], points_3d[y, x, 2]))
            dis = ( (points_3d[y, x, 0] ** 2 + points_3d[y, x, 1] ** 2 + points_3d[y, x, 2] **2) ** 0.5) / 1000
            print('Point (%d, %d) Distance to left camera %0.3f m' %(x, y, dis) )

        # Show image
    cv2.namedWindow("disparity",0)
    cv2.imshow("disparity", disp)
    cv2.setMouseCallback("disparity", onMouse, 0)

    

    # Build point cloud--Point_XYZRGBA
    pointcloud = DepthColor2Cloud(points_3d, iml)
    # pointcloud = hw3ToN3(points_3d)
    # for i in range(len(pointcloud)):
    #     for j in range(3):
    #         if(not isinstance(pointcloud[i][j],np.float32)):
    #             print(type(pointcloud[i][j]))

    colors = np.random.randn(921600,3)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pointcloud[1:,0:3])
    open3d.visualization.draw_geometries([pcd])
    
    print(pointcloud.shape)

    # show point cloud 
    # view_cloud(pointcloud)

    cv2.waitKey(0)
    cv2.destroyAllWindows()