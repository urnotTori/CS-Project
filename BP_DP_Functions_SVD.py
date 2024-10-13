import trimesh  # Import trimesh for 3D mesh operations
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
import cv2  # Import OpenCV for image processing
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import maximum_filter
import pcl
import open3d as o3d
from PIL import Image
import colorsys
import plotly.express as px  # For interactive 3D plotting
from sklearn.cluster import KMeans  # Import KMeans for clustering


# Load .glb file and extract vertices
# def load_mesh_glb(file_path):
#     scene = trimesh.load(file_path, process=False)
#     if isinstance(scene, trimesh.Scene):
#         first_mesh_key = list(scene.geometry.keys())[0]
#         first_mesh = scene.geometry[first_mesh_key]
#         if hasattr(first_mesh.visual, 'vertex_colors'):
#             vertices = first_mesh.vertices
#             colors = first_mesh.visual.vertex_colors[:, :3]
#             if vertices.ndim != 2 or vertices.shape[1] != 3:
#                 raise ValueError(f"Expected a 2D array with shape (num_points, 3), but got shape {vertices.shape}")
#             return vertices, colors
#         else:
#             print("No vertex colors found in the first mesh.")
#             return None, None
#     elif isinstance(scene, trimesh.Trimesh):
#         if hasattr(scene.visual, 'vertex_colors'):
#             vertices = scene.vertices
#             colors = scene.visual.vertex_colors[:, :3]
#             if vertices.ndim != 2 or vertices.shape[1] != 3:
#                 raise ValueError(f"Expected a 2D array with shape (num_points, 3), but got shape {vertices.shape}")
#             return vertices, colors
#         else:
#             print("No vertex colors found in the mesh.")
#             return None, None
#     else:
#         print("Loaded object is neither a Mesh nor a Scene.")
#         return None, None
def load_mesh_glb(file_path):
    scene = trimesh.load(file_path, process=False)
    if isinstance(scene, trimesh.Scene):
        print("Scene")
        first_mesh_key = list(scene.geometry.keys())[0]
        first_mesh = scene.geometry[first_mesh_key]
        vertices = first_mesh.vertices
        colors = first_mesh.visual.vertex_colors
        print("Vertex shape:", vertices.shape)
        print("Colors shape:", colors.shape)

        if colors.ndim == 1:
            colors = np.stack([colors] * 3, axis=-1)
        else:
            colors = colors[:, :3]

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected a 2D array with shape (num_points, 3), but got shape {vertices.shape}")
        return vertices, colors

    elif isinstance(scene, trimesh.Trimesh):
        print("Trimesh")
        vertices = scene.vertices
        colors = scene.visual.vertex_colors
        print("Vertex colors shape:", colors.shape)

        if colors.ndim == 1:
            colors = np.stack([colors] * 3, axis=-1)
        else:
            colors = colors[:, :3]

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected a 2D array with shape (num_points, 3), but got shape {vertices.shape}")
        return vertices, colors

    else:
        print("Loaded object is neither a Mesh nor a Scene.")
        return None, None


def load_mesh_ply(file_path):
    # Load the .ply file using trimesh
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices  # Get vertices of the mesh
    try:
        colors = mesh.visual.vertex_colors  # Try to get vertex colors
    except AttributeError:
        print("No vertex colors available")
        colors = None
    print("Vertices:\n", vertices)
    if colors is not None:
        print("Vertex Colors:\n", colors)
    return vertices, colors


def load_mesh_obj(file_path):
    # Load the .obj file using trimesh
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices  # Get vertices of the mesh
    try:
        colors = mesh.visual.vertex_colors  # Try to get vertex colors
    except AttributeError:
        print("No vertex colors available")
        colors = None

    print("Vertices:\n", vertices)
    if colors is not None:
        print("Vertex Colors:\n", colors)
        if colors.ndim == 1:
            colors = np.stack([colors] * 3, axis=-1)
        else:
            colors = colors[:, :3]

    return vertices, colors


def spatial_object_detection(colored_points, distance_threshold=0.1, angle_threshold=15):
    """
    检测符合棋盘格特征的点。
    colored_points: 经过颜色筛选的平面点
    distance_threshold: 用于判断点之间是否符合棋盘格结构的距离阈值
    angle_threshold: 用于判断邻居点的排列角度是否接近90度
    """
    # 找到最接近的邻居点
    nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(colored_points)
    distances, indices = nbrs.kneighbors(colored_points)

    # 基于距离和几何排列分析点的结构是否符合棋盘格
    grid_points = []

    for i, point in enumerate(colored_points):
        neighbors = colored_points[indices[i]]

        # 计算邻居点的平均距离
        avg_distance = np.mean(distances[i])

        # 如果点与邻居的平均距离符合棋盘格的结构特征
        if avg_distance < distance_threshold:
            # 计算点与邻居之间的角度
            angles = []
            for neighbor in neighbors:
                vector = neighbor - point
                angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
                angles.append(angle)

            # 将角度差异较小的点视为符合棋盘格结构的点
            angles = np.array(angles)
            angle_diffs = np.abs(np.diff(np.sort(angles)))

            # 检查角度是否接近90度
            if np.all(np.abs(angle_diffs - 90) < angle_threshold):
                grid_points.append(point)

    return np.array(grid_points)

# def extract_colored_points(points, colors, plane_points, hsv_ranges):
#     colored_points = []
#     print('here 2')
#
#     # 将 plane_points 转换为集合，加快查找速度
#     plane_points_set = set(map(tuple, plane_points))
#     # 批量转换颜色为HSV
#     colors_hsv = cv2.cvtColor(np.uint8([colors]), cv2.COLOR_RGB2HSV)[0]
#
#     # Loop through plane points and check their colors
#     for i, point in enumerate(points):
#         if tuple(point) in plane_points_set:
#             color_hsv = colors_hsv[i]
#             # 检查颜色是否在HSV范围内
#             for (lower_hsv, upper_hsv) in hsv_ranges:
#                 if np.all(color_hsv >= lower_hsv) and np.all(color_hsv <= upper_hsv):
#                     colored_points.append(point)
#                     break
#     print('here 3')
#     return np.array(colored_points)

###################### Exclude points which are close to black/white/gray
def extract_colored_points(points, colors, plane_points):
    colored_points = []
    # 将 plane_points 转换为集合，加快查找速度
    plane_points_set = set(map(tuple, plane_points))
    # 批量转换颜色为HSV
    colors_hsv = cv2.cvtColor(np.uint8([colors]), cv2.COLOR_RGB2HSV)[0]

    # Loop through plane points and check their colors
    for i, point in enumerate(points):
        if tuple(point) in plane_points_set:
            color_hsv = colors_hsv[i]
            H, S, V = color_hsv
            # Check if the point has a distinct color
            # Filter based on S and V to exclude gray, black, and white
            if S > 100 and 70 < V < 200:
                # Filter specific color ranges (Red, Orange, Yellow, Green, Cyan, Blue, Purple)
                if (0 <= H <= 10) or (170 <= H <= 180) or \
                        (10 < H <= 25) or (25 < H <= 35) or \
                        (35 < H <= 85) or (85 < H <= 100) or \
                        (100 < H <= 130) or (130 < H <= 160):
                    colored_points.append(point)

    return np.array(colored_points)
    # # 对提取出的有颜色的平面点进行空间特征检测
    # colored_points = np.array(colored_points)
    # print("No. colored_points: ", len(colored_points))
    # grid_points = spatial_object_detection(colored_points)
    # print("No. grid points: ", len(grid_points))
    #
    # return grid_points

def cluster_hsv(df, n_clusters=5):
    """
    Perform K-Means clustering on the HSV values.

    Args:
        df (pd.DataFrame): DataFrame containing 'HSV H', 'HSV S', 'HSV V' columns.
        n_clusters (int): Number of clusters for K-Means.

    Returns:
        pd.DataFrame: Original DataFrame with an additional 'Cluster' column.
    """
    # Extract HSV values
    hsv_values = df[['HSV H', 'HSV S', 'HSV V']].values

    # Initialize K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit and predict cluster labels
    df['Cluster'] = kmeans.fit_predict(hsv_values)

    print(f"K-Means clustering completed with {n_clusters} clusters.")
    return df


def plot_hsv_clusters(df):
    """
    Create an interactive 3D scatter plot of HSV clusters using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing 'HSV H', 'HSV S', 'HSV V', and 'Cluster' columns.
    """
    fig = px.scatter_3d(
        df,
        x='HSV H',
        y='HSV S',
        z='HSV V',
        color='Cluster',
        title='HSV Clusters',
        labels={'HSV H': 'Hue', 'HSV S': 'Saturation', 'HSV V': 'Value'},
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(scene=dict(
        xaxis_title='Hue',
        yaxis_title='Saturation',
        zaxis_title='Value'),
        margin=dict(l=0, r=0, b=0, t=50))
    fig.show()


# SVD plane detection
def svd_plane_detection(points, colors, threshold):
    plane_points = []
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    for i, point in enumerate(points):
        neighbors = points[indices[i]]
        centroid = np.mean(neighbors, axis=0)
        centered_neighbors = neighbors - centroid
        _, s, _ = np.linalg.svd(centered_neighbors)
        lambda1, lambda2, lambda3 = s
        if lambda3 / (lambda1+0.0000000000001) < threshold:
            plane_points.append(point)
        # Define HSV color ranges for the colors you want to extract
    # print('here 0')
    # hsv_ranges = [
    #     # 赤色 (Red)
    #     (np.array([0, 50, 50]), np.array([10, 255, 255])),
    #     (np.array([170, 50, 50]), np.array([180, 255, 255])),
    #     # 橙色 (Orange)
    #     (np.array([10, 50, 50]), np.array([25, 255, 255])),
    #     # 黄色 (Yellow)
    #     (np.array([25, 50, 50]), np.array([35, 255, 255])),
    #     # 绿色 (Green)
    #     (np.array([35, 50, 50]), np.array([85, 255, 255])),
    #     # 青色 (Cyan)
    #     (np.array([85, 50, 50]), np.array([100, 255, 255])),
    #     # 蓝色 (Blue)
    #     (np.array([100, 50, 50]), np.array([130, 255, 255])),
    #     # 紫色 (Purple)
    #     (np.array([130, 50, 50]), np.array([160, 255, 255])),
    #     # # 粉红色 (Pink)
    #     # (np.array([160, 50, 50]), np.array([170, 255, 255])),
    #     # # 棕色 (Brown) - 低亮度、低饱和度的橙色
    #     # (np.array([10, 100, 20]), np.array([20, 200, 100])),
    #     # # 土黄色 (Tan/Khaki)
    #     # (np.array([20, 50, 100]), np.array([30, 150, 200])),
    #
    #     ]

    # Extract colored points from the plane points
    # print('here 1')
    # colored_plane_points = extract_colored_points(points, colors, plane_points, hsv_ranges)
    # colored_plane_points = extract_colored_points(points, colors, plane_points)
    # print('here 4')
    # return colored_plane_points

    return np.array(plane_points)


def hough_plane_detection(points, threshold=0.1):
    # Create a PointCloud object
    cloud = o3d.geometry.PointCloud()

    # Convert the numpy array of points to PointCloud
    cloud.points = o3d.utility.Vector3dVector(points)

    # Perform plane segmentation using RANSAC
    plane_model, inliers = cloud.segment_plane(distance_threshold=threshold,
                                               ransac_n=3,
                                               num_iterations=1000)

    # Extract the points that belong to the detected plane
    plane_points = np.asarray(cloud.points)[inliers]

    return plane_points


# def find_colors_for_plane_points(plane_points, vertices, colors):
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vertices)
#     distances, indices = nbrs.kneighbors(plane_points)
#     plane_colors = colors[indices.flatten()]
#     return plane_colors
def find_colors_for_plane_points(plane_points, vertices, colors):
    # 检查空数据
    if colors is None or len(colors) == 0:
        raise ValueError("Colors array is empty or None.")
    if len(colors) != len(vertices):
        raise ValueError(f"Colors array size {len(colors)} does not match vertices size {len(vertices)}.")
    if plane_points is None or len(plane_points) == 0:
        raise ValueError("Plane points are empty or None.")

    # 计算最近邻
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vertices)
    distances, indices = nbrs.kneighbors(plane_points)

    # 检查索引范围
    if np.max(indices) >= len(colors):
        raise ValueError(f"Max index {np.max(indices)} is out of bounds for colors with size {len(colors)}.")

    plane_colors = colors[indices.flatten()]
    return plane_colors


def convert_to_black_and_white(plane_colors):
    # 初始化新的颜色数组
    new_colors = plane_colors.copy()

    # 遍历所有颜色
    for i, color in enumerate(plane_colors):
        # 将 RGB 转换为 HSV
        r, g, b = color[:3] / 255.0  # 归一化
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        # 判断Hue值是否在黑白的范围之外
        # 例如，如果 Hue 不在黑白或灰色的范围 (一般 Hue 值接近0或360代表红色，近中间值表示颜色)
        # hue is not in the range of black and white
        if not (h < 0.05 or h > 0.95 or s < 0.1):
            # set the point as black
            new_colors[i] = [0, 0, 0]

    return new_colors


def calculate_SVD(vertices):
    # Perform Singular Value Decomposition (SVD) on the vertices
    avg_values = vertices.mean(axis=0)  # Calculate mean of vertices
    vertices_remove_avg = vertices - avg_values  # Center the vertices
    Q = np.dot(vertices_remove_avg.T, vertices_remove_avg)  # Compute covariance matrix
    # Q = np.dot(vertices.T, vertices)

    U, sigma, Vt = np.linalg.svd(Q)  # Perform SVD
    A_tilde = np.dot(U.T, vertices_remove_avg.T).T  # Compute A_tilde matrix
    # A_tilde = np.dot(U.T, vertices.T).T
    return A_tilde, U, sigma

def scale_coordinates(A_tilde):
    # Scale the coordinates of the vertices to a fixed range
    x_min = np.min(A_tilde[:, 0])  # Get minimum X value
    x_max = np.max(A_tilde[:, 0])  # Get maximum X value
    y_min = np.min(A_tilde[:, 1])  # Get minimum Y value
    y_max = np.max(A_tilde[:, 1])  # Get maximum Y value
    range_x_org = np.max(A_tilde[:, 0]) - np.min(A_tilde[:, 0])  # Compute original X range
    range_y_org = np.max(A_tilde[:, 1]) - np.min(A_tilde[:, 1])  # Compute original Y range

    print("x_min: ", x_min)
    print("x_max: ", x_max)
    print("y_min: ", y_min)
    print("y_max: ", y_max)

    #########
    range_org = max(range_x_org, range_y_org)
    print("#### Scale range: ", range_org)

    # scaled_X = ((A_tilde[:, 0] - x_min) * 400 / range_x_org).astype(int)  # Scale X values
    # scaled_Y = ((A_tilde[:, 1] - y_min) * 400 / range_y_org).astype(int)  # Scale Y values
    #########
    s = 400 / range_org
    print("#### Coordinates scale: ", s)
    scaled_X = ((A_tilde[:, 0] - x_min) * s).astype(int)  # Scale X values
    scaled_Y = ((A_tilde[:, 1] - y_min) * s).astype(int)  # Scale Y values

    scaled_A_tilde_remove_Z = np.column_stack((scaled_X, scaled_Y))  # Combine scaled X and Y
    return scaled_A_tilde_remove_Z, s

def visualize_image(scaled_A_tilde_remove_Z, colors_R_reshaped, filename, width, height):
    # Visualize the image based on scaled coordinates and R values
    try:
        img = Image.new('L', (width, height), "black")  # Create a new grayscale image
        pixels = img.load()  # Load pixel access object
        for (x, y), color in zip(scaled_A_tilde_remove_Z, colors_R_reshaped):
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = int(color[0])  # Set pixel value based on R value
        img.show()  # Display the image
        # img.save(filename)  # Save the image
        # print(f"Image saved to '{filename}'.")
        return img
    except Exception as e:
        print(f"Error in visualize_image: {e}")
        return None

def rotate_coordinates(vertices, angle):
    # Rotate coordinates by a given angle
    theta = np.radians(angle)  # Convert angle to radians
    c, s = np.cos(theta), np.sin(theta)  # Compute cosine and sine of the angle
    rotation_matrix = np.array([
        [c, -s],
        [s, c]
    ])  # Create rotation matrix
    center = np.mean(vertices, axis=0)  # Compute center of the vertices
    centered_vertices = vertices - center  # Center the vertices
    rotated_vertices = np.dot(centered_vertices, rotation_matrix)  # Rotate the vertices
    return rotated_vertices + center  # Return rotated and recentered vertices


def apply_otsu_threshold(colors_R_reshaped):
    # Apply Otsu's thresholding method to the R values
    ret, binary_img = cv2.threshold(colors_R_reshaped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"OTSU threshold: {ret}")
    return binary_img


def apply_custom_threshold(colors_R_reshaped, threshold):
    # Apply a custom threshold to the R values
    colors_R_binary = np.where(colors_R_reshaped < threshold, 0, 255)  # Apply threshold
    return colors_R_binary

def fill_holes(image):
    # Apply morphological closing to fill holes in the image
    try:
        image_array = np.array(image).astype(np.uint8)  # Convert image to numpy array of type uint8
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if necessary
        kernel_size = 3  # Define kernel size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Create a square kernel
        image_closed = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel)  # Apply morphological closing
        img_processed = Image.fromarray(image_closed)  # Convert array back to image
        img_processed.show()  # Display the image
        img_processed.save('output_image_filled.png')  # Save the image
        return img_processed
    except Exception as e:
        print(f"Error in fill_holes: {e}")
        return None


def detect_edges_and_lines(image):
    try:
        image_array = np.array(image).astype(np.uint8)  # Convert image to numpy array of type uint8
        # edges = cv2.Canny(image_array, 50, 150, apertureSize=3)  # Detect edges using Canny edge detector
        edges = cv2.Canny(image_array, 25, 75, apertureSize=3)

        threshold = 85  # Initial threshold value
        min_lines = 60  # Minimum number of lines to detect
        max_lines = 100  # Maximum number of lines to detect

        while True:
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)  # Detect lines using Hough transform

            if lines is not None:
                num_lines = len(lines)
                # print("The number of lines: ", num_lines)
            else:
                num_lines = 0
            print(f"Adjusted threshold: {threshold}, Number of lines detected: {num_lines}")

            if num_lines < min_lines:
                threshold -= 1  # Decrease threshold
                if threshold <= 0:
                    print("Threshold too low, stopping adjustment.")
                    break
            elif num_lines > max_lines:
                threshold += 1  # Increase threshold
            else:
                break  # Suitable number of lines found

        return image_array, lines
    except Exception as e:
        print(f"Error in detect_edges_and_lines: {e}")
        return None, None

def draw_lines(image_array, lines):
    try:
        line_array = np.copy(image_array) * 0  # Create a copy of the original image for drawing lines

        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)  # Compute cosine of theta
                    b = np.sin(theta)  # Compute sine of theta
                    x0 = a * rho  # Compute x component of the line's intersection with rho
                    y0 = b * rho  # Compute y component of the line's intersection with rho
                    x1 = int(x0 + 1000 * (-b))  # Compute x coordinate of the line's endpoint
                    y1 = int(y0 + 1000 * (a))  # Compute y coordinate of the line's endpoint
                    x2 = int(x0 - 1000 * (-b))  # Compute x coordinate of the line's other endpoint
                    y2 = int(y0 - 1000 * (a))  # Compute y coordinate of the line's other endpoint
                    cv2.line(line_array, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Draw the line in red

            img_lines = Image.fromarray(line_array)  # Convert array back to image
            img_lines.show()  # Display the image
            img_lines.save('output_image_with_lines.png')  # Save the image
            print("Lines detected and drawn successfully.")
            return line_array, img_lines
        else:
            print("No lines were detected.")
            return None, None

    except Exception as e:
        print(f"Error in draw_lines: {e}")
        return None, None


def calculate_distance_transform(image):
    image_array = np.array(image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)  # use Euclidean distance

    print("Distance Transform Calculated")
    # print(dist_transform)
    # Save distance transform to CSV
    np.savetxt('distance_transform.csv', dist_transform, delimiter=",")
    print(f"Distance Transform saved to '{'distance_transform.csv'}'.")
    return dist_transform


def calculate_median_in_range(values, lower_bound, upper_bound):
    # 过滤出在指定范围内的值
    values_in_range = [v for v in values if lower_bound <= v <= upper_bound]
    if len(values_in_range) == 0:
        print(f"No values found in the range {lower_bound} to {upper_bound}.")
        return np.nan  # 返回 NaN 表示没有值
    # 计算这些值的中位数
    median_value = np.median(values_in_range)
    print(f"Median Value in range {lower_bound} to {upper_bound}: {median_value}")
    return median_value


def extract_local_maxima_values(dist_transform, pool_size=19, padding=7):
    """
    Extract local maxima values from a distance transform matrix.

    Args:
    dist_transform (np.ndarray): The distance transform matrix.
    pool_size (int): The size of the pooling window.
    padding (int): Padding applied to the distance transform matrix.

    Returns:
    np.ndarray: Array of local maxima values.
    """
    # Padding the distance transform matrix
    padded_dist_transform = np.pad(dist_transform, pad_width=padding, mode='constant', constant_values=0)

    # Perform Max Pooling
    pooled = maximum_filter(padded_dist_transform, size=pool_size)

    # Calculate J matrix
    J = pooled[padding:-padding, padding:-padding] - dist_transform  # Slicing to undo padding
    is_maxima = (J == 0).astype(int)
    # print("maxxxx:", is_maxima)
    # Image.fromarray(is_maxima).show()

    K = np.sign(1 - is_maxima).astype(np.uint8)  # Create K matrix where peak points are 0 and others are 1, and convert to uint8
    # Image.fromarray(K*255).show()  # Display K matrix for verification

    # Find indices of local maxima
    maxima_indices = np.argwhere(is_maxima == 1)

    # Retrieve local maxima values from the original distance transform matrix
    maxima_values = dist_transform[maxima_indices[:, 0], maxima_indices[:, 1]]

    # print("Peak points: ", maxima_values)
    print("Peak Num: ", len(maxima_values))
    # Save maxima values to a CSV file
    np.savetxt('peak_values.csv', maxima_values, delimiter=',')
    print("Peak values have been saved in 'peak_values.csv'.")

    return maxima_values, K


def plot_histogram(max_values):
    plt.figure()  # Create a new figure
    x_max = np.max(max_values)
    # x_max = 20
    hist, bins = np.histogram(max_values, bins=int(x_max), range=(0, x_max))
    # hist, bins = np.histogram(max_values, bins=100, range=(0, 20))
    # plt.figure()
    plt.title("Histogram of Local Max Values in Distance Transform")
    plt.xlabel("Distance Value")
    plt.ylabel("Frequency")
    plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")
    plt.show(block=False)
    print("Histogram Plotted")

    # 获取频率最高的区间
    max_freq_index = np.argmax(hist)  # 获取频率最高的区间的索引
    max_freq_bin = bins[max_freq_index]  # 对应的区间起始值
    max_freq_bin_end = bins[max_freq_index + 1]  # 对应的区间结束值
    max_freq = hist[max_freq_index]  # 对应的频率值

    print(f"Max frequency bin: [{max_freq_bin}, {max_freq_bin_end}), Frequency value: {max_freq}")

    return hist, (max_freq_bin, max_freq_bin_end), max_freq


def interactive_plot_with_points(line_image):
    """
    Display an interactive 2D plot where the user can click to select points.
    Shows the coordinates of each clicked point and the Euclidean distance
    between two selected points.

    Args:
    line_image (np.ndarray): The image to be displayed.
    """
    plt.figure()  # Create a new figure
    plt.imshow(line_image, cmap='gray')
    plt.title("Interactive 2D Image: Click to select points")

    points = []

    def onclick(event):
        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:
            points.append((ix, iy))
            plt.plot(ix, iy, 'ro')
            plt.draw()
            print(f"Selected Point: ({ix:.2f}, {iy:.2f})")
            if len(points) == 2:
                p1, p2 = points
                distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                print(f"Selected Points: {p1}, {p2}")
                print(f"Euclidean Distance: {distance:.2f}")
                plt.text(0, 0, f"Distance: {distance:.2f}", bbox=dict(facecolor='white', alpha=0.5))
                plt.draw()
                points.clear()  # Clear points list to allow for new selection

    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()


# Visualization
def visualize_3Dplanes(points, planes, method_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', marker='o', s=1)

    for plane in planes:
        ax.scatter(plane[:, 0], plane[:, 1], plane[:, 2], c='red', marker='o', s=1)

    plt.title(f'{method_name} Plane Detection')
    plt.show()

