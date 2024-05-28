import trimesh
import pyrender
import csv
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import cv2

# Load the .glb file using trimesh
scene = trimesh.load('chessboard.glb', process=False)

# check the loaded object is Mesh or Scene
if isinstance(scene, trimesh.Scene):
    # get the first Mesh in the Scene
    first_mesh_key = list(scene.geometry.keys())[0]
    first_mesh = scene.geometry[first_mesh_key]

    # check whether the vertices have colors
    if hasattr(first_mesh.visual, 'vertex_colors'):
        vertices = first_mesh.vertices  # get vertices' coordinates of the first mesh
        print("Vertices of a mesh in the scene:\n", vertices)  # print vertices' coordinates
        colors = first_mesh.visual.vertex_colors[:, :3]  # get vertices' colors of the first mesh, ignore the value of alpha
        print("Colors of a mesh in the scene:\n", colors)  # print vertices' colors(R, G, B)
        print(vertices.shape)  # print the dimension of vertices
        print(vertices.dtype)  # print the data type of vertices
    else:
        print("No vertex colors found in the first mesh.")

elif isinstance(scene, trimesh.Trimesh):
    # 直接处理单个Mesh
    if hasattr(scene.visual, 'vertex_colors'):
        vertices = scene.vertices
        print("Vertices of a mesh in the scene:\n", vertices)
        colors = scene.visual.vertex_colors[:, :3]
        print("Colors of a mesh in the scene:\n", colors)
        print(vertices.shape)  # 打印vertices的维度
        print(vertices.dtype)  # 打印vertices的数据类型
    else:
        print("No vertex colors found in the mesh.")

else:
    print("Loaded object is neither a Mesh nor a Scene.")

# create DataFrame
data = np.hstack((vertices, colors))
df = pd.DataFrame(data, columns=['X', 'Y', 'Z', 'R', 'G', 'B'])

# save the dataframe to a CSV file
df.to_csv('first_mesh_vertices_colors.csv', index=False)
print("Vertices and colors of the first mesh saved to 'first_mesh_vertices_colors.csv'.")

# calculate the average of each row in colors
colors_avg = np.mean(colors, axis=1)
print("Average colors:", colors_avg)
# extract the R colors
colors_R = colors[:, 0] # get the first column in colors
print("R Value:", colors_R)


# Center the vertex coordinates, calculate the covariance matrix of the centered vertices,
# and perform singular value decomposition (SVD) for subsequent analysis or dimensionality reduction:

# calculate the average of every column( average of X, Y, Z)
# avg_x, avg_y, avg_z = vertices.mean(axis=0)
avg_values = vertices.mean(axis=0)

# remove the average from the original X, Y, Z coordinates of vertices
#这个步骤被称为中心化，其目的是将数据的中心移至原点（0,0,0），这有助于去除数据的均值对于后续分析的影响，
# 使得处理过程主要关注于数据的变化和结构，而不是数据的绝对位置。
vertices_remove_avg = vertices - avg_values

# Calculate the dot product of the transposed matrix and the original matrix
# 协方差矩阵是一个描述数据各维度间线性关系密切程度的矩阵，它的对角线元素表示各维度的方差，非对角线元素表示不同维度间的协方差。
# 这一步骤是为了准备进行主成分分析（PCA）或其他相关的统计分析。
Q = np.dot(vertices_remove_avg.T, vertices_remove_avg)

# Single Value Composition- SVD 奇异值分解
# 对协方差矩阵 Q 进行奇异值分解，得到矩阵 U（左奇异向量）、sigma（奇异值）和 Vt（右奇异向量的转置）。SVD 是一种强大的矩阵分解技术，
# 用于提取数据的主要特征和降维。在这里，U 的列向量表示数据的主方向（或称为主成分），sigma 表示这些方向的重要性（方差大小）。
U, sigma, Vt = np.linalg.svd(Q)
print("SVD Result, U matrix:\n", U)
print("SVD Result, Sigma values:\n", sigma)

# 这一步实际上是将原始数据投影到由 U 的列向量定义的新空间中。
# 这个新空间的坐标系由数据的主成分构成，可以用于数据压缩、去噪或其他形式的分析。
A_tilde = np.dot(U.T, vertices_remove_avg.T)
A_tilde = A_tilde.T  ###################
print("A_tilde matrix:\n", A_tilde)

df = pd.DataFrame(A_tilde, columns=['X', 'Y', 'Z'])
# export to a CSV file
df.to_csv('A_tilde.csv', index=False)
print("A_tilde saved to 'A_tilde.csv'.")

# To remove the last column (Z dimension), use slicing
A_tilde_remove_Z = A_tilde[:, :-1]
# Display the modified matrix
print("A_tilde with the last column removed:")
print(A_tilde_remove_Z)

# Scaling the coordinates of vertices
x_min = np.min(A_tilde[:, 0])
x_max = np.max(A_tilde[:, 0])
y_min = np.min(A_tilde[:, 1])
y_max = np.max(A_tilde[:, 1])
print("X min:", x_min, "X max:", x_max)
print("Y min:", y_min, "Y max:", y_max)

coordinates_range = max(x_max-x_min, y_max-y_min)  ##### get max range
print("range 1: ", x_max-x_min)
print("range 2: ", y_max-y_min)
print("coordinates range:", coordinates_range)

range_x_org = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
range_y_org = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
print("original range x: ", range_x_org)
print("original range y: ", range_y_org)

# Scale X and Y values to be within [0, 200] -- ##### change range to original image size
scaled_X = ((A_tilde[:, 0] - x_min) * 400 / range_x_org).astype(int)  ###### Scale X values to be within [0, 400]
scaled_Y = ((A_tilde[:, 1] - y_min) * 400 / range_y_org).astype(int)  ###### Scale Y values to be within [0, 400]

# Combine scaled X and Y into a new matrix
scaled_A_tilde_remove_Z = np.column_stack((scaled_X, scaled_Y))
# Output the scaled matrix
print("Scaled A_tilde matrix:\n", scaled_A_tilde_remove_Z)

# colors_avg_reshaped = colors_avg.reshape(-1, 1)  # change colors_avg to 2D array
colors_R_reshaped = colors_R.reshape(-1, 1)  # change colors_R to 2D array
print("R Value (reshape):",colors_R_reshaped)

# data_trans = np.hstack((scaled_A_tilde_remove_Z, colors_avg_reshaped))
# df = pd.DataFrame(data_trans, columns=['X', 'Y', 'Color'])
# df.to_csv('Trans_vertices_and_colors.csv', index=False)
# print("Transformation of vertices and colors saved to 'Trans_vertices_and_colors.csv'.")
data_trans = np.hstack((scaled_A_tilde_remove_Z, colors_R_reshaped))
df = pd.DataFrame(data_trans, columns=['X', 'Y', 'R'])
# export to a CSV file
df.to_csv('Trans_XY_and_R.csv', index=False)
print("Transformation of vertices and R colors saved to 'Trans_XY_and_R.csv'.")


# Rotate coordinates
def rotate_coordinates(vertices, angle):
    """ Rotate coordinates by a given angle in degrees centered around the origin. """
    theta = np.radians(angle)  # Convert angle to radians
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [c, -s],
        [s, c]
    ])
    # Adjust the center
    center = np.mean(vertices, axis=0)
    centered_vertices = vertices - center
    rotated_vertices = np.dot(centered_vertices, rotation_matrix)
    # Recenter the vertices
    return rotated_vertices + center

# Example usage
angle = -50  # Adjust the angle according to your need
rotated_XY = rotate_coordinates(scaled_A_tilde_remove_Z, angle)

##### OTSU  #####
# Ensure the array is of type uint8
# colors_R_uint8 = colors_R_reshaped.astype(np.uint8)
print("max R color: ", np.max(colors_R_reshaped))
print("min R color: ", np.min(colors_R_reshaped))
# Apply Otsu's thresholding
ret, thresh = cv2.threshold(colors_R_reshaped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Convert thresholded values to binary 0 and 1
# binary_colors_R = (thresh // 255).astype(int)

# Output some of the binary results to verify
# print("new_colors_R: ", binary_colors_R[:30])  # Display the first 30 entries to check
print("OTSU_colors_R: ", thresh[:30])  # Display the first 30 entries to check
print("threshold: ", ret)

#### self-define threshold
threshold = 44
colors_R_binary = np.where(colors_R_reshaped < threshold, 0, 255)
print("new_colors_R: ", colors_R_binary[:30])  # Display the first 30 entries to check

# Visualization
# 假设 scaled_A_tilde_remove_Z 和 colors_R 已经正确计算和调整形状
# scaled_A_tilde_remove_Z 是 [N, 2] 形状的 NumPy 数组
# colors_R_reshaped 是 [N, 1] 形状的 NumPy 数组

# Determine new image dimensions
x_min, y_min = rotated_XY.min(axis=0)
x_max, y_max = rotated_XY.max(axis=0)
width = int(x_max - x_min) + 50  # 30 can be changed
height = int(y_max - y_min) + 60

# width = int(np.max(scaled_A_tilde_remove_Z[:, 0])) + 1
# height = int(np.max(scaled_A_tilde_remove_Z[:, 1])) + 1
# create a new gray image
img = Image.new('L', (width, height), "black")  # 'L' 代表灰度图

pixels = img.load()  # 获取图像的像素访问对象

## fill the pixel with gray value according to the coordinates
# for (x, y), color in zip(scaled_A_tilde_remove_Z, colors_R_reshaped):
for (x, y), color in zip(rotated_XY, colors_R_binary):  ##### change to binary_colors_R
     if 0 <= x < width and 0 <= y < height:  # ensure the coordinate in the scope of the image
         pixels[x, y] = int(color[0])  # set the R value to the gray value

# img.show()  # show the image

# # save the image
# img.save('output_image-2.png')

##### Closing (Erosion, Dilation) ######
image_array = np.array(img)

# Convert image to grayscale if it's not already
if len(image_array.shape) == 3:
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

# Define the kernel size for the morphological operation
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), np.uint8)

# Apply Closing to fill in small holes and connect small gaps
# closing small holes inside the foreground objects or small black points on the object
# It is achieved by a dilation followed by an erosion
image_closed = cv2.morphologyEx(image_array, cv2.MORPH_CLOSE, kernel)

# Convert back to PIL image to use the existing display/save functionality
img_processed = Image.fromarray(image_closed)

# Display the processed image
img_processed.show()

# Optionally save the processed image
# img_processed.save('output_image_processed.png')


############################## Detect edges
# Edge Detection using Canny
edges = cv2.Canny(image_closed, 50, 150, apertureSize=3)

####### Hough Line Transform #######
lines = cv2.HoughLines(edges, 1, np.pi / 180, 70)  # 70 can be adjusted to control the threshold

# Create an image to draw lines
line_image = np.copy(image_closed) * 0  # Creating a blank to draw lines on

# Draw the lines on the edge image
if lines is not None:
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)

# Convert the line image to a PIL image to show/save
img_lines = Image.fromarray(line_image)
img_lines.show()
# img_lines.save('output_image_with_lines.png')
