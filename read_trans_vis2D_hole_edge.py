import trimesh  # Import trimesh for 3D mesh operations
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
from PIL import Image  # Import PIL for image processing
import cv2  # Import OpenCV for image processing

def load_mesh(file_path):
    # Load the .glb file using trimesh
    scene = trimesh.load(file_path, process=False)
    if isinstance(scene, trimesh.Scene):
        # If the scene contains multiple meshes
        first_mesh_key = list(scene.geometry.keys())[0]
        first_mesh = scene.geometry[first_mesh_key]
        if hasattr(first_mesh.visual, 'vertex_colors'):
            # Get vertices and colors of the first mesh
            vertices = first_mesh.vertices
            colors = first_mesh.visual.vertex_colors[:, :3]
            return vertices, colors
        else:
            print("No vertex colors found in the first mesh.")
            return None, None
    elif isinstance(scene, trimesh.Trimesh):
        # If the scene contains a single mesh
        if hasattr(scene.visual, 'vertex_colors'):
            vertices = scene.vertices
            colors = scene.visual.vertex_colors[:, :3]
            return vertices, colors
        else:
            print("No vertex colors found in the mesh.")
            return None, None
    else:
        print("Loaded object is neither a Mesh nor a Scene.")
        return None, None

def save_vertices_colors_to_csv(vertices, colors, filename):
    # Save the vertices and colors to a CSV file
    data = np.hstack((vertices, colors))  # Combine vertices and colors horizontally
    df = pd.DataFrame(data, columns=['X', 'Y', 'Z', 'R', 'G', 'B'])  # Create a DataFrame
    df.to_csv(filename, index=False)  # Save DataFrame to CSV
    print(f"Vertices and colors of the first mesh saved to '{filename}'.")

def calculate_SVD(vertices):
    # Perform Singular Value Decomposition (SVD) on the vertices
    avg_values = vertices.mean(axis=0)  # Calculate mean of vertices
    vertices_remove_avg = vertices - avg_values  # Center the vertices
    Q = np.dot(vertices_remove_avg.T, vertices_remove_avg)  # Compute covariance matrix
    U, sigma, Vt = np.linalg.svd(Q)  # Perform SVD
    A_tilde = np.dot(U.T, vertices_remove_avg.T).T  # Compute A_tilde matrix
    return A_tilde, U, sigma

def save_A_tilde_to_csv(A_tilde, filename):
    # Save the A_tilde matrix to a CSV file
    df = pd.DataFrame(A_tilde, columns=['X', 'Y', 'Z'])  # Create a DataFrame
    df.to_csv(filename, index=False)  # Save DataFrame to CSV
    print(f"A_tilde saved to '{filename}'.")

def scale_coordinates(A_tilde):
    # Scale the coordinates of the vertices to a fixed range
    x_min = np.min(A_tilde[:, 0])  # Get minimum X value
    x_max = np.max(A_tilde[:, 0])  # Get maximum X value
    y_min = np.min(A_tilde[:, 1])  # Get minimum Y value
    y_max = np.max(A_tilde[:, 1])  # Get maximum Y value
    range_x_org = np.max(A_tilde[:, 0]) - np.min(A_tilde[:, 0])  # Compute original X range
    range_y_org = np.max(A_tilde[:, 1]) - np.min(A_tilde[:, 1])  # Compute original Y range
    scaled_X = ((A_tilde[:, 0] - x_min) * 400 / range_x_org).astype(int)  # Scale X values
    scaled_Y = ((A_tilde[:, 1] - y_min) * 400 / range_y_org).astype(int)  # Scale Y values
    scaled_A_tilde_remove_Z = np.column_stack((scaled_X, scaled_Y))  # Combine scaled X and Y
    return scaled_A_tilde_remove_Z, x_min, x_max, y_min, y_max

def save_scaled_coordinates_and_R(scaled_A_tilde_remove_Z, colors_R, filename):
    # Save the scaled coordinates and R values to a CSV file
    colors_R_reshaped = colors_R.reshape(-1, 1)  # Reshape R values
    data_trans = np.hstack((scaled_A_tilde_remove_Z, colors_R_reshaped))  # Combine coordinates and R values
    df = pd.DataFrame(data_trans, columns=['X', 'Y', 'R'])  # Create a DataFrame
    df.to_csv(filename, index=False)  # Save DataFrame to CSV
    print(f"Transformation of vertices and R colors saved to '{filename}'.")

def visualize_image(scaled_A_tilde_remove_Z, colors_R_reshaped, filename, width, height):
    # Visualize the image based on scaled coordinates and R values
    try:
        img = Image.new('L', (width, height), "black")  # Create a new grayscale image
        pixels = img.load()  # Load pixel access object
        for (x, y), color in zip(scaled_A_tilde_remove_Z, colors_R_reshaped):
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = int(color[0])  # Set pixel value based on R value
        img.show()  # Display the image
        img.save(filename)  # Save the image
        print(f"Image saved to '{filename}'.")
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
    ret, thresh = cv2.threshold(colors_R_reshaped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"OTSU threshold: {ret}")
    return thresh

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

def detect_edges_and_draw_lines(image):
    # Detect edges and draw lines using the Hough transform
    try:
        image_array = np.array(image).astype(np.uint8)  # Convert image to numpy array of type uint8
        edges = cv2.Canny(image_array, 50, 150, apertureSize=3)  # Detect edges using Canny edge detector
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 70)  # Detect lines using Hough transform
        line_image = np.copy(image_array) * 0  # Create a blank image for drawing lines
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
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)  # Draw the line
        img_lines = Image.fromarray(line_image)  # Convert array back to image
        img_lines.show()  # Display the image
        img_lines.save('output_image_with_lines.png')  # Save the image
        return img_lines
    except Exception as e:
        print(f"Error in detect_edges_and_draw_lines: {e}")
        return None


def main():
    # Main function to execute all steps
    vertices, colors = load_mesh('chessboard.glb')  # Load mesh from file
    if vertices is not None and colors is not None:
        save_vertices_colors_to_csv(vertices, colors, 'first_mesh_vertices_colors.csv')  # Save vertices and colors to CSV
        A_tilde, U, sigma = calculate_SVD(vertices)  # Perform SVD on vertices
        save_A_tilde_to_csv(A_tilde, 'A_tilde.csv')  # Save A_tilde matrix to CSV
        scaled_A_tilde_remove_Z, x_min, x_max, y_min, y_max = scale_coordinates(A_tilde)  # Scale coordinates
        colors_R = colors[:, 0]  # Extract R values from colors
        save_scaled_coordinates_and_R(scaled_A_tilde_remove_Z, colors_R, 'Trans_XY_and_R.csv')  # Save scaled coordinates and R values to CSV
        img = visualize_image(scaled_A_tilde_remove_Z, colors_R.reshape(-1, 1), 'output_image-2D.png', int(np.max(scaled_A_tilde_remove_Z[:, 0])) + 50, int(np.max(scaled_A_tilde_remove_Z[:, 1])) + 60)  # Visualize initial image
        if img is None:
            print("Failed to generate the initial image.")
            return
        rotated_XY = rotate_coordinates(scaled_A_tilde_remove_Z, -50)  # Rotate coordinates
        colors_R_binary = apply_custom_threshold(colors_R.reshape(-1, 1), 44)  # Apply custom threshold to R values
        img_enhanced = visualize_image(rotated_XY, colors_R_binary, 'output_image-enhanced.png', int(np.max(rotated_XY[:, 0])) + 50, int(np.max(rotated_XY[:, 1])) + 60)  # Visualize enhanced image
        if img_enhanced is None:
            print("Failed to generate the enhanced image.")
            return
        img_filled = fill_holes(img_enhanced)  # Fill holes in the image
        if img_filled is None:
            print("Failed to fill holes in the image.")
            return
        detect_edges_and_draw_lines(img_filled)  # Detect edges and draw lines

if __name__ == "__main__":
    # Execute the main function
    main()

