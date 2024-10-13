import numpy as np
import os
from PIL import Image
import cv2
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from BP_DP_Functions_SVD import (load_mesh_glb, svd_plane_detection, hough_plane_detection, visualize_3Dplanes,
                              calculate_SVD, scale_coordinates, visualize_image, find_colors_for_plane_points,
                              rotate_coordinates, apply_otsu_threshold, apply_custom_threshold, fill_holes,
                              load_mesh_ply, calculate_distance_transform, calculate_median_in_range,
                              plot_histogram, extract_local_maxima_values,
                              interactive_plot_with_points,
                              detect_edges_and_lines, draw_lines, convert_to_black_and_white, load_mesh_obj,
                              extract_colored_points, cluster_hsv, plot_hsv_clusters
                              )


# Recover the scale from scaled coordinates compared with Main_0.py
# Add planar detection compared with Main.py
# Binary the planar

def main():
    # Main function to execute all steps
    file_path = 'chessboard.glb'
    # file_path = '0_4_8_10_12_14_18_22_26_28.glb'
    # file_path = 'Chessborad-senjian/chessboard3_Liam.ply'
    # Determine the file format
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.glb':
        vertices, colors = load_mesh_glb(file_path)  # Load mesh from .glb file
        # pn = len(colors)
        # print("Points Num: ", pn)
    elif file_extension == '.ply':
        vertices, colors = load_mesh_ply(file_path)  # Load mesh from .ply file
        # pn = len(colors)
        # print("Points Num: ", pn)
    elif file_extension == '.obj':
        vertices, colors = load_mesh_obj(file_path)  # Load mesh from .obj file
    else:
        print("Unsupported file format")
        return

    ###### SVD Plane Detection
    plane_points_svd = svd_plane_detection(vertices, colors, 0.15)
    # chessboard10_ql.ply(4539850)--threshold:0.65, Detected points percentage:0.829
    # chessboard9_jh.ply(12996670)--detecting plane points takes too much time(>2h)
    # chessboard_jw8.ply(13734685)--detecting plane points takes too much time(>2h)
    # chessboard_yl7.ply(619320)--threshold:0.65, Detected points percentage:0.9056;threshold:0.6, Detected points percentage:0.833
    # chessboard6_zec.ply(789369)--threshold:0.65, Detected points percentage:0.897;threshold:0.6, Detected points percentage:0.8166
    # chessboard5_PH.ply(4132323)--threshold:0.65, Detected points percentage:0.853
    # chessboard4_DP.ply(717907)--threshold:0.65, Detected points percentage:0.8985;threshold:0.6, Detected points percentage:0.818;
    # chessboard3_Liam.ply(51515)--threshold:0.65, Detected points percentage:0.959;hreshold:0.5, Detected points percentage:0.7936

    pn = len(vertices)
    print("Points number:", pn)
    dpn = len(plane_points_svd)
    print("SVD detected plane points num:", dpn)
    print("Detected points percentage: ", dpn/pn)
    ##### Hough Plane Detection
    # plane_points_hough = hough_plane_detection(vertices, 0.01)
    # print("Hough detected plane points:", len(plane_points_hough))

    colored_plane_points = extract_colored_points(vertices, colors, plane_points_svd)
    cpn = len(colored_plane_points)
    print("Colored Points number:", cpn)

    # Find colors for plane points
    o_plane_colors_svd = find_colors_for_plane_points(plane_points_svd, vertices, colors)
    o_colors_hsv = cv2.cvtColor(np.uint8([o_plane_colors_svd]), cv2.COLOR_RGB2HSV)[0]
    # # Create a DataFrame with the data
    # df = pd.DataFrame({
    #     'Point X': [p[0] for p in plane_points_svd],
    #     'Point Y': [p[1] for p in plane_points_svd],
    #     'Point Z': [p[2] for p in plane_points_svd],
    #     'RGB R': [c[0] for c in o_plane_colors_svd],
    #     'RGB G': [c[1] for c in o_plane_colors_svd],
    #     'RGB B': [c[2] for c in o_plane_colors_svd],
    #     'HSV H': [hsv[0] for hsv in o_colors_hsv],
    #     'HSV S': [hsv[1] for hsv in o_colors_hsv],
    #     'HSV V': [hsv[2] for hsv in o_colors_hsv],
    # })

    # # Perform K-Means clustering on HSV values
    # n_clusters = 5  # You can adjust the number of clusters as needed
    # df_clustered = cluster_hsv(df, n_clusters=n_clusters)
    #
    # # Visualize the HSV clusters in an interactive 3D plot
    # plot_hsv_clusters(df_clustered)
    #
    #
    # # Save the DataFrame to an Excel file
    # file_path = "colored_points_output.xlsx"
    # df.to_excel(file_path, index=False)


    # plane_colors_svd = find_colors_for_plane_points(colored_plane_points, vertices, colors)
    plane_colors_svd = find_colors_for_plane_points(o_plane_colors_svd, vertices, colors)

    plane_binary_colors = convert_to_black_and_white(plane_colors_svd)

    # Print or use the plane_points_svd and plane_colors_svd as needed
    # print("Plane points (SVD):", plane_points_svd)
    # print("Plane colors (SVD):", plane_colors_svd)
    # visualize_3Dplanes(vertices, [plane_points_svd], 'SVD')

    if plane_points_svd is not None and plane_colors_svd is not None:
        # save_vertices_colors_to_csv(vertices, colors, 'first_mesh_vertices_colors.csv')  # Save vertices and colors to CSV
        A_tilde, U, sigma = calculate_SVD(plane_points_svd)  # Perform SVD on vertices
        # save_A_tilde_to_csv(A_tilde, 'A_tilde.csv')  # Save A_tilde matrix to CSV
        scaled_A_tilde_remove_Z, s = scale_coordinates(A_tilde)  # Scale coordinates
        colors_R = plane_binary_colors[:, 0]  # Extract R values from colors
        # save_scaled_coordinates_and_R(scaled_A_tilde_remove_Z, colors_R, 'Trans_XY_and_R.csv')  # Save scaled coordinates and R values to CSV
        img = visualize_image(scaled_A_tilde_remove_Z, colors_R.reshape(-1, 1), 'output_image-2D.png', int(np.max(scaled_A_tilde_remove_Z[:, 0])) + 50, int(np.max(scaled_A_tilde_remove_Z[:, 1])) + 60)  # Visualize initial image
        if img is None:
            print("Failed to generate the initial image.")
            return

        rotated_XY = rotate_coordinates(scaled_A_tilde_remove_Z, 0)  # -50, Rotate coordinates

        thre = np.median(colors_R)
        print("Color threshold: ", thre)

        # colors_R_binary = apply_otsu_threshold(colors_R.reshape(-1, 1))  #### how???
        colors_R_binary = apply_custom_threshold(colors_R.reshape(-1, 1),
                                                 thre)  # 44, 128 - Apply custom threshold to R values
        img_enhanced = visualize_image(rotated_XY, colors_R_binary, 'output_image-enhanced.png',
                                       int(np.max(rotated_XY[:, 0])) + 50,
                                       int(np.max(rotated_XY[:, 1])) + 60)  # Visualize enhanced image
        if img_enhanced is None:
            print("Failed to generate the enhanced image.")
            return
        img_filled = fill_holes(img_enhanced)  # Fill holes in the image
        if img_filled is None:
            print("Failed to fill holes in the image.")
            return

        ########################################################################
        image_array, lines = detect_edges_and_lines(img_filled)  # Detect edges and lines

        if lines is not None:
            line_array, img_lines = draw_lines(image_array, lines)  # Draw lines on the edge-detected image
        else:
            print("Edge and line detection failed.")

        # Step 1: Calculate distance transform
        dist_transform = calculate_distance_transform(img_lines)
        # dist_transform = calculate_distance_transform(img_morph)

        # 显示距离变换结果
        dist_transform_image = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        Image.fromarray(dist_transform_image).show()

        # Step 2: Visualize 3D surface
        # plot_3d_surface(dist_transform)

        # Step 3: Find and extract local maxima values
        peak_values, K = extract_local_maxima_values(dist_transform)
        M = K + line_array
        # M = K + img_morph
        Image.fromarray((M * 255).astype(np.uint8)).show()  # Display the M matrix

        print("Max peak: ", np.max(peak_values))
        print("Min peak: ", np.min(peak_values))
        # Step 4: Plot histogram of local maxima values
        hist, max_bin, max_freq = plot_histogram(peak_values)

        # Step 5: Calculate median of values within specified range
        # lower_bound, upper_bound = 13, 18  # Example bounds, adjust as needed
        lower_bound, upper_bound = max_bin[0], max_bin[1]
        median_value = calculate_median_in_range(peak_values, lower_bound, upper_bound)

        grid_length = 2 * median_value / s
        # grid_length = 2 * 25 / s
        print("Grid Length: " + str(grid_length) + " pixels")

        physical_length = 0.0175
        # physical_length = 0.012  # unit: m
        print("Physical Length: "+str(physical_length)+" m")

        scale = physical_length / grid_length
        print("Scale:  " + str(scale) + " m/pixel")

        # Call the interactive plot function
        interactive_plot_with_points(line_array)

    else:
        print("No points or colors can be detected.")

if __name__ == "__main__":
    main()
