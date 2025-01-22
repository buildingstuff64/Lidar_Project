import cv2
import numpy as np

# Function to manually select points on an image
def select_points(image_path):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", img)

    # Load the image
    img = cv2.imread(image_path)
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(points, dtype=np.float32)

# Intrinsic parameters for the two cameras (example values, replace with actual calibration data)
camera_matrix1 = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs1 = np.zeros(5)  # Assuming no lens distortion

camera_matrix2 = np.array([[950, 0, 620], [0, 950, 340], [0, 0, 1]], dtype=np.float32)
dist_coeffs2 = np.zeros(5)  # Assuming no lens distortion

# Extrinsic parameters (example, replace with actual calibration results)
R = np.eye(3)  # Identity rotation for camera 1 (reference)
T = np.zeros((3, 1))  # Zero translation for camera 1

# Projection matrices for triangulation
proj_matrix1 = np.hstack((R, T))  # [R|T] for camera 1
proj_matrix1 = camera_matrix1 @ proj_matrix1

# Assuming camera 2 is offset along X-axis for this example
R2 = np.eye(3)
T2 = np.array([[0.5], [0], [0]])  # Translation: 0.5 meters along X-axis
proj_matrix2 = np.hstack((R2, T2))
proj_matrix2 = camera_matrix2 @ proj_matrix2

# Step 1: Select points from the two images
print("Select points from the first image.")
points1 = select_points("Left.jpg")
print("Select points from the second image.")
points2 = select_points("Right.jpg")

# Step 2: Triangulate 3D points
points1_homogeneous = np.vstack((points1.T, np.ones(points1.shape[0])))  # Convert to homogeneous
points2_homogeneous = np.vstack((points2.T, np.ones(points2.shape[0])))

# Triangulate points
points_4d_homogeneous = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1_homogeneous[:2], points2_homogeneous[:2])

# Convert to 3D points (Euclidean)
points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
points_3d = points_3d.T  # Transpose to shape (N, 3)
print("3D Points:")
print(points_3d)

# Step 3: Compute transformation from points_3d to second camera's frame
retval, R_est, T_est, inliers = cv2.solvePnPRansac(points_3d, points2, camera_matrix2, dist_coeffs2)

# Step 4: Validate results
print("Estimated Rotation Vector (Camera 1 to Camera 2):")
print(R_est)
print("Estimated Translation Vector (Camera 1 to Camera 2):")
print(T_est)

# Optional: Reproject points to verify alignment
projected_points, _ = cv2.projectPoints(points_3d, R_est, T_est, camera_matrix2, dist_coeffs2)
print("Reprojected Points on Camera 2:")
print(projected_points.reshape(-1, 2))

# Visualization (if needed)
image2 = cv2.imread("Right.jpg")
for point in projected_points:
    point = tuple(point.ravel().astype(int))
    cv2.circle(image2, point, 5, (0, 0, 255), -1)

cv2.imshow("Reprojected Points", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()