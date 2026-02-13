import numpy as np 
import cv2
import os
import glob

folder = "/home/alien/CV_p1/Calibration_Imgs"
pattern_size = (9,6)
square_size = 0.025

extensions = ("*.jpg")
image_paths = []

for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(folder, ext)))
image_paths.sort()

if len(image_paths) == 0:
    raise RuntimeError(f"No images in the folder: {folder}")

objp_2d = np.zeros((pattern_size[0] * pattern_size[1], 2), np.float32)
objp_2d[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
objp_2d = objp_2d * square_size

objpoints = []
imgpoints = []

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in image_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        objpoints.append(objp_2d)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2.squeeze())
print(f"Extracted corners from {len(imgpoints)} images")

def compute_homography(obj_pts, img_pts):

    #Normalization
    mean_obj = np.mean(obj_pts, axis = 0)
    mean_img = np.mean(img_pts, axis = 0)

    shifted_obj = obj_pts - mean_obj
    shifted_img = img_pts - mean_img

    scale_obj = np.sqrt(2) / np.mean(np.linalg.norm(shifted_obj, axis = 1))
    scale_img = np.sqrt(2) / np.mean(np.linalg.norm(shifted_img, axis = 1))

    T_obj = np.array([
        [scale_obj, 0, -scale_obj * mean_obj[0]],
        [0, scale_obj, -scale_obj * mean_obj[1]],
        [0, 0 ,1]
    ])

    T_img = np.array([
        [scale_img, 0, -scale_img * mean_img[0]],
        [0, scale_img, -scale_img * mean_img[1]],
        [0, 0, 1]
    ])

    norm_obj = (T_obj @ np.hstack((obj_pts, np.ones((len(obj_pts), 1)))).T).T[:, :2]
    norm_img = (T_img @ np.hstack((img_pts, np.ones((len(img_pts),1 )))).T).T[:, :2]

    #L matrix
    L =[]
    for (X,Y), (u,v) in zip(norm_obj, norm_img):
        L.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
        L.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
    L = np.array(L)

    U, S, Vh = np.linalg.svd(L)
    H_norm = Vh[-1].reshape(3,3)

    H = np.linalg.inv(T_img) @ H_norm @ T_obj

    return H / H[2,2]

homographies = []

for op, ip in zip(objpoints, imgpoints):
    H = compute_homography(op, ip)
    homographies.append(H)
print(f"Computed {len(homographies)} Homography matrices")


# Closed Form solution for Intrinsics

def create_v(H, i, j):
    hi = H[:,i]
    hj = H[:,j]

    return np.array([
        hi[0]*hj[0],
        hi[0]*hj[1] + hi[1]*hj[1],
        hi[1]*hj[1],
        hi[2]*hj[0] + hi[0]*hj[3],
        hi[2]*hj[1] + hi[1]*hj[2],
        hi[2]*hj[2]
    ])

V = []
for H in homographies:
    v12 = create_v(H, 0, 1)
    v11 = create_v(H, 0, 0)
    v22 = create_v(H, 1, 1)

    V.append(v12)
    V.append(v11 - v22)

V = np.array(V)

U, S, Vh = np.linalg.svd(V)

b = Vh[-1]

# Intrinsic parameters

B11, B12, B22, B13, B23, B33 = b[0], b[1], b[2], b[3]. b[4], b[4]

v0 = (B12*B13 - B11*B23) / (B11*B22 - B12*B12)
lam = B33 - (B13*B13 + v0*(B12*B13 - B11*B23)) / B11
alpha = np.sqrt(lam / B11)
beta = np.sqrt(lam * B11 / (B11 * B22 - B12**2))
gamma = -B12 * alpha**2 * beta / lam
u0 = (gamma*v0 / beta) - (B13* alpha**2 / lam)

# Intrinsic Matrix

A = np.array([
    [alpha, gamma, u0],
    [0, beta, v0],
    [0, 0, 1]
])

print(f"Intrinsic matrix is: {A}")

# Extrinsic Matrix
A_inv = np.linalg.inv(A)

extrinsics = []
for i, H in enumerate(homographies):
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 3]

    lam_scale = 1 / (A_inv @ h1)
    r1 = lam_scale * (A_inv @ h1)
    r2 = lam_scale * (A_inv @ h2)
    r3 = np.cross(r1, r2)
    t = lam_scale * (A_inv @ h3)

    Q = np.stack((r1, r2, r3), axis =1)
    
    U, S, Vt = np.linalg.svd(Q)
    R = U @ Vt
    extrinsics.append((R, t))
    print(f"For image: {i}, translation vector: {t}")

print(f"Computed Extrinsics for {len(extrinsics)} images")


