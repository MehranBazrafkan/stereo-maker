import cv2
import uuid
import tempfile
import numpy as np
from flask_cors import CORS
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
CORS(app)

# ---------------------------
# Utility functions
# ---------------------------

def center_square_crop_and_resize(img, size=1024):
    h, w = img.shape[:2]
    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - side // 2, cy - side // 2
    crop = img[y1:y1+side, x1:x1+side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

def approximate_undistort(img, k1=-0.25):
    h, w = img.shape[:2]
    fx = fy = 0.8 * w
    cx, cy = w / 2, h / 2
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    dist = np.array([k1, 0, 0, 0, 0])
    return cv2.undistort(img, K, dist)

def match_color_lab(source, reference):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    matched_lab = np.zeros_like(source_lab)
    for i in range(3):
        src = source_lab[:,:,i].ravel()
        ref = ref_lab[:,:,i].ravel()
        src_hist, _ = np.histogram(src, 256, [0,256])
        ref_hist, _ = np.histogram(ref, 256, [0,256])
        src_cdf = np.cumsum(src_hist).astype(np.float64); src_cdf /= src_cdf[-1]
        ref_cdf = np.cumsum(ref_hist).astype(np.float64); ref_cdf /= ref_cdf[-1]
        lut = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_idx in range(256):
            while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
                ref_idx += 1
            lut[src_idx] = ref_idx
        matched_lab[:,:,i] = lut[source_lab[:,:,i]]
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

# ---------------------------
# Flask route
# ---------------------------

@app.route("/stereo", methods=["POST"])
def create_stereo():
    if 'left' not in request.files or 'right' not in request.files:
        return jsonify({"error": "Please upload 'left' and 'right' images"}), 400

    # Read uploaded files into OpenCV images
    left_file = request.files['left']
    right_file = request.files['right']

    left_bytes = np.frombuffer(left_file.read(), np.uint8)
    right_bytes = np.frombuffer(right_file.read(), np.uint8)
    img_left = cv2.imdecode(left_bytes, cv2.IMREAD_COLOR)
    img_right = cv2.imdecode(right_bytes, cv2.IMREAD_COLOR)

    # Preprocess: center crop & resize
    img_left  = center_square_crop_and_resize(img_left, 1024)
    img_right = center_square_crop_and_resize(img_right, 1024)

    # Undistort
    img_left_ud = approximate_undistort(img_left)

    # ORB feature matching
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(img_left_ud, None)
    kp2, des2 = orb.detectAndCompute(img_right, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        return jsonify({"error": "Not enough good feature matches found"}), 400

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp
    h, w = img_right.shape[:2]
    img_left_warped = cv2.warpPerspective(img_left_ud, H, (w, h))

    # Crop common area
    gray = cv2.cvtColor(img_left_warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    left_crop  = img_left_warped[y:y+h, x:x+w]
    right_crop = img_right[y:y+h, x:x+w]

    # Color match
    left_color_matched = match_color_lab(left_crop, right_crop)

    # Combine stereo
    stereo = np.hstack((left_color_matched, right_crop))

    # Save to temporary file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmp_file.name, stereo)
    tmp_file.close()

    # Return the image
    return send_file(tmp_file.name, mimetype="image/png", as_attachment=True, download_name=f"stereo-{uuid.uuid4()}.png")

# ---------------------------
# Run Flask
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)