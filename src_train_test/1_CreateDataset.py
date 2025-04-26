import cv2
import numpy as np
import os
import random

def align_images_method_ORB(reference, image):
    # Convert images to grayscale
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use ORB detector for feature matching
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray_ref, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_img, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched points
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp image to align with reference
    aligned_image = cv2.warpPerspective(image, H, (reference.shape[1], reference.shape[0]))
    return aligned_image

def align_images_method_ECC(reference, image):
    # Convert images to grayscale
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, _, red_ref = cv2.split(reference)
    _, _, red_img = cv2.split(image)

    gray_ref = red_ref.copy()
    gray_img = red_img.copy()

    # ECC alignment
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
    try:
        cc, warp_matrix = cv2.findTransformECC(gray_ref, gray_img, warp_matrix, warp_mode, criteria)
        # Warp image to align with reference
        aligned_image = cv2.warpAffine(image, warp_matrix, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except:
        cc = 0
        aligned_image = None

    return cc, aligned_image

def extract_object(reference, image, mask, fileindex):
    # Convert images to grayscale
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, _, red_ref = cv2.split(reference)
    _, _, red_img = cv2.split(image)

    gray_ref = red_ref.copy()
    gray_img = red_img.copy()

    gray_ref = cv2.bitwise_and(gray_ref, gray_ref, mask=mask)
    gray_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    # Compute absolute difference
    diff = cv2.absdiff(gray_img, gray_ref)
    cv2.imwrite(f'../Images/Diff/diff_{fileindex:03d}.png', diff)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask and bounding box
    mask = np.zeros_like(gray_img)
    x, y, w, h = 0, 0, 0, 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Create transparent image
    extracted = cv2.bitwise_and(image, image, mask=mask)
    b, g, r = cv2.split(extracted)
    alpha = mask
    extracted_rgba = cv2.merge((b, g, r, alpha))

    return extracted_rgba, (x, y, w, h)

def rotate_and_merge(reference, object_rgba, bbox, angles=[0, 90, 180, 270]):
    x, y, w, h = bbox
    obj = object_rgba[y:y+h, x:x+w]

    results = []
    for angle in angles:
        # Compute new bounding box dimensions based on the absolute rotation matrix values
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int(h * sin_val + w * cos_val)
        new_h = int(h * cos_val + w * sin_val)

        # Adjust translation to fit the new bounding box
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        if new_w > 0 and new_h > 0:
            # Rotate with correct bounding box
            rotated = cv2.warpAffine(obj, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            # Ensure roi has the same shape as rotated
            overlay = reference.copy()
            img_h, img_w = overlay.shape[:2]
            roi_x, roi_y = max(0, x - (new_w - w) // 2), max(0, y - (new_h - h) // 2)
            roi_h, roi_w = rotated.shape[:2]

            if roi_y + roi_h >= img_h:
                roi_y = max(0, img_h - roi_h - 1)
                if roi_y==0 and roi_h >= img_h:
                    roi_h = img_h
                    rotated = rotated[0:roi_h,:]
            if roi_x + roi_w >= img_w:
                roi_x = max(0, img_w - roi_w - 1)
                if roi_x==0 and roi_w >= img_w:
                    roi_w = img_w
                    rotated = rotated[:,0:roi_w]

            # Adjust ROI dimensions to avoid mismatch
            roi = overlay[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            if roi.shape[:2] != rotated.shape[:2]:
                continue  # Skip if dimensions do not match

            # Blend transparent image
            alpha = rotated[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + rotated[:, :, c] * alpha
            overlay[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi
            results.append(overlay)

    return results

def make_fg_mask(reference):
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray_ref.shape[:2], dtype="uint8")
    ret, img_th = cv2.threshold(gray_ref, 70, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(img_th, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.bitwise_not(mask)

    return mask


PATH_REFERENCE_IMAGE = '../Images/empty.png'
PATH_GOOD_INPUT_IMAGES = '../Images/Originals/Peukjes'
PATH_ROTATED_GOOD_IMAGES = '../Images/Dataset'
PATH_EXTRACT = '../Images/Extract'
PATH_BAD_INPUT_IMAGES = '../Images/Originals/GeenPeukjes'
PATH_ROTATED_BAD_IMAGES = '../Images/Dataset'

def main():

    cwd = os.getcwd()
    print(cwd)
    if 'src_train_test' not in cwd:
        os.chdir('src_train_test')

    reference_img = cv2.imread(PATH_REFERENCE_IMAGE)
    href, wref = reference_img.shape[:2]
    empty_mask = make_fg_mask(reference_img)

    for category in ['Good', 'Bad']:
        if category == 'Good':
            fromdir = PATH_GOOD_INPUT_IMAGES
            todir = PATH_ROTATED_GOOD_IMAGES
            angles = np.arange(0, 350, 20)
        elif category == 'Bad':
            fromdir = PATH_BAD_INPUT_IMAGES
            todir = PATH_ROTATED_BAD_IMAGES
            angles = np.arange(0, 350, 45)
        else:
            continue

        for fileindex, fromfn in enumerate(os.listdir(fromdir)):
            fromdirfn = os.path.join(fromdir, fromfn)
            input_img = cv2.imread(fromdirfn)
            h, w = input_img.shape[:2]

            confidence, aligned_img = align_images_method_ECC(reference_img, input_img)
            print(f"ECC Score for {fileindex+1}:", confidence)

            if confidence > 0.6:
                object_rgba, bbox = extract_object(reference_img, aligned_img, empty_mask, fileindex)
                todirfn = os.path.join(PATH_EXTRACT, f'{fromfn}_{fileindex+1:03d}_extract.png')
                cv2.imwrite(todirfn, object_rgba)

                augmented_images = rotate_and_merge(reference_img, object_rgba, bbox, angles)
                #augmented_images = []

                for i, img in enumerate(augmented_images):
                    todirfn = os.path.join(todir, f'{category}_{fileindex+1:03d}_Augment_{i+1:02d}.png')
                    cv2.imwrite(todirfn, img)

if __name__ == '__main__':
    main()
