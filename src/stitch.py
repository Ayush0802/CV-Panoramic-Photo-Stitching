import numpy as np
import cv2
import utils

def RANSAC(matched_pairs, prev_shift):
    matched_pairs = np.asarray(matched_pairs)
    
    use_random = True if len(matched_pairs) > utils.RANSAC_K else False

    best_shift = []
    K = utils.RANSAC_K if use_random else len(matched_pairs)
    threshold_distance = utils.RANSAC_THRES_DISTANCE
    
    max_inliner = 0
    for k in range(K):
        # Random pick a pair of matched feature
        idx = int(np.random.random_sample()*len(matched_pairs)) if use_random else k
        sample = matched_pairs[idx]
        
        # fit the warp model
        shift = sample[1] - sample[0]
        
        # calculate inliner points
        shifted = matched_pairs[:,1] - shift
        difference = matched_pairs[:,0] - shifted
        
        inliner = 0
        for diff in difference:
            if np.sqrt((diff**2).sum()) < threshold_distance:
                inliner = inliner + 1
        
        if inliner > max_inliner:
            max_inliner = inliner
            best_shift = shift

    return best_shift


# Stitch the 2 image together using their matched features
def stitching(img1, img2, shift, pool, blending=True):
    padding = [
        (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, -shift[1]),
        (0, 0)
    ]
    shifted_img1 = np.lib.pad(img1, padding, 'constant', constant_values=0)

    # Cut out unnecessary region
    split = img2.shape[1] + abs(shift[1])
    splited = shifted_img1[:, split:] if shift[1] > 0 else shifted_img1[:, :-split]
    shifted_img1 = shifted_img1[:, :split] if shift[1] > 0 else shifted_img1[:, -split:]

    h1, w1, _ = shifted_img1.shape
    h2, w2, _ = img2.shape
    
    inv_shift = [h1 - h2, w1 - w2]
    inv_padding = [
        (inv_shift[0], 0) if shift[0] < 0 else (0, inv_shift[0]),
        (inv_shift[1], 0) if shift[1] < 0 else (0, inv_shift[1]),
        (0, 0)
    ]
    shifted_img2 = np.lib.pad(img2, inv_padding, 'constant', constant_values=0)

    direction = 'left' if shift[1] > 0 else 'right'

    if blending:
        seam_x = shifted_img1.shape[1] // 2
        tasks = [
            (shifted_img1[y], shifted_img2[y], seam_x, utils.BLEND_WIDTH, direction) for y in range(h1)
        ]
        blended_rows = pool.starmap(blend, tasks)
        shifted_img1 = np.asarray(blended_rows)
        shifted_img1 = np.concatenate((shifted_img1, splited) if shift[1] > 0 else (splited, shifted_img1), axis=1)

    else:
        raise ValueError('Blending must be True for this stitching function.')

    return shifted_img1


# Blending Algorithm
def blend(row1, row2, seam_x, width, direction='left'):
    if direction == 'right':
        row1, row2 = row2, row1

    new_row = np.zeros(shape=row1.shape, dtype=np.uint8)

    for x in range(len(row1)):
        if x < seam_x - width:
            new_row[x] = row2[x]
        elif seam_x - width <= x <= seam_x + width:
            alpha = (x - (seam_x - width)) / (2 * width)
            new_row[x] = alpha * row1[x] + (1 - alpha) * row2[x]
        else:
            new_row[x] = row1[x]

    return new_row


# Apply end-to-end Alignment
def end2end_align(img, shifts):
    sum_y, sum_x = np.sum(shifts, axis=0)

    y_shift = np.abs(sum_y)
    col_shift = None

    # same sign
    if sum_x*sum_y > 0:
        col_shift = np.linspace(y_shift, 0, num=img.shape[1], dtype=np.uint16)
    else:
        col_shift = np.linspace(0, y_shift, num=img.shape[1], dtype=np.uint16)

    aligned = img.copy()
    for x in range(img.shape[1]):
        aligned[:,x] = np.roll(img[:,x], col_shift[x], axis=0)

    return aligned


# Crop the Black portion of the Image
def crop(img):
    _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    upper, lower = [-1, -1]

    black_pixel_num_threshold = img.shape[1]//100

    for y in range(thresh.shape[0]):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            upper = y
            break
        
    for y in range(thresh.shape[0]-1, 0, -1):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            lower = y
            break

    return img[upper:lower, :]