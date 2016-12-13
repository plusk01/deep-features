import sys
import cv2
import numpy as np
import random


def blend_image(img1, img2, img3):
    r_1,g_1,b_1 = cv2.split(img1)
    r_2,g_2,b_2 = cv2.split(img2)       
    r_3,g_3,b_3 = cv2.split(img3)
    rs = [r_1, r_2, r_3] 
    gs = [g_1, g_2, g_3] 
    bs = [b_1, b_2, b_3] 
    
    b = b_1
    g = g_1
    r = r_1

    for y in range(len(b_1)):
        for x in range(len(b_1[0])):
            b[y][x] = max(b_1[y][x], b_2[y][x], b_3[y][x])
            g[y][x] = max(g_1[y][x], g_2[y][x], g_3[y][x])
            r[y][x] = max(r_1[y][x], r_2[y][x], r_3[y][x])
    
    rgb_img = cv2.merge([r,g,b])

    return rgb_img


def warp_image(img, homography, x_max, y_max, title):
    h,w = img.shape[:2]
    h = y_max + y_max + h
    w = x_max + y_max + w

    translate = np.matrix([
        [1, 0, x_max] ,
        [0, 1, y_max] ,
        [0, 0, 1] ,
        ])

    M = translate * homography
    dst = cv2.warpPerspective(img, M, (w,h))
    cv2.imwrite(title, dst)

    return dst


def find_max(x_max_12, y_max_12, x_max_32, y_max_32):
    x_max = max(x_max_12, x_max_32)
    y_max = max(y_max_12, y_max_32)

    return x_max, y_max


def find_homography(good, kp_s, kp_t): 
    A = np.matrix([0])

    while(np.linalg.det(A) == 0):
        points = []
        for i in range(4):
            rnd = random.randint(1, len(good)-1)
            points.append((int(kp_s[good[rnd][0].queryIdx].pt[0]), int(kp_s[good[rnd][0].queryIdx].pt[1])))
            points.append((int(kp_t[good[rnd][0].trainIdx].pt[0]), int(kp_t[good[rnd][0].trainIdx].pt[1])))

        A = np.matrix([
            [points[0][0], points[0][1], 1, 0, 0, 0, -points[1][0] * points[0][0], -points[1][0] * points[0][1]] ,
            [0, 0, 0, points[0][0], points[0][1], 1, -points[1][1] * points[0][0], -points[1][1] * points[0][1]] ,
            [points[2][0], points[2][1], 1, 0, 0, 0, -points[3][0] * points[2][0], -points[3][0] * points[2][1]] ,
            [0, 0, 0, points[2][0], points[2][1], 1, -points[3][1] * points[2][0], -points[3][1] * points[2][1]] ,
            [points[4][0], points[4][1], 1, 0, 0, 0, -points[5][0] * points[4][0], -points[5][0] * points[4][1]] ,
            [0, 0, 0, points[4][0], points[4][1], 1, -points[5][1] * points[4][0], -points[5][1] * points[4][1]] ,
            [points[6][0], points[6][1], 1, 0, 0, 0, -points[7][0] * points[6][0], -points[7][0] * points[6][1]] ,
            [0, 0, 0, points[6][0], points[6][1], 1, -points[7][1] * points[6][0], -points[7][1] * points[6][1]] 
            ])

        x = np.matrix([
            [points[1][0]] , 
            [points[1][1]] , 
            [points[3][0]] , 
            [points[3][1]] , 
            [points[5][0]] , 
            [points[5][1]] , 
            [points[7][0]] , 
            [points[7][1]]
            ])

    h = A.I * x
    homography = np.matrix([
        [h.item(0), h.item(1), h.item(2)] ,
        [h.item(3), h.item(4), h.item(5)] ,
        [h.item(6), h.item(7), 1]
        ])

    return homography


def distance(outcome, target):
    return np.linalg.norm(outcome-target)


def ransac(good, kp_s, img_b, kp_t, img_t):
    best_consensus = 0
    tolrance = 2.0
    best_homography = np.matrix([0])
    
    for i in range(50):
        homography = find_homography(good, kp_s, kp_t);
        consensus = 0

        for i in range(len(good)):
            s = (int(kp_s[good[i][0].queryIdx].pt[0]), int(kp_s[good[i][0].queryIdx].pt[1]))
            t = (int(kp_t[good[i][0].trainIdx].pt[0]), int(kp_t[good[i][0].trainIdx].pt[1]))

            source = np.matrix([
                [s[0]] ,
                [s[1]] ,
                [1]
                ])

            target = np.matrix([
                [t[0]] ,
                [t[1]] ,
                [1]
                ])

            outcome = homography * source
            outcome = outcome / outcome.item(2)
            dist = distance(outcome, target)
            if dist < tolrance:
                consensus+=1

        if consensus > best_consensus:
            best_consensus = consensus
            best_homography = homography

    return best_homography

def test_points(img, homography):
    h,w = img.shape[:2]
    # (0,0)
    source = np.matrix([
        [0] ,
        [0] ,
        [1]
        ])
    top_left = homography * source
    top_left = top_left / top_left.item(2,0)

    # (0,H-1)
    source = np.matrix([
        [0] ,
        [h-1] ,
        [1]
        ])
    top_right = homography * source
    top_right = top_right / top_right.item(2,0)

    # (W-1,0)
    source = np.matrix([
        [w-1] ,
        [0] ,
        [1]
        ])
    bottom_left = homography * source
    bottom_left = bottom_left / bottom_left.item(2,0)
    
    # (W-1,H-1)
    source = np.matrix([
        [w-1] ,
        [h-1] ,
        [1]
        ])
    bottom_right= homography * source
    bottom_right = bottom_right / bottom_right.item(2,0)

    max_x = max(abs(top_left[0]), abs(top_right[0]), abs(bottom_left[0]), abs(bottom_right[0]))
    max_y = max(abs(top_left[1]), abs(top_right[1]), abs(bottom_left[1]), abs(bottom_right[1]))

    return max_x, max_y


def find_matches(kp_s, des_s, img_b, kp_t, des_t, img_t, title):
    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(des_s, des_t, k=2)

    # Apply ratio test
    good = []
    for m,n in knn_matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    matches = cv2.drawMatchesKnn(img_b, kp_s, img_t, kp_t, good, None, flags=2)
    cv2.imwrite(title, matches)

    return good


def find_points(img, title):
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(img, None)
    # draw the keypoints on the images
    keypoints = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(title, keypoints)

    return kp, des


if __name__ == '__main__':
    # read in images
    img1 = cv2.imread('tests/test_images/campus1.png')
    img2 = cv2.imread('tests/test_images/campus2.png')
    img3 = cv2.imread('tests/test_images/campus3.png')

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    
    # find key points and descriptors
    kp1, des1 = find_points(img1_gray, 'key1.png')
    kp2, des2 = find_points(img2_gray, 'key2.png')
    kp3, des3 = find_points(img3_gray, 'key3.png')
    
    # find matches
    good_12 = find_matches(kp1, des1, img1_gray, kp2, des2, img2_gray, 'matches_1_2.png')
    good_13 = find_matches(kp1, des1, img1_gray, kp3, des3, img3_gray, 'matches_1_3.png')
    good_32 = find_matches(kp3, des3, img3_gray, kp2, des2, img2_gray, 'matches_3_2.png')

    # ransac and warp images
    homography_12 = ransac(good_12, kp1, img1_gray, kp2, img2_gray)
    homography_32 = ransac(good_32, kp3, img3_gray, kp2, img2_gray)

    x_max_12, y_max_12 = test_points(img1_gray, homography_12)
    x_max_32, y_max_32 = test_points(img1_gray, homography_12)

    # find min, max x and y
    x_max,y_max = find_max(x_max_12, y_max_12, x_max_32, y_max_32)

    warp_1 = warp_image(img1, homography_12, x_max, y_max, 'warp_12.png')
    warp_2 = warp_image(img3, homography_32, x_max, y_max, 'warp_32.png')

    homography = np.matrix([
        [1, 0, 0] ,
        [0, 1, 0] ,
        [0, 0, 1]
        ])

    warp_3 = warp_image(img2, homography, x_max, y_max, 'warp_base.png')
    blended = blend_image(warp_1, warp_2, warp_3)

    # cv2.imwrite('test.png', warp_1/3 + warp_2/3 + warp_3/3)
    cv2.imwrite('test.png', blended)



