from sift.sift_module import wrapper_SIFT
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
MIN_MATCH_COUNT = 5

# reading images in rgb so we return images in rgb while all our work will be in gray scale



def sift_wrap(face_image_rgb,reference_rgb):
    face_image = cv2.cvtColor(face_image_rgb, cv2.COLOR_BGR2GRAY)    # face image

    if face_image.shape[0] > 200 or face_image.shape[1] > 200:
        face_image = cv2.resize(face_image,(face_image.shape[0]//2,face_image.shape[1]//2))
        face_image_rgb = cv2.resize(face_image_rgb,(face_image_rgb.shape[0]//2,face_image_rgb.shape[1]//2))

    reference_image = cv2.cvtColor(reference_rgb, cv2.COLOR_BGR2GRAY)  # group image
    if reference_image.shape[0] > 1200 or reference_image.shape[1] > 1200:
        reference_image = cv2.resize(reference_image,(reference_image.shape[0]//2,reference_image.shape[1]//2))
        reference_rgb = cv2.resize(reference_rgb,(reference_rgb.shape[0]//2,reference_rgb.shape[1]//2))


    key1,desc1 = wrapper_SIFT(face_image)
    key2,desc2 = wrapper_SIFT(reference_image)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([ key1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ key2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        print(M)

        img1_mask = cv2.warpPerspective(np.ones_like(face_image)*255 , M, (reference_image.shape[1], reference_image.shape[0]))
        
        img1_transformed = cv2.warpPerspective(face_image_rgb, M, (reference_image.shape[1], reference_image.shape[0]))
        io.imsave('sift/images/img1_transformed.jpg', img1_transformed)
        plt.imshow(img1_transformed) #mask with image
        plt.show()
        
        img1_mask = img1_mask/255
        io.imsave('sift/images/img1_mask.jpg', img1_mask)
        plt.imshow(img1_mask) #mask with image
        plt.show()

        # putting the needed face on referenced image so we can have background while blending
        reference_rgb[img1_mask == 1,:] = img1_transformed[img1_mask == 1,:]

        cv2.imwrite('sift/images/reference_w_mfatah.jpg', reference_rgb)
        plt.imshow(reference_rgb) #mask with image
        plt.show()
        
        # Draw detected template in scene image
        h, w = face_image.shape
        pts = np.float32([[0, 0],
                        [0, h - 1],
                        [w - 1, h - 1],
                        [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
    
        reference_image = cv2.polylines(reference_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = face_image.shape
        h2, w2 = reference_image.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = face_image
            newimg[:h2, w1:w1 + w2, i] = reference_image
        x = []
        lines=[]
        y = []
        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(key1[m.queryIdx].pt[0]), int(key1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(key2[m.trainIdx].pt[0] + w1), int(key2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))
            lines.append([pt1, pt2])
        
        
        
        #get angle of the line
        print(len(lines),lines[0], lines[1], lines[2], lines[3])
        plt.imshow(newimg)
        plt.show()
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    
    return reference_rgb, img1_mask
