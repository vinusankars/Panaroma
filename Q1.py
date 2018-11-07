#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:32:41 2018

@author: vinusankars
"""

#References: 
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
#https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
#http://6.869.csail.mit.edu/fa17/lecture/lecture14sift_homography.pdf
#https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def crop(img, threshold=0):
    if len(img.shape) == 3:
        flat = np.max(img, 2)
    else:
        flat = img
    assert len(flat.shape) == 2

    rows = np.where(np.max(flat, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flat, 1) > threshold)[0]
        img = img[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        img = img[:1, :1]

    return img

def matchCompute(img1, img2, draw=0, n_match=10):
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    temp = bf.match(des1, des2)
    match = sorted(temp, key = lambda x:x.distance)
    
    if draw == 1:
        img3 = cv.drawMatches(img1, kp1, img2, kp2, match[:n_match], None, flags=6)
        plt.figure(figsize=(20,20))
        plt.imshow(cv.cvtColor(np.array(img3, dtype='uint8'), cv.COLOR_BGR2RGB))
        plt.show()
        
    return match, kp1, des1, kp2, des2
    

def homography(img1, img2, thresh=5, K=1000):
    H, maxInlier = [] , []
    match, kp1, d1, kp2, d2 = matchCompute(img1, img2)
    print('\nCalculating homography...')
    for K in range(1000):
        rand1 = np.random.randint(0, len(match))
        dmatch = match[rand1]
        i1 = dmatch.queryIdx
        i2 = dmatch.trainIdx
        (y1, x1) = kp1[i1].pt
        (y11, x11) = kp2[i2].pt
        
        rand2 = np.random.randint(0, len(match))
        dmatch = match[rand2]
        i1 = dmatch.queryIdx
        i2 = dmatch.trainIdx
        (y2, x2) = kp1[i1].pt
        (y21, x21) = kp2[i2].pt
        
        rand3 = np.random.randint(0, len(match))
        dmatch = match[rand3]
        i1 = dmatch.queryIdx
        i2 = dmatch.trainIdx
        (y3, x3) = kp1[i1].pt
        (y31, x31) = kp2[i2].pt
        
        rand4 = np.random.randint(0, len(match))
        dmatch = match[rand4]
        i1 = dmatch.queryIdx
        i2 = dmatch.trainIdx
        (y4, x4) = kp1[i1].pt
        (y41, x41) = kp2[i2].pt        
          
        mat = []
        mat.append([x1, y1, 1, 0, 0, 0, -x11*x1, -x11*y1, -x11])
        mat.append([0, 0, 0, x1, y1, 1, -y11*x1, -y11*y1, -y11])
        
        mat.append([x2, y2, 1, 0, 0, 0, -x21*x2, -x21*y2, -x21])
        mat.append([0, 0, 0, x2, y2, 1, -y21*x2, -y21*y2, -y21])
        
        mat.append([x3, y3, 1, 0, 0, 0, -x31*x3, -x31*y3, -x31])
        mat.append([0, 0, 0, x3, y3, 1, -y31*x3, -y31*y3, -y31])
        
        mat.append([x4, y4, 1, 0, 0, 0, -x41*x4, -x41*y4, -x41])
        mat.append([0, 0, 0, x4, y4, 1, -y41*x4, -y41*y4, -y41])
        
        mat = np.matrix(mat)
        u, s, v = np.linalg.svd(mat)
        h = np.reshape(v[8], (3,3))
        h = h/h.item(8)
        
        inlier = []
        for i in match:
            (n1, m1) = kp1[i.queryIdx].pt
            (n2, m2) = kp2[i.trainIdx].pt             
            
            p1 = np.transpose(np.matrix([m1, n1, 1]))
            estp2 = np.dot(h, p1)
            estp2 = estp2/estp2.item(2)
            
            p2 = np.transpose(np.matrix([m2, n2, 1]))
            error = p2-estp2
            sigma = thresh
            if np.linalg.norm(error) < sigma*(5.99)**0.5:
                inlier.append([m1, n1, m2, n2])
                
        if len(inlier) > len(maxInlier):
            maxInlier = inlier
            H = h
            
#        threshold = 0.6
#        if len(maxInlier) > threshold*len(match):
#            break
    print(H)
    print(len(maxInlier),len(match))
    return H

     
def warp(img1, img2, h, ksize):
    h = np.linalg.inv(h)
    h = h/h.item(2)
#    mat1 = np.transpose(np.dot(h, np.transpose(np.matrix([1, 0, 1]))))
#    mat1 = mat1/mat1.item(2)
#    mat2 = np.transpose(np.dot(h, np.transpose(np.matrix([1, img1.shape[1], 1]))))
#    mat2 = mat2/mat2.item(2)
    x_off = 500
    y_off = img1.shape[1]*2#max(int(abs(mat1.item(1))), int(abs(mat2.item(1))))
#    print(mat1, mat2)
    im = np.zeros((img1.shape[0]*3, img1.shape[1]*5, 3))
    
    print('\nTransforming...')
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            im[i+x_off][j+y_off] = img1[i][j]
                
#    plt.figure(figsize=(15,15))
#    plt.imshow(cv.cvtColor(np.array(im, dtype='uint8'), cv.COLOR_BGR2RGB))
#    plt.show()
    print('\nStitching...')
    
    for i in range(len(img2)):
        for j in range(len(img2[0])):
            if list(img2[i][j]) != [0,0,0]:
                mat = np.transpose(np.dot(h, np.transpose(np.matrix([i, j, 1]))))
                mat = mat/mat.item(2)
                try:
                    xx = int(mat.item(0))
                    yy = int(mat.item(1))
                    if xx+x_off-ksize>=0 and yy+y_off-ksize>=0:
                        if list(im[xx+x_off][yy+y_off]) == [0,0,0]:
                            im[xx+x_off-ksize:xx+x_off+ksize+1, yy+y_off-ksize:yy+y_off+ksize+1] = np.zeros((ksize*2+1,ksize*2+1,3), dtype='uint8')+img2[i][j]
                except:
                    continue
    
    print('\nBlending...')
    for i in range(10, len(im)-10):
        try:
            im[i][y_off] = np.average(np.average(im[i-10:i+11, y_off-10:y_off+11], 1),0).astype('uint8')
        except: 
            continue
    im = crop(im)
    plt.figure(figsize=(15,15))
    plt.imshow(cv.cvtColor(np.array(im, dtype='uint8'), cv.COLOR_BGR2RGB))
    plt.show()
    
    return im

img = []

for i in ['1.JPG','2.JPG', '3.JPG', '4.JPG']:
    temp = cv.imread(i, 1)
    img.append(temp)
    
h12 = homography(img[0], img[1])
img1 = warp(img[0], img[1], h12, 2).astype('uint8')

h23 = homography(img1, img[2])
img2 = warp(img1, img[2], np.dot(h12,h23), 2).astype('uint8')

h34 = homography(img2, img[3])
img3 = warp(img1, img[2], np.dot(h12,np.dot(h23,h34)), 2).astype('uint8')

'''============================
Part (d)
============================='''
xo = 1000
yo = 1000


'''-----------------------'''
M, k1, d1, k2, d2 = matchCompute(img[2], img[3])
src = np.float32([k1[m.queryIdx].pt for m in M]).reshape(-1,1,2)
dst = np.float32([k2[m.trainIdx].pt for m in M]).reshape(-1,1,2)
M, mask = cv.findHomography(src, dst, cv.RANSAC, 2*5.99**0.5)
print(M)
M = np.dot(np.matrix([[1,0,xo],[0,1,yo],[0,0,1]]), M)
matchesMask = mask.ravel().tolist()

img1 = cv.warpPerspective(img[2], M, (img[2].shape[1]*5, img[2].shape[0]*5))
for i in range(len(img[3])):
    for j in range(len(img[3][0])):
        if list(img1[i+xo][j+yo])==[0,0,0]:
            img1[i+xo][j+yo] = img[3][i][j]
#        else:
#            img1[i+xo][j+yo] = np.uint8(img[1][i][j]/2 + img1[i+xo][j+yo]/2)
            
plt.figure(figsize=(15,15))
img1 = crop(np.array(img1, dtype='uint8'))
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.show()

'''-----------------------'''
M, k1, d1, k2, d2 = matchCompute(img[1], img1)
src = np.float32([k1[m.queryIdx].pt for m in M]).reshape(-1,1,2)
dst = np.float32([k2[m.trainIdx].pt for m in M]).reshape(-1,1,2)
M, mask = cv.findHomography(src, dst, cv.RANSAC, 2*5.99**0.5)
print(M)
M = np.dot(np.matrix([[1,0,xo],[0,1,yo],[0,0,1]]), M)
matchesMask = mask.ravel().tolist()

img2 = cv.warpPerspective(img[1], M, (img[1].shape[1]*5, img[1].shape[0]*5))
for i in range(len(img1)):
    for j in range(len(img1[0])):
        if list(img2[i+xo][j+yo])==[0,0,0]:
            img2[i+xo][j+yo] = img1[i][j]
#        else:
#            img1[i+xo][j+yo] = np.uint8(img[1][i][j]/2 + img1[i+xo][j+yo]/2)
            
plt.figure(figsize=(15,15))
img2 = crop(np.array(img2, dtype='uint8'))
plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
plt.show()

'''-----------------------'''
M, k1, d1, k2, d2 = matchCompute(img[0], img2)
src = np.float32([k1[m.queryIdx].pt for m in M]).reshape(-1,1,2)
dst = np.float32([k2[m.trainIdx].pt for m in M]).reshape(-1,1,2)
M, mask = cv.findHomography(src, dst, cv.RANSAC, 2*5.99**0.5)
print(M)
M = np.dot(np.matrix([[1,0,xo*3],[0,1,yo],[0,0,1]]), M)
matchesMask = mask.ravel().tolist()
yo *= 3
img3 = cv.warpPerspective(img[0], M, (img[0].shape[1]*10, img[0].shape[0]*10))
for i in range(len(img2)):
    for j in range(len(img2[0])):
        if list(img3[i+xo][j+yo])==[0,0,0]:
            img3[i+xo][j+yo] = img2[i][j]
#        else:
#            img1[i+xo][j+yo] = np.uint8(img[1][i][j]/2 + img1[i+xo][j+yo]/2)
            
plt.figure(figsize=(15,15))
img3 = crop(np.array(img3, dtype='uint8'))
plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.show()