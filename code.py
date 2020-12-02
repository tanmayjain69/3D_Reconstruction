import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

path1 =r"C:\Users\tanma\OneDrive\Desktop\imLeft.png"
path2 =r"C:\Users\tanma\OneDrive\Desktop\imRight.png"
imgL = cv2.imread(path1,cv2.IMREAD_GRAYSCALE) 
imgR = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)  

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()

#sift
#sift = cv2.xfeatures2d.SIFT_create()

#keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
#keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
#sift

#bf = cv2.BFMatcher()
#outputMatches = bf.match(descriptors_1,descriptors_2)
#points2f_imL	=	cv2.KeyPoint_convert(keypoints_1)
#points2f_imR	=	cv2.KeyPoint_convert(keypoints_2)
#inliers = len(points2f_imL)
#funda = cv2.findFundamentalMat(points2f_imL,points2f_imR,inliers,1.0,0.98)
