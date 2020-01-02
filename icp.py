import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.optimize
import lap

def best_fit_transform(A, B):
    n,m = A.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    std,s2 = 0,0
    for i in range(n):
        std += np.linalg.norm(A[i]-centroid_A)**2
    for i in range(m):
        s2 += abs(S[i])

    scale = s2/(std)

    c1 = centroid_A.T
    c2 = centroid_B.T

    return R, c1, c2, scale

def nearest_neighbor(src, dst):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

  
def icp(A, B, max_iterations=10, plot=True,turn=1):
    if max_iterations>10:  
        turn = max_iterations/10
    myplot = lambda x,y,col,s : plt.scatter(x,y,s=2, color=col, linewidths=2)
    n,m = A.shape

    src = np.copy(A.T)
    dst = np.copy(B.T)

    dst_X = np.array([ dst[0][i] for i in range(dst.shape[1])])
    dst_Y = np.array([ dst[1][i] for i in range(dst.shape[1])])

    plt.figure(figsize =(12,5))
    k=0
    
    prev_dist = 100

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:,:].T, dst[:,:].T)
        dist = np.mean(distances)
        if abs(prev_dist-dist) < 0.1:
            print("break",i)
            break

        if plot==True and i%turn == 0:
            plt.subplot(2,5,k+1)
            k+=1
            plt.axis("on")

            src_X = np.array([ src[0][i] for i in range(src.shape[1])])
            src_Y = np.array([ src[1][i] for i in range(src.shape[1])])

            myplot( dst_X , dst_Y,"red",s=1)
            myplot( src_X , src_Y,"blue",s=1)
        
            miniX = min( int(min(src_X)) , int(min(dst_X)) )
            maxiX = max( int(max(src_X)) , int(max(dst_X)) )

            miniY = min( int(min(src_Y)) , int(min(dst_Y)) )
            maxiY = max( int(max(src_Y)) , int(max(dst_Y)) )

            plt.title("t = %.1f" %((k)*turn) )
            plt.xlim(miniX-10,maxiX+10)
            plt.ylim(miniY-10,maxiY+10)

        R,c1,c2,scale = best_fit_transform(src[:,:].T, dst[:,indices].T)
        print(scale)
        for i in range(n):
            vec = (src.T)[i]
            vec = scale * (R @ (vec-c1)) + c2
            src[0][i],src[1][i] = vec[0],vec[1]


        prev_dist=dist
    
    plt.show()
    R,c1,c2,scale = best_fit_transform(A, src[:,:].T)

    return (R,c1,c2,scale), i



def icp_ot(A, B, max_iterations=10, plot=True,turn=1):
    if max_iterations>10:  
        turn = max_iterations/10
    myplot = lambda x,y,col,s : plt.scatter(x,y,s=2, color=col, linewidths=2)
    n,m = A.shape

    src = np.copy(A.T)
    dst = np.copy(B.T)

    dst_X = np.array([ dst[0][i] for i in range(dst.shape[1])])
    dst_Y = np.array([ dst[1][i] for i in range(dst.shape[1])])

    plt.figure(figsize =(12,5))
    k=0
    
    prev_dist = 99999
    old = 99999

    for i in range(max_iterations):
        C = scipy.spatial.distance.cdist(src[:m,:].T, dst[:m,:].T)
        distances,_, indices = lap.lapjv(C)
        dist = np.mean(distances)


        if abs(prev_dist-dist) < 0.1:
            print("break",i)
            break
        if abs(prev_dist-dist)==old:
            print("quasi-convergence",i)
            break

        old = prev_dist - dist


        if plot==True and i%turn == 0:
            plt.subplot(2,5,k+1)
            plt.axis("on")

            src_X = np.array([ src[0][i] for i in range(src.shape[1])])
            src_Y = np.array([ src[1][i] for i in range(src.shape[1])])

            myplot( dst_X , dst_Y,"red",s=1)
            myplot( src_X , src_Y,"blue",s=1)
        
            miniX = min( int(min(src_X)) , int(min(dst_X)) )
            maxiX = max( int(max(src_X)) , int(max(dst_X)) )

            miniY = min( int(min(src_Y)) , int(min(dst_Y)) )
            maxiY = max( int(max(src_Y)) , int(max(dst_Y)) )

            plt.title("t = %.1f" %(k*turn) )
            plt.xlim(miniX-10,maxiX+10)
            plt.ylim(miniY-10,maxiY+10)
            k+=1


        R,c1,c2,scale = best_fit_transform(src[:,:].T, dst[:,indices].T)
        for i in range(n):
            vec = (src.T)[i]
            vec = scale * (R @ (vec-c1)) + c2
            src[0][i],src[1][i] = vec[0],vec[1]


        prev_dist=dist
    
    plt.show()
    R,c1,c2,scale = best_fit_transform(A, src[:,:].T)

    return (R,c1,c2,scale), i