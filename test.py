import numpy as np
import matplotlib.pyplot as plt
import time
import icp

# Constants
N = 10                                    # number of random points in the dataset
num_tests = 5                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set

n0 = {"star":772, "3d" : 1138, "shape" : 693}
n1 = n0
modulo = 1

myplot = lambda x,y,col,s : plt.scatter(x,y,s=2, color=col, linewidths=2)

objet = "3d"

def txt_to_figure(directory,n,modulo,it):
    f = open(directory,"r")
    content = f.read()
    iters = content.split('_')
    if it is None:
        it = int( (len(iters)-1) // modulo )

    X,Y = np.zeros((it,n)), np.zeros((it,n))
    itera = 0

    for i in range(len(iters)):
        if i%modulo==0 and itera<it:
            dots = iters[i].split(";")[0:-1]
            for j in range(len(dots)):
                ev = eval(dots[j])
                X[itera][j]=ev[0]
                Y[itera][j]=ev[1]
            itera +=1
        f.close()

    return X,Y



def obtain_figures(dir_src="point_clouds/transfo_"+objet+".txt",dir_dst="point_clouds/"+objet+".txt",obj=objet,modulo=modulo):
    X,Y = txt_to_figure(dir_src,n1[obj],modulo=modulo,it=1)
    XC,YC = txt_to_figure(dir_dst,n0[obj],modulo=modulo,it=1)
    
    source = np.array([X[0],Y[0]]).T
    target = np.array([XC[0],YC[0]]).T

    return source, target

def plot(src, dst, k):
    plt.subplot(2,5,k+1)
    plt.axis("on")

    dst_X = np.array([ dst[i][0] for i in range(dst.shape[0]-1)])
    dst_Y = np.array([ dst[i][1] for i in range(dst.shape[0]-1)])

    src_X = np.array([ src[i][0] for i in range(src.shape[0]-1)])
    src_Y = np.array([ src[i][1] for i in range(src.shape[0]-1)])

    myplot( src_X , src_Y,"blue",s=1)
    myplot( dst_X , dst_Y,"red",s=1)
        
    miniX = min( int(min(src_X)) , int(min(dst_X)) )
    maxiX = max( int(max(src_X)) , int(max(dst_X)) )

    miniY = min( int(min(src_Y)) , int(min(dst_Y)) )
    maxiY = max( int(max(src_Y)) , int(max(dst_Y)) )

    plt.title("t = %.1f" %(k*modulo) )
    plt.xlim(miniX-10,maxiX+10)
    plt.ylim(miniY-10,maxiY+10)

def test_icp(A,B):
    total_time = 0
    m = A.shape[0]

    start = time.time()
    _,_ = icp.icp(A, B, max_iterations=20)
    total_time += time.time() - start

    print('icp time: {:.3}'.format(total_time/num_tests))

    return

def test_icp_ot(A,B):
    total_time = 0
    m,n = A.shape

    start = time.time()
    _, iterations = icp.icp_ot(A, B, max_iterations=20)
    total_time += time.time() - start

    print('icp time: {:.3}'.format(total_time/num_tests))

    return


if __name__ == "__main__":
    source, target = obtain_figures()
    test_icp_ot(source,target)