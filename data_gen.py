import cv2
import numpy as np
#import mat4py
import scipy.io as sio
import matplotlib.pyplot as plt
import L0_helpers
import os, sys

# L0 minimization parameters
kappa = 2.0;
_lambda = 9e-3;

train = sio.loadmat('32x32/train_32x32.mat')
x_train = np.array(train['X'])
#y_train = np.array(train['y'])
print(x_train.shape)
img = x_train[:,:,:,0]
N, M, D = np.int32(img.shape)

#cv2.imwrite("in0.png", img)

S = np.float32(img) / 256


# Compute image OTF
size_2D = [N, M]
fx = np.int32([[1, -1]])
fy = np.int32([[1], [-1]])
otfFx = L0_helpers.psf2otf(fx, size_2D)
otfFy = L0_helpers.psf2otf(fy, size_2D)
otfFx = L0_helpers.psf2otf(fx, size_2D)
otfFy = L0_helpers.psf2otf(fy, size_2D)


# Compute MTF
MTF = np.power(np.abs(otfFx), 2) + np.power(np.abs(otfFy), 2)
MTF = np.tile(MTF[:, :, np.newaxis], (1, 1, D))

# Initialize buffers
h = np.float32(np.zeros((N, M, D)))
v = np.float32(np.zeros((N, M, D)))
dxhp = np.float32(np.zeros((N, M, D)))
dyvp = np.float32(np.zeros((N, M, D)))
FS = np.complex64(np.zeros((N, M, D)))

# Iteration settings
beta_max = 1e5;
beta = 2 * _lambda
iteration = 0
Abefore = np.zeros((N, M, D , 200))
Aafter = np.zeros((N, M, D,200))
for i in range(50000):
    index = i
    S = np.float32(x_train[:,:,:,index]) / 256
    if i < 200:
        Abefore[:,:,:,i] = S
    # Compute F(I)
    FI = np.complex64(np.zeros((N, M, D)))
    FI[:,:,0] = np.fft.fft2(S[:,:,0])
    FI[:,:,1] = np.fft.fft2(S[:,:,1])
    FI[:,:,2] = np.fft.fft2(S[:,:,2])

    # Iterate until desired convergence in similarity
    while beta < beta_max:

      ### Step 1: estimate (h, v) subproblem
      # compute dxSp
      h[:,0:M-1,:] = np.diff(S, 1, 1)
      h[:,M-1:M,:] = S[:,0:1,:] - S[:,M-1:M,:]

      # compute dySp
      v[0:N-1,:,:] = np.diff(S, 1, 0)
      v[N-1:N,:,:] = S[0:1,:,:] - S[N-1:N,:,:]

      # compute minimum energy E = dxSp^2 + dySp^2 <= _lambda/beta
      t = np.sum(np.power(h, 2) + np.power(v, 2), axis=2) < _lambda / beta
      t = np.tile(t[:, :, np.newaxis], (1, 1, 3))

      # compute piecewise solution for hp, vp
      h[t], v[t] = 0,0

      ### Step 2: estimate S subproblem
      # compute dxhp + dyvp
      dxhp[:,0:1,:] = h[:,M-1:M,:] - h[:,0:1,:]
      dxhp[:,1:M,:] = -(np.diff(h, 1, 1))
      dyvp[0:1,:,:] = v[N-1:N,:,:] - v[0:1,:,:]
      dyvp[1:N,:,:] = -(np.diff(v, 1, 0))
      normin = dxhp + dyvp

      FS[:,:,0] = np.fft.fft2(normin[:,:,0])
      FS[:,:,1] = np.fft.fft2(normin[:,:,1])
      FS[:,:,2] = np.fft.fft2(normin[:,:,2])


      # solve for S + 1 in Fourier domain
      denorm = 1 + beta * MTF;
      FS[:,:,:] = (FI + beta * FS) / denorm

      # inverse FFT to compute S + 1
      S[:,:,0] = np.float32((np.fft.ifft2(FS[:,:,0])).real)
      S[:,:,1] = np.float32((np.fft.ifft2(FS[:,:,1])).real)
      S[:,:,2] = np.float32((np.fft.ifft2(FS[:,:,2])).real)

      beta *= kappa
      iteration += 1

    if i < 200:
        Aafter[:,:,:,i] = S
    # Rescale image
    S = S * 256
    cv2.imwrite("32x32/data_all/o%d.png"%index, S)


filename = os.path.join("32x32/", "compair.png")
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for j in range(num_images):
    i = j
    ax = plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.imshow(Abefore[:,:,:,i],  cmap='Greys')
    #plt.xlabel(error)
    #plt.xlabel('z = ['+", ".join(sar)+']')
    plt.xticks([])
    plt.yticks([])
    ax = plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plt.imshow(Aafter[:,:,:,i],  cmap='Greys')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(filename)
#plt.imshow(np.concatenate((img/255.0,S/255.0),axis=1))
plt.show()
"""
aaa = np.zeros((32,32,3))
bbb = np.zeros((32,32,3))
aaa[0:16,0:16,:] = x_train[16:32,16:32,:,1]
aaa[16:32,0:16,:] = x_train[16:32,0:16,:,1]
aaa[0:16,16:32,:] = x_train[0:16,0:16,:,1]
aaa[16:32,16:32,:] = x_train[0:16,16:32,:,1]
plt.imshow(aaa)
for i in range(4):
    for j in range(4):
        px = (3*i+1)%4
        py = (3*j+2)%4
        bbb[i*8+0:i*8+8,j*8+0:j*8+8,:] = x_train[px*8+0:px*8+8,py*8+0:py*8+8,:,30]
"""
#plt.imshow(bbb)
#plt.show()
