import copy as cp
import numpy as np
import cv2

# standard deviation of noise
SD = 50

# size of multi-view images X_{0,1,2,3} in Eq. (1)
SIZE_MVI = [512, 512, 8, 8]

# size of root block X_{0,1,2,3}' in Eq. (7)
SIZE_RB = [32, 32, 8, 8]

# size of sub block X_{0,1,2,3}'' in Eq. (14) 
SIZE_SB = [8, 8, 1, 1]

# size of stride S_{0,1,2,3} in Eq. (7)
SIZE_STR = [8, 8, 1, 1]

def mviread(path):
    
    # buffer for multi-view images
    src = np.zeros(SIZE_MVI)

    # read
    for i2 in range(SIZE_MVI[2]):
        for i3 in range(SIZE_MVI[3]):
            inm = '{0}/sai_{1:02d}_{2:02d}.png'.format(path, i2+1, i3+1)
            img = cv2.imread(inm)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            src[:, :, i2, i3] = img

    return src

def mviwrite(path, lf):

    # write
    for i2 in range(SIZE_MVI[2]):
        for i3 in range(SIZE_MVI[3]):
            inm = '{0}/sai_{1:02d}_{2:02d}.png'.format(path, i2+1, i3+1)
            cv2.imwrite(inm, lf[:, :, i2, i3])

def mvipad(mvi):

    p0 = (SIZE_RB[0] >> 1, SIZE_RB[0] >> 1)
    p1 = (SIZE_RB[1] >> 1, SIZE_RB[1] >> 1)
    p2 = (0, 0)
    p3 = (0, 0)

    return np.pad(mvi, [p0, p1, p2, p3], 'symmetric')

def mviclip(mvi):

    s0 = slice(SIZE_RB[0] >> 1, SIZE_MVI[0] + (SIZE_RB[0] >> 1))
    s1 = slice(SIZE_RB[1] >> 1, SIZE_MVI[1] + (SIZE_RB[1] >> 1))
    s2 = slice(0, SIZE_MVI[2])
    s3 = slice(0, SIZE_MVI[3])

    return mvi[s0, s1, s2, s3]

def winget():

    w0 = np.zeros([SIZE_RB[0], 1, 1, 1]) # triangular
    w1 = np.zeros([1, SIZE_RB[1], 1, 1]) # triangular
    w2 = np.ones ([1, 1, SIZE_RB[2], 1]) # rectangular
    w3 = np.ones ([1, 1, 1, SIZE_RB[3]]) # rectangular

    w0[:, 0, 0, 0] = np.bartlett(SIZE_RB[0])
    w1[0, :, 0, 0] = np.bartlett(SIZE_RB[1])

    return w0 * w1 * w2 * w3

def thresh(cff, thr):

    dst = cff * np.maximum(1 - thr / (np.abs(cff) + 1e-12), 0)
    return dst

def SUREthresh(cff):

    # buffer for denoised main block
    denoised = np.zeros(SIZE_RB, np.complex)

    # number of elements in sub block
    num_elm = SIZE_SB[0] * SIZE_SB[1] * SIZE_SB[2] * SIZE_SB[3]

    # constant
    ls_fwd = np.linspace(num_elm - 1, 0, num_elm)
    ls_bwd = np.linspace(1, num_elm, num_elm)

    # begin SUREthresh
    r0 = range(0, SIZE_RB[0], SIZE_SB[0])
    r1 = range(0, SIZE_RB[1], SIZE_SB[1])
    r2 = range(0, SIZE_RB[2], SIZE_SB[2])
    r3 = range(0, SIZE_RB[3], SIZE_SB[3])

    for i3 in r3:
        s3 = slice(i3, i3 + SIZE_SB[3])
        for i2 in r2:
            s2 = slice(i2, i2 + SIZE_SB[2])
            for i1 in r1:
                s1 = slice(i1, i1 + SIZE_SB[1])
                for i0 in r0:
                    s0 = slice(i0, i0 + SIZE_SB[0])

                    # extract noisy sub block
                    blk = cff[s0, s1, s2, s3]

                    # compute risk in Eq. (22)
                    srt = np.sort(np.abs(blk.reshape(-1) ** 2))
                    cum = np.cumsum(srt)
                    rsk = cum + ls_fwd * srt
                    rsk = (num_elm - 2 * ls_bwd + rsk)

                    # minimizer of Eq. (19)
                    idx = np.argmin(rsk)
                    thr = np.sqrt(srt[idx])

                    # thresholding in Eq. (18)
                    blk = thresh(blk, thr)
                    denoised[s0, s1, s2, s3] = blk

    return denoised

def denoise(noisy):

    # normalize standard deviation
    noisy = noisy / SD

    # padding
    noisy = mvipad(noisy)

    # size of padded multi-view images
    size_mvi = noisy.shape

    # number of elements in root block
    num_elm = SIZE_RB[0] * SIZE_RB[1] * SIZE_RB[2] * SIZE_RB[3]

    # window function in Eq. (8)
    win = winget()

    # normalize window function
    win = win / np.linalg.norm(win) * np.sqrt(num_elm)

    # buffer for normalization factor a_w in Eq. (35)
    denominator = np.zeros(size_mvi)

    # buffer for denoised multi-view images
    denoised = np.zeros(size_mvi)

    # begin denoise
    r0 = range(0, size_mvi[0] - SIZE_RB[0] + 1, SIZE_STR[0])
    r1 = range(0, size_mvi[1] - SIZE_RB[1] + 1, SIZE_STR[1])
    r2 = range(0, size_mvi[2] - SIZE_RB[2] + 1, SIZE_STR[2])
    r3 = range(0, size_mvi[3] - SIZE_RB[3] + 1, SIZE_STR[3])

    for i3 in r3:
        s3 = slice(i3, i3 + SIZE_RB[3])
        for i2 in r2:
            s2 = slice(i2, i2 + SIZE_RB[2])
            for i1 in r1:
                s1 = slice(i1, i1 + SIZE_RB[1])
                for i0 in r0:
                    s0 = slice(i0, i0 + SIZE_RB[0])

                    # extract noisy main block
                    blk = noisy[s0, s1, s2, s3]

                    # short-time DFT in Eq. (8)
                    blk = blk * win
                    blk = np.fft.fftn(blk) / np.sqrt(num_elm)

                    # SURE thresholding in Eq. (18)
                    blk = SUREthresh(blk)

                    # inverse DFT in Eq. (34)
                    blk = np.real(np.fft.ifftn(blk)) * np.sqrt(num_elm)

                    # overlap-add synthesis in Eq. (35)
                    denoised[s0, s1, s2, s3] += blk
                    denominator[s0, s1, s2, s3] += win
    
    # normalization in Eq. (35)
    denoised = denoised / (denominator + 1E-16) * SD

    # clipping
    denoised = mviclip(denoised)

    return denoised

if __name__ == '__main__':

    # read clean multi-view images
    clean = mviread('mvi/clean/')
    #clean = clean[0:128, 0:128, 0:8, 0:8]
    #SIZE_MVI = clean.shape

    # noisy multi-view images
    noisy = np.array(clean, 'float') + np.random.normal(0, SD, SIZE_MVI)

    # denoised multi-view images
    denoised = denoise(noisy)

    # write noisy and denoised multi-view images
    mviwrite('mvi/noisy', noisy)
    mviwrite('mvi/denoised', denoised)
    print(cv2.PSNR(clean[:, :, 3, 3], denoised[:, :, 3, 3]))