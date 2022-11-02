import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import math
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_ubyte, io
from skimage.filters import threshold_multiotsu
from time import process_time
import warnings
warnings.filterwarnings("ignore")

# assigning n = 50
nt = 50

# Start the stopwatch / counter
t1_start = process_time()

for i in range(nt):
    print(i, end=' ')

print()

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def segment(img):
    '''
    Maximum entropy segmentation
    :param img:
    :return:
    '''
    def calculate_current_entropy(hist, threshold):
        data_hist = hist.copy()
        background_sum = 0.
        target_sum = 0.
        for i in range(256):
            if i < threshold:  # Cumulative background
                background_sum += data_hist[i]
            else:  # Cumulative target
                target_sum += data_hist[i]
        background_ent = 0.
        target_ent = 0.
        for i in range(256):
            if i < threshold:  # entropy calculation BACKGROUND
                if data_hist[i] == 0:
                    continue
                ratio1 = data_hist[i] / background_sum
                background_ent -= ratio1 * np.log2(ratio1)
            else:
                if data_hist[i] == 0:
                    continue
                ratio2 = data_hist[i] / target_sum
                target_ent -= ratio2 * np.log2(ratio2)
        return target_ent + background_ent

    def max_entropy_segmentation(img):
        channels = [0]
        hist_size = [256]
        prange = [0, 256]
        hist = cv2.calcHist(img, channels, None, hist_size, prange)
        hist = np.reshape(hist, [-1])
        max_ent = 0.
        max_index = 0
        for i in range(256):
            cur_ent = calculate_current_entropy(hist, i)
            if cur_ent > max_ent:
                max_ent = cur_ent
                max_index = i
        ret, th = cv2.threshold(img, max_index, 255, cv2.THRESH_BINARY)
        return th
    img = max_entropy_segmentation(img)
    return img

def cost(neighbor, act, entropy):
    if mse(neighbor, entropy) < mse(act, entropy):
        return True
    return False

def simulated_annealing(initial_state, entropyTh,list):
    current_state = initial_state
    solution = current_state

    initial_temp = 41
    current_temp = initial_temp
    final_temp = 10
    alpha = 1

    while current_temp > final_temp:
        for i in range(len(list)):
            # Neighbor threshold
            x = random.choice(range(0, 256))
            retN, threshN = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)

            ret, thresh = cv2.threshold(img, list[i], 255, cv2.THRESH_BINARY)
            print(list[i])

            if cost(threshN, current_state, entropyTh) and cost(threshN, thresh, entropyTh):
                solution = threshN
                current_state = threshN
                retS = retN
            elif cost(thresh, current_state, entropyTh):
                solution = thresh
                current_state
                retS = ret
        print(retS)
        current_temp -= alpha
    return solution

img = cv2.imread('Assng2_Images/Street2.tif',0)
image = io.imread('Assng2_Images/Street2.tif',0)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
a = 0
b = 255 #64.52923338185347, 96.93386923901393, 108.08251633986929, 115.62578444747612, 125.07092124503865, 197.71053693984658
n = 5 # number of thresholds
k = 0.7 # free variable to take any positive value
T = [] # list which will contain 'n' thresholds

def multiThresh(img, a, b):
    if a>b:
        s=-1
        m=-1
        return m,s

    img = np.array(img)
    t1 = (img>=a)
    t2 = (img<=b)
    X = np.multiply(t1,t2)
    Y = np.multiply(img,X)
    s = np.sum(X)
    m = np.sum(Y)/s
    return m,s

for i in range(int(math.ceil(n/2-1))):
    img = np.array(img)
    t1 = (img>=a)
    t2 = (img<=b)
    X = np.multiply(t1,t2)
    Y = np.multiply(img,X)
    mu = np.sum(Y)/np.sum(X)

    Z = Y - mu
    Z = np.multiply(Z,X)
    W = np.multiply(Z,Z)
    sigma = math.sqrt(np.sum(W)/np.sum(X))

    T1 = mu - k*sigma
    T2 = mu + k*sigma

    x, y = multiThresh(img, a, T1)
    w, z = multiThresh(img, T2, b)

    T.append(x)
    T.append(w)

    a = T1+1
    b = T2-1
    k = k*(i+1)

T1 = mu
T2 = mu+1
x, y = multiThresh(img, a, T1)
w, z = multiThresh(img, T2, b)
T.append(x)
T.append(w)
T.sort()


ret,thresh1 = cv2.threshold(img,1,255,cv2.THRESH_BINARY)

#otsu method
threshe = segment(img)

result = simulated_annealing(thresh1, threshe,T)

# Applying multi-Otsu threshold for the default value, generating
#n classes
thresholds = threshold_multiotsu(image,classes=6)

# Using the threshold values, we generate the three regions.
regions = np.digitize(img, bins=thresholds)
output = img_as_ubyte(regions)



print(f'PSNR: {psnr(output, result)},\nSSIM: {ssim(output,result)}')
cv2.imshow('Solution', result)
cv2.imshow('Entropy', threshe)
cv2.imshow('Multi-otsu', output)

t1_stop = process_time()

#print("Elapsed time:", t1_stop, t1_start)

print("Elapsed time during the whole program in milli-seconds:",
      ((t1_stop - t1_start)*1000))
#cv2.imshow('Thresh', thresh1)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()