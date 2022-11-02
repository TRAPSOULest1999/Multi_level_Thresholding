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
#import warnings
#warnings.filterwarnings("ignore")

# assigning nt = 50
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

def cost(neighbor, act, otsu):
    if mse(neighbor, otsu) < mse(act, otsu):
        return True
    return False

def simulated_annealing(initial_state, otsuTh,list):
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

            if cost(threshN, current_state, otsuTh) and cost(threshN, thresh, otsuTh):
                solution = threshN
                current_state = threshN
                retS = retN
            elif cost(thresh, current_state, otsuTh):
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
n = 3# number of thresholds (better choose even value)
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
retO, thresho = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

result = simulated_annealing(thresh1, thresho,T)
# Applying multi-Otsu threshold for the default value, generating
# three classes.

thresholds = threshold_multiotsu(image,classes=4)

# Using the threshold values, we generate the three regions.
regions = np.digitize(img, bins=thresholds)
output = img_as_ubyte(regions)



print(f'PSNR: {psnr(output, result)},\nSSIM: {ssim(output,result)}')
cv2.imshow('Solution', result)
cv2.imshow('Otsu', thresho)
cv2.imshow('Multi-otsu', output)

t1_stop = process_time()

print("Elapsed time:", t1_stop, t1_start)

print("Elapsed time during the whole program in milli-seconds:",
      ((t1_stop - t1_start)*1000))
#cv2.imshow('Thresh', thresh1)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()