#作業5-1
import numpy as np 
import pandas as pd
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
data={'國家':['台灣','美國','英國'],'人口':np.random.randint(100000,size=3),}
data1=pd.DataFrame(data)
a=data1.groupby(by='國家')['人口'].mean()
print(a)
print("人口最多的國家是 :{}".format(data1.loc[data1.人口.idxmax(),'國家']))

#作業5-2
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
target_url ='https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt'
response=requests.get(target_url)
data=response.text
split_tag = "\n"
data = data.split(split_tag)
data1=[]
for x in data:
    data1.append(x.split('\t'))
df=pd.DataFrame(data1)
response = requests.get(df.loc[0, 1])
img = Image.open(BytesIO(response.content))
img = np.array(img)
# print(img.shape)
plt.imshow(img)
plt.show()

def img2arr_fromURLs(url_list, resize = False):
    img_list = []
    for url in url_list:
        response = requests.get(url)
        try:
            img = Image.open(BytesIO(response.content))
            if resize:
                img = img.resize((256,256))
            img = np.array(img)
            img_list.append(img)
        except:
            pass
    
    return img_list

result = img2arr_fromURLs(df[0:5][1].values)
print("Total images that we got: %i " % len(result))

for im_get in result:
    plt.imshow(im_get)
    plt.show()




import json
import pickle
data = []
with open("example.txt", 'r') as f:
    for line in f:
        line = line.replace('\n', '').split(',')
        data.append(line)
df = pd.DataFrame(data[1:])
df.columns = data[0]

df.to_json('example01.json')
with open('example01.json', 'r') as f:
    j1 = json.load(f)
print(j1)
df.set_index('id',inplace=True)
df.to_json('example02.json', orient='index')
with open('example02.json', 'r') as f:
    j2 = json.load(f)
print(j2)

array = np.array(data[1:])
print(array)
np.save('example.npy',array)
a=np.load('example.npy')
print(a)

with open('example.pkl', 'wb') as f:
    pickle.dump(file=f, obj=data)

with open('example.pkl', 'rb') as f:
    pkl_data = pickle.load(f)
print(pkl_data)

#作業5-3
import matplotlib.pyplot as plt
import skimage.io as skio
from PIL import Image
import cv2
import scipy.io as sio

img1 = skio.imread('example.jpg')
plt.imshow(img1)
plt.show()

img2 = Image.open('example.jpg')
img2 = np.array(img2)
plt.imshow(img2)
plt.show()

img3 = cv2.imread('example.jpg')
plt.imshow(img3)
plt.show()

img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
plt.imshow(img4)
plt.show()

sio.savemat(file_name='example.mat', mdict={'img': img1})
mat_arr = sio.loadmat('example.mat')
print(mat_arr.keys())
mat_arr = mat_arr['img']
print(mat_arr.shape)
plt.imshow(mat_arr)
plt.show()

