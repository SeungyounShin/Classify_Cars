import os

path = '/Users/seungyoun/Desktop/CNN_Classification/data/train'
lst = os.listdir(path)

def cnt(dir_):
    try :
        r = len(os.listdir(dir_))
    except :
        r = -1
    return r

for i in lst:
    tmp = path+'/'+i
    num = cnt(tmp)
    if(num > 0):
        print(i,' : ',num)
