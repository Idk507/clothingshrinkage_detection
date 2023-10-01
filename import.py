import cv2
import numpy as np
from PIL import Image
import time

# Record the starting time
start_time = time.time()

path="Gokuldas/Gokuldas/MicrosoftTeams-image.png"
fl=path.split("/")[-1]
img=cv2.imread(path)
img_cpy=img.copy()
mask = cv2.inRange(img, (0,0,0), (140,255,255))
result = cv2.bitwise_and(img, img, mask=mask)
grayednow=cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
img = cv2.threshold(grayednow, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
ycoors=[];xcoors=[]
ycoor=[];xcoor=[]
templ1=cv2.imread("reclothshrinkagedetectionkindlyprovideexecutiontime/plusbw.png")
templ1=cv2.cvtColor(templ1, cv2.COLOR_BGR2GRAY)
w,h=templ1.shape[:2]
res=cv2.matchTemplate(img, templ1, cv2.TM_CCOEFF_NORMED)
th=0.4
loc=np.where(res>=th)
if len(loc[0]) > 135:
    th=0.6
    loc=np.where(res>=th)
    
#get unique coors of each
ycoor.extend(loc[0].tolist())
xcoor.extend(loc[1].tolist())
idx=[]
for i in range(len(ycoor)-1):
    for j in range(i+1, len(ycoor)):
        if i==j:
            continue
        else:
            if round(0.95*ycoor[i])<ycoor[j] and round(1.05*ycoor[i])>ycoor[j] and round(0.95*xcoor[i])<xcoor[j] and round(1.05*xcoor[i])>xcoor[j]:
                if j not in idx:
                    idx.append(j)
                    
for i in range(len(idx)):
        for j in range(len(idx)):
            if idx[j]>idx[i] and j!=i:
                idx[j]-=1
        ycoor.pop(idx[i])
        xcoor.pop(idx[i])

ycoor=np.asarray(ycoor)
xcoor=np.asarray(xcoor)
if len(ycoor)<4:
    for i in range(len(ycoor)):
        remy=np.delete(ycoor,i)
        remx=np.delete(xcoor,i)
        if not np.any((ycoor[i]-remy)<0.02*img.shape[0]):
            y4=ycoor[i]
        elif not np.any((xcoor[i]-remx)<0.02*img.shape[1]):
            x4=xcoor[i]
        else:
            continue
    ycoor=ycoor.tolist()
    xcoor=xcoor.tolist()
    #logic for last plus
    im1=Image.fromarray(img)
    for i in range(round(y4*0.975),round(y4*1.025)):
        for j in range(round(x4*0.975), round(x4*1.025)):
            pix1=im1.getpixel((j, i))
            if pix1 <100:
                break
    ycoor.append(i)
    xcoor.append(j)
    
templ1=cv2.imread("reclothshrinkagedetectionkindlyprovideexecutiontime/minusst.png")
templ1=cv2.cvtColor(templ1, cv2.COLOR_BGR2GRAY)
w,h=templ1.shape[:2]
res=cv2.matchTemplate(img, templ1, cv2.TM_CCOEFF_NORMED)
th=0.9
loc=np.where(res>=th)
#get unique coors of each
if type(xcoors)!=list:
    ycoor=ycoor.tolist()
    xcoor=xcoor.tolist()
ycoors.extend(loc[0].tolist())
xcoors.extend(loc[1].tolist())
idx=[]
for i in range(len(ycoors)-1):
    for j in range(i+1, len(ycoors)):
        if i==j:
            continue
        else:
            if round(0.95*ycoors[i])<ycoors[j] and round(1.05*ycoors[i])>ycoors[j] and round(0.95*xcoors[i])<xcoors[j] and round(1.05*xcoors[i])>xcoors[j]:
                if j not in idx:
                    idx.append(j)

for i in range(len(idx)):
        for j in range(len(idx)):
            if idx[j]>idx[i] and j!=i:
                idx[j]-=1
        ycoors.pop(idx[i])
        xcoors.pop(idx[i])

templ1=cv2.imread("reclothshrinkagedetectionkindlyprovideexecutiontime/minussl.png")
templ1=cv2.cvtColor(templ1, cv2.COLOR_BGR2GRAY)
w,h=templ1.shape[:2]
res=cv2.matchTemplate(img, templ1, cv2.TM_CCOEFF_NORMED)
th=0.9
loc=np.where(res>=th)
#get unique coors of each
ycoors.extend(loc[0].tolist())
xcoors.extend(loc[1].tolist())
idx=[]
for i in range(len(ycoors)-1):
    for j in range(i+1, len(ycoors)):
        if i==j:
            continue
        else:
            if round(0.95*ycoors[i])<ycoors[j] and round(1.05*ycoors[i])>ycoors[j] and round(0.95*xcoors[i])<xcoors[j] and round(1.05*xcoors[i])>xcoors[j]:
                if j not in idx:
                    idx.append(j)

for i in range(len(idx)):
        for j in range(len(idx)):
            if idx[j]>idx[i] and j!=i:
                idx[j]-=1
        ycoors.pop(idx[i])
        xcoors.pop(idx[i])

if type(ycoor)!=list:
    ycoor=ycoor.tolist()
    xcoor=xcoor.tolist()
ycoors.extend(ycoor)
xcoors.extend(xcoor)
ycoors=np.asarray(ycoors)
xcoors=np.asarray(xcoors)
loc=[];loc.append(ycoors);loc.append(xcoors)
loc=tuple(loc)

ymin=ycoors.min()
ymax=ycoors.max()
xmax=xcoors.max()
xmin=xcoors.min()

img=img[round(ymin):round(ymax*1.1),round(xmin*0.85):round(xmax*1.1)]
yb1=0; yb2=img.shape[0]; xb1=0; xb2=img.shape[1]
xdu=round(xb2/3); ydu=round(yb2/3)

def getparti(m, xb1, xb2, yb1, yb2, xdu, ydu):
    if m==0:
        im=img.crop((xb1, yb1, xb1+xdu, yb1+ydu))
        biasx=xb1
        biasy=yb1
    elif m==1:
        im=img.crop((xb1+xdu, yb1, xb1+2*xdu, yb1+ydu))
        biasx=xb1+xdu
        biasy=yb1
    elif m==2:
        im=img.crop((xb1+2*xdu, yb1, xb2, yb1+ydu))
        biasx=xb1+2*xdu
        biasy=yb1
    elif m==3:
        im=img.crop((xb1, yb1+ydu, xb1+xdu, yb1+2*ydu))
        biasx=xb1
        biasy=yb1+ydu
    elif m==4:
        im=img.crop((xb1+2*xdu, yb1+ydu, xb2, yb1+2*ydu))
        biasx=xb1+2*xdu
        biasy=yb1+ydu
    elif m==5:
        im=img.crop((xb1, yb1+2*ydu, xb1+xdu, yb2))
        biasx=xb1
        biasy=yb1+2*ydu
    elif m==6:
        im=img.crop((xb1+xdu, yb1+2*ydu, xb1+2*xdu, yb2))
        biasx=xb1+xdu
        biasy=yb1+2*ydu
    else:
        im=img.crop((xb1+2*xdu, yb1+2*ydu, xb2, yb2))
        biasx=xb1+2*xdu
        biasy=yb1+2*ydu
    return im,biasx,biasy

img=Image.fromarray(img)
pixels1=[]; total1=[]
pixels1.append(img.crop((xb1, yb1, xb1+xdu, yb1+ydu)).getdata())
pixels1.append(img.crop((xb1+xdu, yb1, xb1+2*xdu, yb1+ydu)).getdata())
pixels1.append(img.crop((xb1+2*xdu, yb1, xb2, yb1+ydu)).getdata())
pixels1.append(img.crop((xb1, yb1+ydu, xb1+xdu, yb1+2*ydu)).getdata())
pixels1.append(img.crop((xb1+2*xdu, yb1+ydu, xb2, yb1+2*ydu)).getdata())
pixels1.append(img.crop((xb1, yb1+2*ydu, xb1+xdu, yb2)).getdata())
pixels1.append(img.crop((xb1+xdu, yb1+2*ydu, xb1+2*xdu, yb2)).getdata())
pixels1.append(img.crop((xb1+2*xdu, yb1+2*ydu, xb2, yb2)).getdata())
for j in range(8):
    total1.append(len(list(filter(lambda i: i > 0, pixels1[j]))))
total2=sorted(total1)
idx=[]
for i in range(5):
    idx.append(total1.index(total2[i]))
print(idx)
# Record the ending time
end_time = time.time()

# Calculate the execution time
execution_time0 = end_time - start_time


print(f"Execution Time0: {execution_time0:.2f} seconds")   

start_time1 = time.time()    
xcoors=[];ycoors=[];partit=[0,1,2,3,4,5,6,7]
for i in range(len(idx)):
    im, biasx, biasy= getparti(idx[i], xb1, xb2, yb1, yb2, xdu, ydu)
    arr1=[]
    for x in range(im.width):
        for y in range(im.height):
            pixel = im.getpixel((x, y))
            xcoors=np.asarray(xcoors)
            ycoors=np.asarray(ycoors)
            if pixel > 200 and len(xcoors)==0:
                arr1.append((x,y))
            elif pixel > 200:
                arr1.append((x,y))
    xcoor1=0; ycoor1=0
    for i in range(len(arr1)):
        xcoor1+=arr1[i][0]
        ycoor1+=arr1[i][1]
    xcoor1/=len(arr1)
    ycoor1/=len(arr1)
    xcoor1=round(xcoor1+biasx)
    ycoor1=round(ycoor1+biasy)
    if type(xcoors) != list:
        xcoors=xcoors.tolist()
        ycoors=ycoors.tolist()
    xcoors.append(xcoor1)
    ycoors.append(ycoor1)
for i in range(len(partit)):
    if partit[i] not in idx:
        im, biasx, biasy= getparti(partit[i], xb1, xb2, yb1, yb2, xdu, ydu)
        arr1=[]
        for x in range(im.width):
            for y in range(im.height):
                pixel = im.getpixel((x, y))
                xcoors=np.asarray(xcoors)
                ycoors=np.asarray(ycoors)
                if pixel > 200 and np.any(abs(xcoors-x-biasx)<25) and np.any(abs(ycoors-y-biasy)<25):
                    print(x+biasx, y+biasy)
                    arr1.append((x,y))
        xcoor1=0; ycoor1=0
        for i in range(len(arr1)):
            xcoor1+=arr1[i][0]
            ycoor1+=arr1[i][1]
        if len(arr1)!=0:
            xcoor1/=len(arr1)
            ycoor1/=len(arr1)
        xcoor1=round(xcoor1+biasx)
        ycoor1=round(ycoor1+biasy)
        if type(xcoors) != list:
            xcoors=xcoors.tolist()
            ycoors=ycoors.tolist()
        xcoors.append(xcoor1)
        ycoors.append(ycoor1)
xcoors=np.asarray(xcoors)
ycoors=np.asarray(ycoors)
xcoors=xcoors+round(0.85*xmin)
ycoors=ycoors+round(ymin)
img=img_cpy

for i in range(len(xcoors)):
    cv2.circle(img, (xcoors[i],ycoors[i]), 10, (255,0,0), 2)
    
print(ymin,xmin)     
ymin=ycoors.min()
ymax=ycoors.max()
xmax=xcoors.max()
xmin=xcoors.min()

ptx2=xmax
pty2=ymin
ptx3=xmax
pty3=ymax 
ptx1=xmin
pty1=ymin
        
img=cv2.arrowedLine(img, (round(ptx2-0.06*img.shape[1]),pty2), (round(ptx3-0.06*img.shape[1]),pty3), (255,0,0), thickness=2, tipLength=0.03)
img=cv2.arrowedLine(img, (round(ptx3-0.06*img.shape[1]),pty3), (round(ptx2-0.06*img.shape[1]),pty2), (255,0,0), thickness=2, tipLength=0.03)   
img=cv2.arrowedLine(img, (ptx1,round(pty1-0.025*img.shape[0])), (ptx2,round(pty2-0.025*img.shape[0])), (255,0,0), thickness=2, tipLength=0.03)
img=cv2.arrowedLine(img, (ptx2,round(pty2-0.025*img.shape[0])), (ptx1,round(pty1-0.025*img.shape[0])), (255,0,0), thickness=2, tipLength=0.03)                                                                                               
img=cv2.putText(img, "a is a multiplying factor dependent on", (round(img.shape[1]*0.25), round(img.shape[0]*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2, cv2.LINE_AA)
img=cv2.putText(img, "camera focal length, etc.", (round(img.shape[1]*0.25), round(img.shape[0]*0.07)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2, cv2.LINE_AA)

#img=cv2.putText(img, f"distance={distx}a", (round((ptx1+ptx2)/3), round(pty1-0.04*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
#img=cv2.putText(img, f"distance={distx}a", (round(ptx2-0.4*img.shape[1]), round((pty2+pty3)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
# Record the ending time
end_time1 = time.time()

# Calculate the execution time
execution_time = end_time1 - start_time1
print(f"Execution Time: {execution_time:.2f} seconds")

starttime2 = time.time()
cv2.imshow('output', img)
cv2.waitKey(10)
cv2.destroyAllWindows()
end_time2 = time.time()

# Calculate the execution time
execution_time = end_time2 - starttime2
print(f"Execution Time: {execution_time:.2f} seconds")
print(f"Execution Time: {execution_time0:.2f} seconds")
cv2.imwrite("C:/Users/amita/Downloads/Gokuldas/Gokuldas/markedcloth/"+fl, img)