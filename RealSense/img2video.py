'''
Created on Sep 8, 2022

@author: ggutow
'''
import skvideo.io as skio
import cv2 as cv
from os import path
import glob
import sys
if __name__=="__main__":
    args=sys.argv
    folder=args[1]
    img_extension=args[2]
    output_name=args[3]
    imgfiles=sorted(glob.glob(path.expanduser(path.join(folder,"*."+img_extension))))
    output=path.join(folder,output_name)
    print(output)
    video=skio.FFmpegWriter(output,outputdict={'-r':'10'})#tuple(reversed(img.shape[:2])))
    for imgfile in imgfiles:
        img=cv.imread(imgfile)
        video.writeFrame(img[:,:,::-1])
    video.close()
