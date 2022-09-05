'''
Created on Sep 2, 2022

@author: ggutow
'''
'''
Created on Sep 1, 2022

@author: ggutow
'''
import pyrealsense2 as rs
import numpy as np
import time
import torch
from matplotlib import pyplot

from cv2 import createBackgroundSubtractorMOG2

from model.model import VMStiefelSVDNet
from utils.utils import convert_predictions_VMStSVD

def record_frames(dur):
    pipeline=rs.pipeline()
    align=rs.align(rs.stream.color)
    profile=pipeline.start()
    start=time.perf_counter()
    dimages=[]
    cimages=[]
    try:
        while time.perf_counter()-start<dur:
            frames=pipeline.wait_for_frames()
            aligned_frames=align.process(frames)
            depthf=aligned_frames.get_depth_frame()
            colorf=aligned_frames.get_color_frame()
            if not depthf or not colorf:
                continue
            dimages.append(np.array(depthf.get_data()))
            cimages.append(np.array(colorf.get_data()))
    finally:
        pipeline.stop()
        print(len(dimages))
    return np.array(cimages),np.array(dimages),profile

def get_depth_scale(profile=None):
    if profile is None:
        pipeline=rs.pipeline()
        config=rs.config()
        wrapper=rs.pipeline_wrapper(pipeline)
        profile=config.resolve(wrapper)
    return profile.get_device().first_depth_sensor().get_depth_scale()

def remove_background(cimages,dimages,detectShadows=False):
    '''
    DUST-net expects everything except the two bodies of interest to have 0 depth
    
    @param cimages: n element list of width,height,3 (RGB) uint16 color images
    @param dimages: n element list of width,height uint16 depth images
    @param detectShadows: should the opencv background subtractor do shadow detection (default False)
    @return masked cimages, masked dimages using foreground mask obtained from cimages via opencv BackgroundSubtractorMOG2
    '''
    backSub=createBackgroundSubtractorMOG2(detectShadows=detectShadows)
    masks=np.array([backSub.apply(ci) for ci in cimages])//255
    return (masks*cimages.transpose((3,0,1,2))).transpose((1,2,3,0)),masks*dimages
        

def process_frames(dimages,start=38):
    #turn the nimg x height x width depth images into a batches (here, only 1) x 16 images x 3 channels (repeats depth) x height x width
    depth=torch.Tensor(np.tile(np.expand_dims(dimages[start:start+16].astype(np.int16),(0,2)),(1,1,3,1,1)))  
    nimg=depth.shape[1]
      
    #load trained model onto GPU
    dustnet=VMStiefelSVDNet(hidden_size=[1024],img_seq_len=nimg)
    checkpoint = torch.load("data/trained_wts/partnet_vmstsvd.pt")
    dustnet.load_state_dict(checkpoint["model_state_dict"])
    device=torch.device(0)
    dustnet.float().to(device)
    dustnet.eval()
    
    #evaluate network on images and postprocess into a prediction and uncertainty
    with torch.no_grad():
        depthcuda=depth.to(device)
        prediction=dustnet(depthcuda)
        pred,cov=convert_predictions_VMStSVD(prediction,nimg-1)
    return pred[0].cpu(),cov[0].cpu()
def collect_and_process_realsense(duration):
    cimages,dimages,profile=record_frames(duration)
    pred,cov=process_frames(dimages)
    return cimages,dimages,pred,cov

def plot_axis_estimates(prediction,ax3d):
    '''
    plot in 3D the lines estimated to be the axis at each timestep
    '''
    l=np.array(prediction[:,:3])
    m=np.array(prediction[:,3:6])
    p=np.cross(l,m)
    ax3d.clear()
    ax3d.quiver(p[:,0],p[:,1],p[:,2],l[:,0],l[:,1],l[:,2])
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")


class IndexTracker:
    '''
    Call with an axis and a stack of images
    
    Then, select the figure window and press use arrow keys to choose a frame
    '''
    def __init__(self,image_stack,fig=None,ax=None):
        if ax is not None:
            self.ax=ax
        else:
            if fig is None:
                fig=pyplot.figure()
            self.ax=fig.gca()
        self.image_stack=image_stack
        self.index=0
        self.img=self.ax.imshow(self.image_stack[self.index])
        self.ax.figure.canvas.mpl_connect('key_press_event',self.on_key)
        
    def on_key(self,event):
        key=event.key
        if key=="left":
                self.index-=1
                if self.index<0:
                    self.index=0
        elif key=="right":
            self.index+=1
            if self.index>=len(self.image_stack):
                self.index=len(self.image_stack)-1
        self.update()
    def update(self):
        self.img.set_data(self.image_stack[self.index])
        self.ax.figure.canvas.draw()
    def replace_stack(self,new_images):
        self.image_stack=new_images
        self.update()
        
    
