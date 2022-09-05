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

import cv2 as cv

from model.model import VMStiefelSVDNet
from utils.utils import convert_predictions_VMStSVD

def estimation_to_camera_transformation():
    #frame used for estimation is x pointing out, y pointing right, z pointing down
    #but cv.projectPoints wants a camera frame with x pointing right, y pointing down, z pointing out
    #R is thus the matrix that maps points in the estimation frame to the opencv camera frame
    #note that the two frames have the same origin so 0 translation component is needed
    t=np.zeros((3,1))
    R=np.array([[0.0, 1, 0],
       [0, 0, 1],
       [1, 0, 0]])
    return R,t

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

def get_depth_intrinsics(profile=None):
    if profile is None:
        pipeline=rs.pipeline()
        config=rs.config()
        wrapper=rs.pipeline_wrapper(pipeline)
        profile=config.resolve(wrapper)
    streams=profile.get_streams()
    for stream in streams:
        if stream.stream_type()==rs.stream.depth:
            return stream.as_video_stream_profile().get_intrinsics()
    print("No depth stream found")

def get_color_intrinsics(profile=None):
    if profile is None:
        pipeline=rs.pipeline()
        config=rs.config()
        wrapper=rs.pipeline_wrapper(pipeline)
        profile=config.resolve(wrapper)
    streams=profile.get_streams()
    for stream in streams:
        if stream.stream_type()==rs.stream.color:
            return stream.as_video_stream_profile().get_intrinsics()
    print("No color stream found")    

def remove_background(cimages,dimages,detectShadows=False):
    '''
    DUST-net expects everything except the two bodies of interest to have 0 depth
    
    @param cimages: n element list of width,height,3 (RGB) uint16 color images
    @param dimages: n element list of width,height uint16 depth images
    @param detectShadows: should the opencv background subtractor do shadow detection (default False)
    @return masked cimages, masked dimages using foreground mask obtained from cimages via opencv BackgroundSubtractorMOG2
    '''
    backSub=cv.createBackgroundSubtractorMOG2(detectShadows=detectShadows)
    masks=np.array([backSub.apply(ci) for ci in cimages])//255
    return (masks*cimages.transpose((3,0,1,2))).transpose((1,2,3,0)),masks*dimages
        

def process_frames(dimages,start=38):
    #turn the nimg x height x width depth images into a batches (here, only 1) x 16 images x 3 channels (repeats depth) x height x width
    depth=torch.Tensor(np.tile(np.expand_dims(dimages[start:start+16].astype(np.float32),(0,2)),(1,1,3,1,1)))  
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

def streaming_estimation(stabilization_frames=5):
    #setup network
    #load trained model onto GPU
    nimg=16
    dustnet=VMStiefelSVDNet(hidden_size=[1024],img_seq_len=nimg)
    checkpoint = torch.load("data/trained_wts/partnet_vmstsvd.pt")
    dustnet.load_state_dict(checkpoint["model_state_dict"])
    device=torch.device(0)
    dustnet.float().to(device)
    dustnet.eval()
    
    #launch camera
    pipeline=rs.pipeline()
    try:
        profile=pipeline.start()
        color_intrinsics=get_color_intrinsics(profile)
        depth_intrinsics=get_depth_intrinsics(profile)
        depth_scale=get_depth_scale(profile)
        frame=pipeline.wait_for_frames()
        cv.imshow("Axis Estimate",np.array(frame.get_color_frame().get_data())[...,::-1])
        #img=ax.imshow(frame.get_color_frame().get_data())
        for _ in range(stabilization_frames):
            frames=pipeline.wait_for_frames()
            cv.imshow("Axis Estimate",np.array(frame.get_color_frame().get_data())[...,::-1])
            cv.waitKey(10)
            #img.set_data(frames.get_color_frame().get_data())
            #ax.figure.canvas.draw()
        #initialize frame buffer
        dimages=np.zeros((1,nimg,3,depth_intrinsics.height,depth_intrinsics.width),dtype=np.float32)
        while True:
            #get 16 frames
            i=0
            while i<nimg:
                frames=pipeline.wait_for_frames()
                depthf=frames.get_depth_frame()
                if not depthf:
                    continue
                dimages[0,i]=depth_scale*np.array(depthf.get_data())
                i+=1
            #run estimation on frames
            with torch.no_grad():
                prediction=dustnet(torch.Tensor(dimages).to(device))
                pred,cov=convert_predictions_VMStSVD(prediction,nimg-1)
                pred=pred[0,-1].cpu()
                cov=cov[0,-1].cpu()
            #update display
            colorf=frames.get_color_frame()   
            cimg=np.array(colorf.get_data())
            try:        
                color_with_axis=add_axis_to_image(cimg, pred, color_intrinsics, 2)
            except ValueError:
                #unable to slide axis into field of view
                print("Axis does not intersect field of view")
                color_with_axis=cimg
            cv.imshow("Axis Estimate",color_with_axis[...,::-1])
            cv.waitKey(10)
            #img.set_data(color_with_axis)
            #ax.figure.canvas.draw()
    finally:
        pipeline.stop()

def plot_axis_estimates(prediction,ax3d,clear=False):
    '''
    plot in 3D the lines estimated to be the axis at each timestep
    
    DUST-net appears to have been trained to report in a camera frame 
    where the x axis points forward, the y axis points right, and the z axis points down
    that is:
    x points towards increasing depth
    y points towards increasing width index in pixel space
    z points towards increasing height index in pixel space
    '''
    l=np.array(prediction[:,:3])
    m=np.array(prediction[:,3:6])
    p=np.cross(l,m)
    if clear:
        ax3d.clear()
    allpts=np.concatenate((p,l+p))
    mins=np.min(allpts,0)
    maxs=np.max(allpts,0)
    nrange=np.max(maxs-mins)
    ax3d.quiver(p[:,0],p[:,1],p[:,2],l[:,0],l[:,1],l[:,2])
    ax3d.set_xlim(mins[0],mins[0]+nrange)
    ax3d.set_ylim(mins[1],mins[1]+nrange)
    ax3d.set_zlim(mins[2],mins[2]+nrange)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

def axis_in_depth_image_coord(label,intrinsics):
    l=np.array(label[...,:3])
    m=np.array(label[...,3:6])
    p=np.cross(l,m)
    ends=l+p
    return to_image_coordinates(p, intrinsics),to_image_coordinates(ends, intrinsics)

def to_image_coordinates(points,intrinsics):
    R,t=estimation_to_camera_transformation()
    #camera intrinsics are stored using the pyrealsense2 class, but we just need the camera intrinsics matrix:
    intrinsic_matr=np.array([[intrinsics.fx,0,intrinsics.ppx],[0,intrinsics.fy,intrinsics.ppy],[0,0,1]])
    #and the distortion coefficients, which the pyrealsense2 sets to 0 for the D435i depth and color streams
    distortion_coeffs=np.array(intrinsics.coeffs)
    rot_rodrigues=cv.Rodrigues(R)[0]
    transformed=cv.projectPoints(points,rot_rodrigues,t,intrinsic_matr,distortion_coeffs)[0]
    return tuple(transformed.squeeze().astype(np.int64))

def add_axis_to_image(image,label,intrinsics,width=2):
    tail,head=slide_axis_into_view(label, intrinsics)
    return add_arrow_to_image(image, tail, head, width)

def add_arrow_to_image(image,tail,head,width):
    if len(image.shape)>2:
        #color image
        linecolor=(255,255,0)
    else:
        #depth image
        linecolor=(255,)
    lined=cv.arrowedLine(image.copy(),tail,head,linecolor,width,line_type=cv.FILLED)
    return lined

def slide_axis_into_view(label,intrinsics):
    '''
    find a pair of points on the axis that are within the camera's field of view
    '''
    l=np.array(label[:3])
    m=np.array(label[3:6])
    p=np.cross(l,m)
    
    R,t=estimation_to_camera_transformation()
    fx=intrinsics.fx
    fy=intrinsics.fy
    cx=intrinsics.ppx
    cy=intrinsics.ppy
    width=intrinsics.width
    height=intrinsics.height
    #u,v are pixel horizontal, vertical locations respectively
    #we seek constraints on lambda such that lambda*l+p is in the FOV
    upper_bounds=[]
    lower_bounds=[]
    #u must be positive
    uposcoeff=fx*R[0]+cx*R[2]
    uposbound=-np.dot(uposcoeff,p)/np.dot(uposcoeff,l)
    if np.dot(uposcoeff,l)<0:#if negative, lambda must be LESS than the bound.
        upper_bounds.append(uposbound)
    else:
        lower_bounds.append(uposbound)
    
    #u must be less than the width of the FOV in pixels
    uwidcoeff=fx*R[0]+(cx-width)*R[2]
    uwidbound=-np.dot(uwidcoeff,p)/np.dot(uwidcoeff,l)
    if np.dot(uwidcoeff,l)>=0:#if negative, lambda must be GREATER than the bound.
        upper_bounds.append(uwidbound)
    else:
        lower_bounds.append(uwidbound)
    #v must be positive
    vposcoeff=fy*R[1]+cy*R[2]
    vposbound=-np.dot(vposcoeff,p)/np.dot(vposcoeff,l)
    if np.dot(vposcoeff,l)<0:#if negative, lambda must be LESS than the bound.
        upper_bounds.append(vposbound)
    else:
        lower_bounds.append(vposbound)    
    #u must be less than the width of the FOV in pixels
    vheicoeff=fy*R[1]+(cy-height)*R[2]
    vheibound=-np.dot(vheicoeff,p)/np.dot(vheicoeff,l)
    if np.dot(vheicoeff,l)>=0:#if negative, lambda must be GREATER than the bound.
        upper_bounds.append(vheibound)
    else:
        lower_bounds.append(vheibound)    
    if len(lower_bounds)==0:
        greatest_lower=-np.inf
    else:
        greatest_lower=np.max(lower_bounds)
    if len(upper_bounds)==0:
        least_upper=np.inf
    else:
        least_upper=np.min(upper_bounds)
    if least_upper<greatest_lower:
        raise ValueError("To be in FOV lambda must be both <{u} and >{l}".format(u=least_upper,l=greatest_lower))
    else:
        if 0>greatest_lower and 0<least_upper:
            #if possible use the location the network picked
            l1=0
            pslid=p
        else:
            #if not, translate as little as possible
            #note that in this case, both bounds have the same sign!
            l1=min((abs(greatest_lower),abs(least_upper)))*np.sign(greatest_lower)
            pslid=p+l1*l
        #then, scale l such that pslid+lscaled is still in the FOV
        #3,0,1:l2>-1,l2<2
        #first, try to not scale at all:
        if l1+1<least_upper and l1+1>greatest_lower:
                lscaled=l
        else:
            #would run off the edge in at least one direction
            positive_scaling=least_upper-l1
            negative_scaling=greatest_lower-l1
            if np.abs(positive_scaling)>np.abs(negative_scaling):
                scale=min((abs(positive_scaling),1))*np.sign(positive_scaling)
            else:
                scale=min((abs(negative_scaling),1))*np.sign(negative_scaling)
        if scale<0:
            #swap ends of arrow to keep the sign the same
            pslid=pslid+scale*l
            scale*=-1
        lscaled=scale*l
        return to_image_coordinates(pslid, intrinsics),to_image_coordinates(pslid+lscaled, intrinsics)
    
    
    
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
        
    
