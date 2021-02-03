import matplotlib.pyplot as plt
import numpy as np


def merge_in_swap(arm_array,p_array):
    arm_array[:,:,14:27,:] = p_array[:,:,0:13,:]
    #arm_array=rearrange(arms)
    return arm_array



def visualize_coordinate_transformations_single_trial(tt,e,T_,i=0):
    nd = graph_trialized_robot_positions(tt[i,:,:,:,0],e[i,:,:],T_)
    tt_posture=merge_in_swap(tt[:,:,:,:,0],tt[:,:,:,:,1])
    kcd,rcd = split_data_into_coord_matching(tt_posture,e)
    fixit1 = plt.figure(figsize=(12,6))
    axel = fixit1.add_subplot(1,1,1)
    plt.plot(rcd[0,i,:],label='XPos Potentiometer')
    plt.plot(kcd[0,i,:],label='XPos DLC/Video')
    plt.plot(kcd[1,i,:],label='YPos DLC/Video')
    plt.plot(rcd[1,i,:],label='YPos Potentiometer')
    plt.plot(rcd[2,i,:],label='ZPos Potentiometer')
    plt.plot(kcd[2,i,:],label='ZPos DLC/Video')
    plt.xlabel('Frames')
    plt.ylabel('Pos (mm) ')
    plt.legend()
    plt.savefig('PosPlots_NoXFORM_avgh.png')
    plt.show()
    fixit2 = plt.figure(figsize=(12,6))
    axel1 = fixit2.add_subplot(2,1,1)
    plt.plot(rcd[0,i,:],label='XPos Potentiometer')
    plt.plot(nd[:,0],label='XPos DLC/Video XFormed')
    plt.plot(nd[:,1],label='YPos DLC/Video XFormed')
    plt.plot(rcd[1,i,:],label='YPos Potentiometer')
    plt.plot(rcd[2,i,:],label='ZPos Potentiometer')
    plt.plot(nd[:,2],label='ZPos DLC/Video XFormed')
    plt.xlabel('Frames')
    plt.ylabel('Pos (mm) ')
    plt.legend()
    plt.savefig('PosPlots_XForm_avgh.png')
    plt.show()
    return


def single_trial_xform_array(k_m, eit, cl=0):
    rv=np.dot(k_m.T,eit[0:3,0:3]) + cl # good Xform value, then use this value
    return rv


def graph_trialized_robot_positions(arms,e,T_):
    print(arms.shape)
    fixit1 = plt.figure(figsize=(10,6))
    axel = fixit1.add_subplot(1,1,1, projection='3d',label='Position of Robot during Trial')
    axel.scatter(arms[0,0,:],arms[1,0,:],arms[2,0,:],label='Handle')
    axel.scatter(e[7,:],e[8,:],e[9,:],label='Robot Position')
    nd=single_trial_xform_array(arms[:,0,:],T_)
    axel.scatter(nd[:,0],nd[:,1],nd[:,2],label='Handle Position Post-Transformation')
    plt.legend()
    plt.show()
    plt.savefig('3D_coordmatching_kinematic_inc_1st_avgh.png')
    return nd


def split_data_into_coord_matching(tka,exp):
    robot_coordinates = exp[:,7:10,:]
    kinematic_coordinates = np.average(tka[:,:,0:2,:],axis=2) # Back Handle + Handle Average
    #kinematic_coordinates=tka[:,:,1,:]
    print(kinematic_coordinates.shape)
    return kinematic_coordinates,robot_coordinates