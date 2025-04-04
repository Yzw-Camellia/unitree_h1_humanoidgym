import os
import sys
import numpy as np
import itertools
import math
sys.path.append("/opt/openrobots/lib/python3.8/site-packages")

from pinocchio import visualize
import pinocchio
# import example_robot_data
import crocoddyl
from pinocchio.robot_wrapper import RobotWrapper
 
current_directory = os.getcwd()

 
# change path ??
# modelPath = current_directory + '/resources/robots/XBot/'
# URDF_FILENAME = "urdf/XBot-L.urdf"
modelPath=current_directory+'/resources/robots/h1/urdf'
URDF_FILENAME="/h1.urdf"
print("Model Path:", modelPath)
# modelPath=current_directory+'/resources/robots/jvrc/urdf'
# URDF_FILENAME="/jvrc.urdf"
 
# Load the full model
rrobot = RobotWrapper.BuildFromURDF(modelPath + URDF_FILENAME, [modelPath], pinocchio.JointModelFreeFlyer())  # Load URDF file
rmodel = rrobot.model

 
rightFoot = 'right_ankle_link'
leftFoot = 'left_ankle_link'
 
display = crocoddyl.MeshcatDisplay(
    rrobot
)
print(rrobot.model.nq)
q0 = pinocchio.utils.zero(rrobot.model.nq)
display.display([q0])
print("---------------initial pos-----------")
 
rdata = rmodel.createData()
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
 
rfId = rmodel.getFrameId(rightFoot)
lfId = rmodel.getFrameId(leftFoot)
print("rfId:", rfId)
print("lfId:", lfId)
rfFootPos0 = rdata.oMf[rfId].translation
lfFootPos0 = rdata.oMf[lfId].translation
 
comRef = pinocchio.centerOfMass(rmodel, rdata, q0)
 
 
print("--------------compute com--------------")
 
# initialAngle = np.array([0.3, 0.1, 0.3, -0.5, -0.2, 0.0,
#                          0.3, 0.1, 0.3, -0.5, -0.2, 0.0,
#                          0.0, 0.00, 0.0,
#                          0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
#                          0.0, 0.0, 0.0,
#                          0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
# q0 = pinocchio.utils.zero(rrobot.model.nq)
# q0[6] = 1  # q.w
# q0[2] =0.5848  # z
# q0[ 7:39] = initialAngle
# display.display([q0])
 
for i in range(rrobot.model.nq-7):
    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[i+7] = 1
    display.display([q0])
    print("-----------")
for i in range(rrobot.model.nq-7-6): #same time
    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[i+7] = 1
    q0[i+13] = 1
    display.display([q0])
    print("-----左右关节同时转动对比------")
    
 
print("--------------start to play ref pos--------------")
for i in range(5000):
    phase = i * 0.001
    sin_pos = np.sin(2 * np.pi * phase)
    sin_pos_l = sin_pos.copy()
    sin_pos_r = sin_pos.copy()
 
    ref_dof_pos = np.zeros((1,10))
    scale_1 = 0.35
    scale_2 = 2 * scale_1
    # left foot stance phase set to default joint pos
    if sin_pos_l > 0 :
        sin_pos_l = sin_pos_l * 0
    ref_dof_pos[:, 2] = sin_pos_l * scale_2 + (-0.1)
    ref_dof_pos[:, 3] = -sin_pos_l * scale_2 + (0.3)
    ref_dof_pos[:, 4] = sin_pos_l * scale_1 + (-0.2)
    # right foot stance phase set to default joint pos
    if sin_pos_r < 0:
        sin_pos_r = sin_pos_r * 0
    ref_dof_pos[:, 7] = -sin_pos_r * scale_2 + (-0.1)
    ref_dof_pos[:, 8] = sin_pos_r * scale_2 + (0.3)
    ref_dof_pos[:, 9] = -sin_pos_r * scale_1 + (-0.2)
    # Double support phase
    ref_dof_pos[np.abs(sin_pos) < 0.2] = 0
 
    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[7:rrobot.model.nq] = ref_dof_pos
    display.display([q0])

print("-------------finish-----------------")
 
 
 
 
 
for i in range(rrobot.model.nq-7):
    q0 = pinocchio.utils.zero(rrobot.model.nq)
    q0[6] = 1  # q.w
    q0[2] = 0  # z
    q0[i+7] = 1
    print("q0:", q0)
    display.display([q0])
    print("------------reset pos-----------------")