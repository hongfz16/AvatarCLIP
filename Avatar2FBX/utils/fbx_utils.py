from math import degrees
import sys
sys.path.append('/path/to/fbxsdk/build/lib/Python37_x64')

from FbxCommon import *
from fbx import *
import numpy as np
from scipy.spatial.transform import Rotation as R

JointsSize = 100.0

Child2Father = {
    1:  0,
    2:  0,
    3:  0,
    4:  1,
    5:  2,
    6:  3,
    7:  4,
    8:  5,
    9:  6,
    10: 7,
    11: 8,
    12: 9,
    13: 9,
    14: 9,
    15: 12,
    16: 13,
    17: 14,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21
}

Num2Joints = {
    0:  'mixamorig:Hips',
    1:  'mixamorig:LeftUpLeg',
    2:  'mixamorig:RightUpLeg',
    3:  'mixamorig:Spine',
    4:  'mixamorig:LeftLeg',
    5:  'mixamorig:RightLeg',
    6:  'mixamorig:Spine1',
    7:  'mixamorig:LeftFoot',
    8:  'mixamorig:RightFoot',
    9:  'mixamorig:Spine2',
    10: 'mixamorig:LeftToeBase',
    11: 'mixamorig:RightToeBase',
    12: 'mixamorig:Neck',
    13: 'mixamorig:LeftShoulder',
    14: 'mixamorig:RightShoulder',
    15: 'mixamorig:Head',
    16: 'mixamorig:LeftArm',
    17: 'mixamorig:RightArm',
    18: 'mixamorig:LeftForeArm',
    19: 'mixamorig:RightForeArm',
    20: 'mixamorig:LeftHand',
    21: 'mixamorig:RightHand',
    22: 'mixamorig:LeftHandMiddle1',
    23: 'mixamorig:RightHandMiddle1',
}

Joints2Num = {
    'mixamorig:Hips':     0 ,
    'mixamorig:LeftUpLeg':      1 ,
    'mixamorig:RightUpLeg':      2 ,
    'mixamorig:Spine':     3 ,
    'mixamorig:LeftLeg':     4 ,
    'mixamorig:RightLeg':     5 ,
    'mixamorig:Spine1':     6 ,
    'mixamorig:LeftFoot':    7 ,
    'mixamorig:RightFoot':    8 ,
    'mixamorig:Spine2':     9 ,
    'mixamorig:LeftToeBase':     10,
    'mixamorig:RightToeBase':     11,
    'mixamorig:Neck':       12,
    'mixamorig:LeftShoulder':   13,
    'mixamorig:RightShoulder':   14,
    'mixamorig:Head':       15,
    'mixamorig:LeftArm': 16,
    'mixamorig:RightArm': 17,
    'mixamorig:LeftForeArm':    18,
    'mixamorig:RightForeArm':    19,
    'mixamorig:LeftHand':    20,
    'mixamorig:RightHand':    21,
    'mixamorig:LeftHandMiddle1':     22,
    'mixamorig:RightHandMiddle1':     23,
}


def CreateMesh(pSdkManager, pName, smpl_object):
    # get verticies and triangles
    verticies = smpl_object['vertices']
    triangles = smpl_object['triangles']
    
    # preparation
    lMesh = FbxMesh.Create(pSdkManager, pName)
    lMesh.InitControlPoints(len(verticies))
    lControlPoints = lMesh.GetControlPoints()
    vertexLocList = []

    lMesh.CreateLayer()
    layer0 = lMesh.GetLayer(0)

    # add verticies
    for i in range(0, len(verticies)):
        vertexLoc = FbxVector4(float(verticies[i, 0]), float(verticies[i, 1]), float(verticies[i, 2]))
        vertexLocList.append(vertexLoc)
        lControlPoints[i] = vertexLoc
    
    # add faces (triangles)
    for i in range(0, len(triangles)):
        lMesh.BeginPolygon(i)
        lMesh.AddPolygon(int(triangles[i, 0]))
        lMesh.AddPolygon(int(triangles[i, 1]))
        lMesh.AddPolygon(int(triangles[i, 2]))
        lMesh.EndPolygon()
    
    # assign control points
    for i in range(0, len(verticies)):
        lMesh.SetControlPointAt(lControlPoints[i], i)

    layer_elt = FbxLayerElementVertexColor.Create(lMesh, "")
    direct = layer_elt.GetDirectArray()
    layer_elt.SetMappingMode(FbxLayerElement.eByControlPoint)
    layer_elt.SetReferenceMode(FbxLayerElement.eDirect)
    for c in smpl_object['colors']:
        direct.Add(FbxColor(c[0], c[1], c[2]))
    layer0.SetVertexColors(layer_elt)

    return lMesh

def inv_fbxdouble3(vec, r):
    arr = np.array([vec[0], vec[1], vec[2]])
    new_arr = r.apply(arr)
    return FbxDouble3(float(new_arr[0]), float(new_arr[1]), float(new_arr[2]))

def CreateSkeleton(pSdkManager, pName, smpl_object):
    # read
    joints = smpl_object['joints']
    loc_rot_dict = {}
    for i in range(0, len(joints)):
        counter = 0
        child = -1
        for k in Child2Father:
            if Child2Father[k] == i:
                counter += 1
                child = k
        if counter == 1:
            child_loc = joints[child]
            father_loc = joints[i]
            delta_vec = child_loc - father_loc
            delta_vec = delta_vec / np.linalg.norm(delta_vec)
            starting_vec = np.array([0, 1, 0])
            z_vec = np.array([0, 0, 1])
            neg_z_vec = np.array([0, 0, -1])
            x_vec = np.array([1, 0, 0])
            if i in [11, 20, 18, 16, 13, 14, 17, 19, 21, 23]:
                starting_vec = np.array([[0, 1, 0], [0, 0, 1]])
                delta_vec = np.stack([delta_vec, np.array([0, 0, 1])], 0)
                rot, loss = R.align_vectors(delta_vec, starting_vec)
            elif i == 1:
                additional_angle = 190/180*np.pi
                # additional_angle = -230/180*np.pi
                additional_vec = np.array([np.sin(additional_angle), 0, np.cos(additional_angle)])
                starting_vec = np.stack([starting_vec, z_vec, x_vec], 0)
                delta_vec = np.stack([delta_vec, additional_vec, x_vec], 0)
                rot, loss = R.align_vectors(delta_vec, starting_vec)
            elif i == 2:
                additional_angle = -200/180*np.pi
                # additional_angle = 240/180*np.pi
                additional_vec = np.array([np.sin(additional_angle), 0, np.cos(additional_angle)])
                starting_vec = np.stack([starting_vec, z_vec, x_vec], 0)
                delta_vec = np.stack([delta_vec, neg_z_vec, x_vec], 0)
                rot, loss = R.align_vectors(delta_vec, starting_vec)
            elif i == 4:
                additional_angle = 30/180*np.pi
                # additional_angle = 220/180*np.pi
                additional_vec = np.array([np.sin(additional_angle), 0, np.cos(additional_angle)])
                starting_vec = np.stack([starting_vec, z_vec, x_vec], 0)
                delta_vec = np.stack([delta_vec, neg_z_vec, x_vec], 0)
                rot, loss = R.align_vectors(delta_vec, starting_vec)
            elif i == 5:
                additional_angle = -30/180*np.pi
                # additional_angle = 120/180*np.pi
                additional_vec = np.array([np.sin(additional_angle), 0, np.cos(additional_angle)])
                starting_vec = np.stack([starting_vec, z_vec, x_vec], 0)
                delta_vec = np.stack([delta_vec, neg_z_vec, x_vec], 0)
                rot, loss = R.align_vectors(delta_vec, starting_vec)
            else:
                rot, loss = R.align_vectors(delta_vec.reshape(1, 3), starting_vec.reshape(1, 3))
            loc_rot_dict[i] = rot
            print(i, loss)
        else:
            if i == 0:
                loc_rot_dict[i] = R.from_euler('xyz', [-158 , 0.0 , 0.0], degrees=True)
            else:
                loc_rot_dict[i] = R.identity()

    lSkeletonRootAttribute = FbxSkeleton.Create(pSdkManager, Num2Joints[0])
    lSkeletonRootAttribute.SetSkeletonType(FbxSkeleton.eLimbNode)
    lSkeletonRootAttribute.Size.Set(JointsSize)
    lSkeletonRoot = FbxNode.Create(pSdkManager, Num2Joints[0])
    lSkeletonRoot.SetNodeAttribute(lSkeletonRootAttribute)
    # lSkeletonRoot.LclRotation.Set(FbxDouble3(float(Num2Rot[0][0]), float(Num2Rot[0][1]), float(Num2Rot[0][2])))
    # lSkeletonRoot.LclTranslation.Set(inv_fbxdouble3([float(joints[0, 0]), float(joints[0, 1]), float(joints[0, 2])], Inv_Num2Rot[0]))
    lSkeletonRoot.LclTranslation.Set(FbxDouble3(float(joints[0, 0]), float(joints[0, 1]), float(joints[0, 2])))
    # lclrot = (loc_rot_dict[0]).as_euler('xyz', degrees=True)
    # lSkeletonRoot.LclRotation.Set(FbxDouble3(float(lclrot[0]), float(lclrot[1]), float(lclrot[2])))

    nodeDict = {}
    nodeDict[0] = lSkeletonRoot
    locDict = {0: (float(joints[0, 0]), float(joints[0, 1]), float(joints[0, 2]))}
    # loc_rot_acc_dict = {
    #     0: loc_rot_dict[0]
    # }
    # for i in range(1, len(loc_rot_dict)):
    #     loc_rot_acc_dict[i] = loc_rot_dict[i] * loc_rot_acc_dict[Child2Father[i]]

    for i in range(1, len(joints)):
        skeletonName = Num2Joints[i]
        skeletonAtrribute = FbxSkeleton.Create(pSdkManager, skeletonName)
        skeletonAtrribute.SetSkeletonType(FbxSkeleton.eLimbNode)
        skeletonAtrribute.Size.Set(JointsSize)
        skeletonNode = FbxNode.Create(pSdkManager, skeletonName)
        skeletonNode.SetNodeAttribute(skeletonAtrribute)
        nodeDict[i] = skeletonNode
        locDict[i] = (float(joints[i, 0]), float(joints[i, 1]), float(joints[i, 2]))
        skeletonFather = int(Child2Father[i])
        fatherNode = nodeDict[skeletonFather]
        skeletonNode.LclTranslation.Set(FbxDouble3(float(float(joints[i, 0]) - float(locDict[skeletonFather][0])),
                                                   float(float(joints[i, 1]) - float(locDict[skeletonFather][1])),
                                                   float(float(joints[i, 2]) - float(locDict[skeletonFather][2]))))
        # lclrot = (loc_rot_dict[Child2Father[i]].inv() * loc_rot_dict[i]).as_euler('xyz', degrees=True)
        # skeletonNode.LclRotation.Set(FbxDouble3(float(lclrot[0]), float(lclrot[1]), float(lclrot[2])))
        # skeletonNode.LclTranslation.Set(inv_fbxdouble3([float(float(joints[i, 0]) - float(locDict[skeletonFather][0])),
        #                                            float(float(joints[i, 1]) - float(locDict[skeletonFather][1])),
        #                                            float(float(joints[i, 2]) - float(locDict[skeletonFather][2]))], loc_rot_dict[Child2Father[i]].inv()))
        fatherNode.AddChild(skeletonNode)

    return lSkeletonRoot, nodeDict


def LinkMeshToSkeleton(pSdkManager, pMeshNode, lSkin, smpl_object, nodeDict):
    blend_weights = smpl_object['blend_weights']  # shape: (24, xxxxx)
    for i in range(0, blend_weights.shape[0]):
        skeletonNode = nodeDict[i]
        skeletonName = skeletonNode.GetName()
        skeletonNum = Joints2Num[str(skeletonName)]
        skeletonWeightsInfo = blend_weights[skeletonNum]
        skeletonCluster = FbxCluster.Create(pSdkManager, "")
        skeletonCluster.SetLink(skeletonNode)
        skeletonCluster.SetLinkMode(FbxCluster.eNormalize)
        for j in range(0, blend_weights.shape[1]):
            skeletonCluster.AddControlPointIndex(j, float(skeletonWeightsInfo[j]))

        # Now we have the Mesh and the skeleton correctly positioned,
        # set the Transform and TransformLink matrix accordingly.
        lXMatrix = FbxAMatrix()
        lScene = pMeshNode.GetScene()
        if lScene:
            lXMatrix = lScene.GetAnimationEvaluator().GetNodeGlobalTransform(pMeshNode)
        skeletonCluster.SetTransformMatrix(lXMatrix)
        lScene = skeletonNode.GetScene()
        if lScene:
            lXMatrix = lScene.GetAnimationEvaluator().GetNodeGlobalTransform(skeletonNode)
        skeletonCluster.SetTransformLinkMatrix(lXMatrix)
        # Add the clusters to the Mesh by creating a skin and adding those clusters to that skin.
        # After add that skin.
        lSkin.AddCluster(skeletonCluster)

    pMeshNode.GetNodeAttribute().AddDeformer(lSkin)


def AddShape(pScene, node, smpl_object):
    verticies = smpl_object['vertices']


    lBlendShape = FbxBlendShape.Create(pScene, "BlendShapes")

    for j in range(0, 1):
        lBlendShapeChannel = FbxBlendShapeChannel.Create(pScene, "ShapeChannel"+str(j))
        lShape = FbxShape.Create(pScene, "Shape"+str(j))
        lShape.InitControlPoints(len(verticies))
        for i in range(0, len(verticies)):
            ctrlPInfo = verticies[i]
            lShape.SetControlPointAt(FbxVector4(0, 0, 0), i) # (?) Align with Smplx2FBX. To be further investigated.
        lBlendShapeChannel.AddTargetShape(lShape)
        lBlendShape.AddBlendShapeChannel(lBlendShapeChannel)
    node.GetMesh().AddDeformer(lBlendShape)


def CreateScene(pSdkManager, pScene, smpl_object):
    # Create scene info
    lSceneInfo = FbxDocumentInfo.Create(pSdkManager, "SceneInfo")
    lSceneInfo.mTitle = "SMPL"
    lSceneInfo.mSubject = "Human SMPL model with weighted skin"
    lSceneInfo.mAuthor = "MMLab@NTU"
    lSceneInfo.mRevision = "rev. 1.0"
    lSceneInfo.mKeywords = "human smpl weighted"
    lSceneInfo.mComment = "N/A"
    pScene.SetSceneInfo(lSceneInfo)

    lMeshNode = FbxNode.Create(pScene, "meshNode")
    smplMesh = CreateMesh(pSdkManager, "Mesh", smpl_object)
    lControlPoints = smplMesh.GetControlPoints()
    lMeshNode.SetNodeAttribute(smplMesh)

    lSkeletonRoot, nodeDict = CreateSkeleton(pSdkManager, "Skeleton", smpl_object)

    pScene.GetRootNode().AddChild(lMeshNode)
    pScene.GetRootNode().AddChild(lSkeletonRoot)

    lSkin = FbxSkin.Create(pSdkManager, "")

    LinkMeshToSkeleton(pSdkManager, lMeshNode, lSkin, smpl_object, nodeDict)
    AddShape(pScene, lMeshNode, smpl_object)  # Might be redundant!
    # AnimateSkeleton(pSdkManager, pScene, lSkeletonRoot)