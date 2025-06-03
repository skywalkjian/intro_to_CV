from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class PointNetfeat(nn.Module):
    '''
        The feature extractor in PointNet, corresponding to the left MLP in the pipeline figure.
        Args:
        d: the dimension of the global feature, default is 1024.
        segmentation: whether to perform segmentation, default is True.
    '''
    def __init__(self, segmentation = False, d=1024):
        super(PointNetfeat, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the feature extractor. ##
        self.d=d    
        self.segmentation=segmentation
        self.fc1=nn.Linear(3, 64)
        self.fc2=nn.Linear(64, 128)
        self.fc3=nn.Linear(128, d)
        self.bn1=nn.BatchNorm1d(64)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(d)
        self.relu=nn.ReLU()

                
        ## ------------------------------------------- ##

    def forward(self, x):
        '''
            If segmentation == True
                return the concatenated global feature and local feature. # (B, d+64, N)
            If segmentation == False
                return the global feature, and the per point feature for cruciality visualization in question b). # (B, d), (B, N, d)
            Here, B is the batch size, N is the number of points, d is the dimension of the global feature.
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass. ##  
        x=self.fc1(x)
        #print(x.shape)
        x = x.transpose(2, 1)
        x = self.bn1(x)
        x = self.relu(x)
        lf=x# (B, 64, N)
        x=x.transpose(2, 1)
        x=self.fc2(x)
        #print(x.shape)
        x = x.transpose(2, 1)
        x = self.bn2(x) 
        x = self.relu(x)
        x=x.transpose(2, 1)
        x=self.fc3(x)
        #print(x.shape)
        x = x.transpose(2, 1)
        x = self.bn3(x)
        x = self.relu(x)# (B, d, N)
        ppf=x.transpose(2,1)# (B, N,d)
        #print(x.shape)
        x=nn.MaxPool1d(kernel_size=x.shape[2])(x)#(B,d,1)
        #print(x.shape,1)
        gf=x.squeeze(-1)#(B,d)

        if self.segmentation:
            x=x.expand(-1, -1, lf.shape[2])#(B,d,N)
            #print(x.shape,lf.shape)
            x=torch.cat((x, lf), dim=1)# (B, d+64, N)
            return x
        else:
            return gf, ppf




        ## ------------------------------------------- ##


class PointNetCls1024D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the middle right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the classifier.        ##
        ## ------------------------------------------- ##
        self.PointNetfeat= PointNetfeat(segmentation=False, d=1024)
        self.fc1 = nn.Linear(1024, 512)

        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=1024)
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        gf, ppf = self.PointNetfeat(x)
        gf= self.fc1(gf)
        gf = self.bn1(gf)
        gf = self.relu(gf)
        gf = self.fc2(gf)
        gf = self.bn2(gf)
        gf = self.relu(gf)
        gf = self.fc3(gf)
        gf = F.log_softmax(gf, dim=1)  # (B, k)
        return gf, ppf
        

class PointNetCls256D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the upper right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2 ):
        super(PointNetCls256D, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the classifier.        ##
        ## ------------------------------------------- ##
        self.PointNetfeat = PointNetfeat(segmentation=False, d=256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()


    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=256)
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        gf, ppf = self.PointNetfeat(x)
        gf = self.fc1(gf)
        gf = self.bn1(gf)
        gf = self.relu(gf)
        gf = self.fc2(gf)
        gf = F.log_softmax(gf, dim=1)
        return gf, ppf


class PointNetSeg(nn.Module):
    '''
        The segmentation head in PointNet, corresponding to the lower right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k = 2):
        super(PointNetSeg, self).__init__()
        ## ------------------- TODO ------------------- ##
        ## Define the layers in the segmentation head. ##
        ## ------------------------------------------- ##
        self.PointNetfeat = PointNetfeat(segmentation=True, d=1024)
        self.fc1 = nn.Linear(1024 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
            Input:
                x: the input point cloud. # (B, N, 3)
            Output:
                the log softmax of the segmentation result. # (B, N, k)
        '''
        ## ------------------- TODO ------------------- ##
        ## Implement the forward pass.                 ##
        ## ------------------------------------------- ##
        x= self.PointNetfeat(x)# (B, d+64, N)
        x = x.transpose(2, 1)# (B, N, d+64)
        x = self.fc1(x)# (B, N, 512)
        x=x.transpose(2, 1)

        x = self.bn1(x)
        x = self.relu(x)
        x = x.transpose(2, 1)
        x = self.fc2(x)
        x = x.transpose(2, 1)
        x = self.bn2(x)
        x = self.relu(x)
        x = x.transpose(2, 1)
        x = self.fc3(x)
        x = x.transpose(2, 1)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.transpose(2, 1)
        x = self.fc4(x)# (B, N, k)
        x = F.log_softmax(x, dim=2)
        return x

