# coding: utf-8

'''
custom loss function
'''

import math
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F


# # =========================  proxy Contrastive loss ==========================
class ProxyLoss(nn.Module):
	'''
	pass
	'''
	
	def __init__(self, scale=1, thres=0.1):
		super(ProxyLoss, self).__init__()
		self.scale = scale
		self.thres = thres
	
	def forward(self, feature, pred, target):
		feature = F.normalize(feature, p=2, dim=1)  # normalize
		feature = torch.matmul(feature, feature.transpose(1, 0))  # (B, B)
		label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
		feature = feature * ~label_matrix  # get negative matrix
		feature = feature.masked_fill(feature < self.thres, -np.inf)
		pred = torch.cat([pred, feature], dim=1)  # (N, C+N)
		
		loss = F.nll_loss(F.log_softmax(self.scale * pred, dim=1), \
		                  target)
		
		return loss


class ProxyPLoss(nn.Module):
	'''
	pass
	'''
	
	def __init__(self, num_classes, scale):
		super(ProxyPLoss, self).__init__()
		self.soft_plus = nn.Softplus()
		self.label = torch.LongTensor([i for i in range(num_classes)])
		self.scale = scale
	
	def forward(self, feature, target, proxy):
		feature = F.normalize(feature, p=2, dim=1)
		pred = F.linear(feature, F.normalize(proxy, p=2, dim=1))  # (N, C)
		label = (self.label.unsqueeze(1).to(feature.device) == target.unsqueeze(0))  # (C, N)
		pred = torch.masked_select(pred.transpose(1, 0), label)  # N,
		
		pred = pred.unsqueeze(1)  # (N, 1)
		
		feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
		label_matrix = target.unsqueeze(1) == target.unsqueeze(0)  # (N, N)
		
		index_label = torch.LongTensor([i for i in range(feature.shape[0])])  # generate index label
		index_matrix = index_label.unsqueeze(1) == index_label.unsqueeze(0)  # get index matrix
		
		feature = feature * ~label_matrix  # get negative matrix
		feature = feature.masked_fill(feature < 1e-6, -np.inf)  # (N, N)
		
		logits = torch.cat([pred, feature], dim=1)  # (N, 1+N)
		label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
		loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)
		
		return loss


class PosAlign(nn.Module):
	'''
	pass
	'''
	
	def __init__(self):
		super(PosAlign, self).__init__()
		self.soft_plus = nn.Softplus()
	
	def forward(self, feature, target):
		feature = F.normalize(feature, p=2, dim=1)
		
		feature = torch.matmul(feature, feature.transpose(1, 0))  # (N, N)
		label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
		
		positive_pair = torch.masked_select(feature, label_matrix)
		
		# print("positive_pair.shape", positive_pair.shape)
		
		loss = 1. * self.soft_plus(torch.logsumexp(positive_pair, 0))
		
		return loss


if __name__ == "__main__":
	pcl = ProxyPLoss(num_classes=100, scale=12)
	fea = torch.rand(256, 128)
	target = torch.randint(high=100, size=(256,))
	proxy = torch.rand(100, 128)
	
	out = pcl(fea, target, proxy)
	print("out", out)