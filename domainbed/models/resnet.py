# coding:utf-8

'''
models libary
you are supposed to load imagenet pretrained model
and locate the model into the specific path './checkpoints/imagenet_pretrained'
generally, any model architecture can be used for representation transfer
'''

import math
import os
import logging

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision
import torch

from timm.models.layers import trunc_normal_, DropPath

logger = logging.getLogger('root')

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet152']

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
	'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
}


class Bottleneck(nn.Module):
	expansion = 4
	
	def __init__(self, inplanes, planes, stride=1, downsample=None,
	             drop_path=0., layer_scale_init_value=1e-6):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
		                       padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		# self.act = nn.GELU()
		self.downsample = downsample
		self.stride = stride
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((planes * 4)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
	
	def forward(self, x):
		residual = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.act(out)
		
		out = self.conv3(out)
		out = self.bn3(out)
		# if self.gamma is not None:
		# 	out = out.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
		# 	out = self.gamma * out
		# 	out = out.permute(0, 3, 1, 2)
		
		if self.downsample is not None:
			residual = self.downsample(x)
		
		out += residual
		# out = residual + self.drop_path(out)
		out = self.act(out)
		
		return out


class ResNetFeature(nn.Module):
	
	def __init__(self, block, layers):
		self.inplanes = 64
		super(ResNetFeature, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		                       bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
	
	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		
		return nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		
		return x


def resnet18(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetFeature(Bottleneck, [2, 2, 2, 2], **kwargs)
	if pretrained:
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
		pretrained_model = torchvision.models.resnet18(pretrained=pretrained)
		load_pretrained_model(model, pretrained_model)
	return model


def resnet34(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetFeature(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
	return model


def resnet50(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetFeature(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
		pretrained_model = torchvision.models.resnet50(pretrained=pretrained)
		load_pretrained_model(model, pretrained_model)
		
	return model


def resnet101(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetFeature(Bottleneck, [3, 4, 23, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
	return model


def resnet152(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetFeature(Bottleneck, [3, 8, 36, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
	return model


def load_pretrained_model(model, pretrained_model, omit_module=['fc', 'gamma']):
	pretrained_dict = pretrained_model.state_dict()
	
	model_dict = model.state_dict()
	
	for k, v in model_dict.items():
		if any([x in k for x in omit_module]):
			pass
		else:
			model_dict[k] = pretrained_dict[k]
	
	model.load_state_dict(model_dict)

