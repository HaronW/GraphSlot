"""Return model, loss, and eval metrics in 1 go 
for the SAVi model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graphslot.lib.utils import init_fn

import graphslot.modules as modules
import graphslot.modules.misc as misc
from graphslot.modules import CNN2, ResidualBlock


def build_model(args):
	slot_size = 128
	gcn_hidden_dim = 128
	num_slots = args.num_slots
	weight_init = args.weight_init
	# Encoder
	encoder_backbone = modules.CNN2(
    	conv_modules=nn.ModuleList([
        	nn.Conv2d(3, 64, kernel_size=3, padding=1),
        	nn.Conv2d(64, 64, kernel_size=3, padding=1),
        
        	ResidualBlock(64, 128, stride=2),
        	ResidualBlock(128, 128),
        
        	ResidualBlock(128, 256, stride=2),
    		ResidualBlock(256, 256),
        
    		ResidualBlock(256, 512, stride=2),
        	ResidualBlock(512, 512),
        
        	ResidualBlock(512, 512, stride=2),
    		ResidualBlock(512, 512)
		]),
    	weight_init=weight_init
	)
	encoder = modules.FrameEncoder(
		backbone=encoder_backbone,
			pos_emb=modules.PositionEmbedding(
				input_shape=(-1, 8, 8, 512),
				embedding_type="linear",
				update_type="project_add",
				output_transform=modules.MLP(
					input_size=512,
					hidden_size=512,
					output_size=512,
					layernorm="pre",
					weight_init=weight_init),
				weight_init=weight_init)
		)

	# Corrector
	corrector = modules.GraphCorrector(
		slot_attention = modules.SlotAttention(
			input_size=512, 
			qkv_size=128,
			slot_size=slot_size,
			num_iterations=1,
			weight_init=weight_init,
			),
		construct_graph = modules.ConstructGraph(),
    	graph_emb = modules.GraphEmb(hidden_dim=gcn_hidden_dim),
		alpha = 1.0,
		beta = 1.0
		)

	# Predictor
	predictor = modules.TransformerBlock(
		embed_dim=slot_size,
		num_heads=4,
		qkv_size=128,
		mlp_size=256,
		weight_init=weight_init)
	# Initializer
	initializer = modules.CoordinateEncoderStateInit(
		embedding_transform=modules.MLP(
			input_size=4,
			hidden_size=256,
			output_size=slot_size,
			layernorm=None,
			weight_init=weight_init),
		prepend_background=True,
		center_of_mass=False)
	# Decoder
	readout_modules = nn.ModuleList([
		nn.Linear(64, out_features) for out_features in args.targets.values()])
	for module in readout_modules.children():
		init_fn[weight_init['linear_w']](module.weight)
		init_fn[weight_init['linear_b']](module.bias)

	decoder_backbone = CNN2(
		conv_modules=nn.ModuleList([
      		nn.ConvTranspose2d(slot_size, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
      		ResidualBlock(512, 512),
      
    		nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  
  			ResidualBlock(256, 256),
      
      		nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
      		ResidualBlock(128, 128),
      
      		nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
      		ResidualBlock(64, 64)
  		]),
		transpose_modules=[True, False, True, False, True, False, True, False],
  		weight_init=weight_init
	)
	decoder = modules.SpatialBroadcastDecoder(
		resolution=(8,8), # Update if data resolution or strides change.
		backbone=decoder_backbone,
		pos_emb=modules.PositionEmbedding(
			input_shape=(-1, 8, 8, slot_size),
			embedding_type="linear",
			update_type="project_add",
			weight_init=weight_init),
		target_readout=modules.Readout(
			keys=list(args.targets),
			readout_modules=readout_modules),
			weight_init=weight_init)
	# SAVi Model
	model = modules.SAVi(
		encoder=encoder,
		decoder=decoder,
		corrector=corrector,
		predictor=predictor,
		initializer=initializer,
		decode_corrected=True,
		decode_predicted=False)
	
	return model


def build_modules(args):
	"""Return the model and loss/eval processors."""
	model = build_model(args)	
	loss = misc.ReconLoss()
	metrics = misc.ARI()

	return model, loss, metrics
