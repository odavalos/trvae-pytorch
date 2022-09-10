# install anywhere pytorch and scanpy are installed

# this script is based on the example from the trvaep github page
# https://github.com/theislab/trvaep
# https://nbviewer.org/github/theislab/trvaep/blob/master/example/sample_notebook.ipynb
# https://nbviewer.org/github/theislab/trvaep/blob/master/example/multi_condition_sample.ipynb


import argparse
import scanpy as sc
import sys
sys.path.append("../")
import trvae
import numpy as np


# creating the parser
parser = argparse.ArgumentParser()

parser.add_argument('--in_path', type = str, required = True, help = 'Path to adata object')
parser.add_argument('--out_path', type = str, required = True, help = 'Path for output anndata object')
parser.add_argument('--latent_dim', type = int, default = 10, help = 'Number of latent dimensions')
parser.add_argument('--condition_key', type = str, required = True, help = 'Name of column containing the scRNASeq batch information')
parser.add_argument('--target_cond', type = str, required = True, help = 'Name of condition want to use as target for mapping data onto')
parser.add_argument('--celltype_key', type = str, required = True, help = 'Name of column containing the scRNASeq celltype information')
parser.add_argument('--n_epochs', type = int, default = 50, help = 'Number of epochs')
parser.add_argument('--batch_size', type = int, default = 512, help = 'Model batch size')
parser.add_argument('--out_name', type = str, default = '', required = False, help = 'Name for output h5ad file. Default name will be trvae_corrected.h5ad')

# parsing the arguments
args = parser.parse_args()

# reading in dataset
adata = sc.read_h5ad(args.in_path)

# column names for batch and celltype information
condition_key = args.condition_key
celltype_key = args.celltype_key
target_cond_key = args.target_cond

# make a 'condition' column from the batch column
adata.obs['condition'] = adata.obs[condition_key]

# skipping normalization and log transformation since for my data it was already done
# also skipping the visualization of the data running on a cluster

# calculate number of batches
conditions = len(adata.obs[condition_key].unique().tolist())

# set up the model
model = trvae.CVAE(adata.shape[1], 
                    num_classes=conditions,
                    encoder_layer_sizes=[128, 32], 
                    decoder_layer_sizes=[32, 128], 
                    latent_dim=args.latent_dim,
                    alpha=0.0001,
                    use_mmd=True, 
                    beta=10)

# train the model
trainer = trvae.Trainer(model, adata)
trainer.train_trvae(args.n_epochs, args.batch_size, early_patience=30)


# get the latent space data
latent_y = model.get_y(adata.X.A, model.label_encoder.transform(adata.obs[condition_key])) # get the latent space
adata_latent = sc.AnnData(latent_y)

# add original metadata to the latent space adata object
adata_latent.obs[celltype_key] = adata.obs[celltype_key].tolist() # add celltype information
adata_latent.obs[condition_key] = adata.obs[condition_key].tolist() # add batch information

# save the latent space data
adata_latent.write_h5ad(args.out_path + args.out_name + '_trvaep_latent.h5ad')

# get the expression corrected data
batch_removed = model.predict(x=adata.X.todense(), y=adata.obs[condition_key].tolist(), target=target_cond_key)
corrected_data = sc.AnnData(batch_removed)
# add original metadata to the corrected adata object
corrected_data.obs[celltype_key] = adata.obs[celltype_key].tolist()
corrected_data.obs[condition_key] = adata.obs[condition_key].tolist()

# save the corrected data
corrected_data.write_h5ad(args.out_path + args.out_name + '_trvaep_corrected.h5ad')
