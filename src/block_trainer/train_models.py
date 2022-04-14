import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn, optim
import pytorch_lightning as pl
import torch.nn.functional as F
import vcf_data_loader
import models 
from models import ChildModel,ParentModel
import argparse

def train_child_block_models(n_blocks, dataset, block_size = 1000, samples = 2548,  epochs=100):
    
    parser = parse_args()

    if parser.arrays != -1:
        index = parser.arrays
        genotype_dataset = TensorDataset(torch.tensor(dataset, dtype=torch.float32))
        train_child_block(index, genotype_dataset, block_size, samples, epochs)
    else:
        for index in range(n_blocks):
            x = dataset[:,index*block_size:(index+1)*block_size]
            genotype_dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
            train_child_block(index, genotype_dataset, block_size, samples, epochs)



def train_child_block(index, tensor_dataset, block_size, samples, epochs = 100):
    print("-----------------------------------")
    print("Child Block Training Process Begin")
    print("-----------------------------------")
    train_loader = DataLoader(tensor_dataset, batch_size = samples, num_workers=0)
    model = ChildModel(block_size, 16)
    trainer = pl.Trainer(
    log_every_n_steps=1,    # set the logging frequency
    gpus=-1,                # use all GPUs
    max_epochs=epochs,      # number of epochs
    deterministic=False,    # keep it non-deterministic
    auto_lr_find = True     # Find the learning rate
    )
    trainer.fit(model, train_loader)
    print('Training process has finished. Saving trained model.')
    print('Starting testing')
    #Save model 
    path = f"../saved-models/child_model_block-{index}.pth"
    torch.save(model.state_dict(), path)

def train_parent_block_models(n_blocks, block_size = 1000, samples = 2548,  epochs=200, *datasets):
    
    parser = parse_args()

    if len(datasets) != 1 or len(datasets) !=2: 

        raise ValueError("Must have one or two datasets as input.")
    
    if parser.arrays != -1:

        if len(datasets) !=2: 

            raise ValueError("Must have two datasets representing two adjacent blocks.")
        
        index = parser.arrays
        genotype_dataset = TensorDataset(
        datasets[0].to(torch.float32),
        datasets[1].to(torch.float32))
        train_parent_block(index, genotype_dataset, block_size, samples, epochs)
    
    else:

        if len(datasets) != 1: 

            raise ValueError("Must have one dataset spanning two adjacent blocks.")
        
        for index in range(n_blocks-1):
            x1 = datasets[0][:,index*block_size:(index+1)*block_size]
            x2 = datasets[0][:,(index+1)*block_size:(index+2)*block_size]
            genotype_dataset = TensorDataset(torch.tensor(x1, dtype=torch.float32),torch.tensor(x2, dtype=torch.float32) )
            train_child_block(index, genotype_dataset, block_size, samples, epochs)

def train_parent_block(index, dataset, block_size, samples, epochs):


    modela = ChildModel(block_size, 16)
    modela.load_state_dict(
        torch.load(f"../saved-models/model-{index}.pth")
    )  # REPLACE WITH BLOCK NUMBER

    modelb = ChildModel(block_size, 16)
    modelb.load_state_dict(
        torch.load(f"../saved-models/model-{index+1}.pth")
    )  # REPLACE WITH BLOCK NUMBER

    modelc = ChildModel(2*block_size, 32)

    parent = ParentModel(modela.encoder, modelb.encoder, modelc.decoder)

    print("-----------------------------------")
    print("Training Process Begin")
    print("-----------------------------------")



    train_loader = DataLoader(dataset, batch_size=samples)

    trainer = pl.Trainer(
        log_every_n_steps=1,    # set the logging frequency
        gpus=-1,                # use all GPUs
        max_epochs=epochs,      # number of epochs
        deterministic=False,     # keep it deterministic
    )

    trainer.fit(parent, train_loader)
    # Save model
    path = (
        f"../saved-models/model-{index}-{index+1}.pth"
    )
    torch.save(parent.state_dict(), path)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--arrays",
     type=int, default=-1, 
     help="If training is done parallel with SLURM arrays instead of iteratively.")

    return parser.parse_args()