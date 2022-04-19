import torch
import zarr
import numpy as np
import vcf_data_loader
from vcf_data_loader import FixedSizeVCFChunks

from models import ChildModel, ParentModel
from block_utils import Indexer

class SaveModelReconstructions():
    def __init__(self, num_parent_blocks, dataset, samples = 2548) -> None:
        self.num_parent_blocks = num_parent_blocks
        self.dataset = dataset
        self.samples = samples
        pass

    def SaveAndWriteZarrArrays(self):
        saved_file = save_predictions(self.num_parent_blocks, self.samples, self.dataset)
        combine_overlapping_predictions(saved_file)


def load_parent_model():
    # Load model a.
    pass


def save_predictions(num_parent_blocks, samples, dataset, block_size = 1000):
    vcf = FixedSizeVCFChunks(
        dataset
    )
    indexer = Indexer(vcf, block_size)

    n = samples
    filename = "../zarr_arrays/parent_block_outputs.zarr"
    
    z = zarr.open(
        filename,
        mode="w",
        shape=(n, 3, num_parent_blocks * block_size * 2),
        chunks=(n, 3, 4000),
        dtype=np.float32,
    )
    
    for i in range(0,num_parent_blocks):
        modela = ChildModel(block_size, 16)
        modelb = ChildModel(block_size, 16)
        modelc = ChildModel(2* block_size, 32)
        parent = ParentModel(modela.encoder, modelb.encoder, modelc.decoder)

        # Load the weights from a pretrained save parent model.
        parent.load_state_dict(
            torch.load(f"../saved-models/model-{i}-{i+1}.pth")
        )
        parent.eval()

        data_a = indexer.extract_chunk(i).to(torch.float32)
        data_b = indexer.extract_chunk(i+1).to(torch.float32)

        predictions = parent(data_a, data_b).detach().numpy()

        z[:, :, i*2*block_size:(i+1)*2*block_size] = predictions

    print("LOAD MODELS COMPLETE")
    return filename

def combine_chunks(chunk1, chunk2):
    combined = np.stack((chunk1, chunk2), axis=3)
    combined = np.mean(combined, axis=3)

    return combined


def write_chunk(z, left_index, chunk, hard_predictions):
    """Write the array "chunk" into the zarr array "z" starting on the third
    dimension at index left_index.

    """
    if hard_predictions:
        chunk_size = chunk.shape[2]
        chunk = np.argmax(chunk, axis=1)
        z[:, left_index:(left_index+chunk_size)] = chunk
    else: 
        chunk_size = chunk.shape[2]
        z[:, :, left_index:(left_index+chunk_size)] = chunk

def combine_overlapping_predictions(filename, chunk_size = 1000):
    z = zarr.open(filename)
    n = z.shape[0]

    output_n_snps = ((z.shape[2] - 2 * chunk_size) // 2) + (2 * chunk_size)
    output_z_hard = zarr.open(
        f"../zarr_arrays/hard_predictions.zarr",
        shape=(n, output_n_snps),
        dtype=np.float32,
        mode="w"
    )

    # We remove the first and last chunk because they don't have overlapping
    # predictions. The final size should be the remaining divided by two.
    output_z_soft = zarr.open(
        "../zarr_arrays/combined_predictions.zarr",
        shape=(n, 3, output_n_snps),
        dtype=np.float32,
        mode="w"
    )

    # Write the first output (that has no overlapping models).
    write_chunk(output_z_hard, 0, z[:, :, :chunk_size], True)
    write_chunk(output_z_soft, 0, z[:, :, :chunk_size], False)

    chunk_left = chunk_size  # Left index of the chunk.
    left_index = chunk_size

    # Iterate up to last block.
    while left_index < output_n_snps-chunk_size:

        chunk_l = z[:, :, chunk_left:chunk_left+chunk_size]
        chunk_r = z[:, :, (chunk_left+chunk_size):(chunk_left+2*chunk_size)]

        combined = combine_chunks(chunk_l, chunk_r)

        write_chunk(output_z_hard, left_index, combined, True)
        write_chunk(output_z_soft, left_index, combined, False)
        chunk_left += 2*chunk_size
        left_index += chunk_size

    # Write the last output.
    chunk_left = output_n_snps-chunk_size
    write_chunk(output_z_hard, chunk_left, z[:, :, -chunk_size:], True)
    write_chunk(output_z_soft, chunk_left, z[:, :, -chunk_size:], False)