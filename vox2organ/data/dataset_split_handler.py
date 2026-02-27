
""" Convenient dataset splitting. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from data.cortex import CortexDataset
from data.abdomen import AbdomenCTDataset, AbdomenMRIDataset
from data.voxels2vertices import Voxels2VerticesDataset
from data.supported_datasets import (
    CortexDatasets,
    AbdomenCTDatasets,
    AbdomenMRIDatasets,
    SupportedDatasets,
)

# Mapping supported datasets to split functions
dataset_split_handler = {
    **{x.name: CortexDataset.split for x in CortexDatasets},
    **{x.name: AbdomenCTDataset.split for x in AbdomenCTDatasets},
    **{x.name: AbdomenMRIDataset.split for x in AbdomenMRIDatasets},
}

# Custom loader for flat voxels2vertices exports
dataset_split_handler[SupportedDatasets.VOXELS2VERTICES.name] = Voxels2VerticesDataset.split
