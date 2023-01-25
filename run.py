import sys

import ml3d.torch as ml3d

import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from ml3d.utils.config import Config
from ml3d.torch.models.randlanet import RandLANet
from ml3d.datasets.s3dis import S3DIS
from  ml3d.torch.pipelines.semantic_segmentation import SemanticSegmentation
import warnings
warnings.filterwarnings("ignore")


# Make some changes to the open3d.ml.torch module

# Reload the module



cfg_file = "/content/OML/ml3d/configs/randlanet_s3dis.yml"
cfg = Config.load_from_file(cfg_file)

model = RandLANet(**cfg.model)
cfg.dataset['dataset_path'] = "/content/Stanford3dDataset_v1.2_Aligned_Version"
dataset = S3DIS(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = SemanticSegmentation(model, dataset=dataset, device="gpu", **cfg.pipeline)
pipeline.run_train()
