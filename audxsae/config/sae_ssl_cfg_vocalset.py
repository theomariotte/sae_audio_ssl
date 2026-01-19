import os
import random
import socket
import string

cluster = socket.gethostname()
slurm = "SLURM_JOB_ID" in os.environ

EXP_ROOT = os.environ["EXP_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]

if slurm:
  job_id = os.environ["SLURM_JOB_ID"]
else:
  job_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))

common_parameters = {
    #'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ssl'),
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ssl'),
    'sample_rate': 16000,
    'optim': {
        'epochs': 100,
        'batch_size': 16,  
        'learning_rate': 0.001,
        'patience_early_stop': 15,
        'patience_scheduler': 200,
        'loss_weights': (1.0, 0.0) # BCE, MSE Spectro
    },
    'data': {
      "dataset_name": "vocalset",
      "train": {
        "root": os.path.join(DATA_ROOT,"VocalSet/FULL"),
        "split": "train",
        "seed": 42,
        "ratio": 0.8,
        "duration": 5.0,
        "target_sr": 16000,},
      "valid": {
        "root": os.path.join(DATA_ROOT,"VocalSet/FULL"),
        "split": "valid",
        "seed": 42,
        "ratio": 0.8, # ratio is the amount of train data. The <split> parameter allows to select the train files or the valid files
        "duration": 5.0,
        "target_sr": 16000,},
    },
    'model': {
      "encoder_type": "wavlm",
      "sae_method": "top-k",
      "sae_dim": 2048,
      "sparsity": 0.9,
      "freeze": True,
      "layer_indices": [5,6,7,8]
    },
    
    "job_id": job_id,
    "cluster": cluster
}
conf = {
  # AST
  "001": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.95,
        "freeze": True,
        "layer_indices": [6, 12],
        "pooling_method": "mean"
        },
    },
  "002": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [6, 12],
        "pooling_method": "mean"
        },
    },
  "003": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.85,
        "freeze": True,
        "layer_indices": [6, 12],
        "pooling_method": "mean"
        },
    },
  "004": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.8,
        "freeze": True,
        "layer_indices": [6, 12],
        "pooling_method": "mean"
        },
    },
  "005": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.75,
        "freeze": True,
        "layer_indices": [6, 12],
        "pooling_method": "mean"
        },
    },
  # WAVLM
  "006": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.95,
        "freeze": True,
        "layer_indices": [2, 12],
        "pooling_method": "mean"
        },
    },
  "007": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [2, 12],
        "pooling_method": "mean"
        },
    },
  "008": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.85,
        "freeze": True,
        "layer_indices": [2, 12],
        "pooling_method": "mean"
        },
    },
  "009": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.8,
        "freeze": True,
        "layer_indices": [2, 12],
        "pooling_method": "mean"
        },
    },
  "010": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.75,
        "freeze": True,
        "layer_indices": [2, 12],
        "pooling_method": "mean"
        },
    },
  # MERT
  "011": {
        "sample_rate": 24000,
        'model': {
          "encoder_type": "mert",
          "sae_method": "top-k",
          "sae_dim": 2048,
          "sparsity": 0.95,
          "freeze": True,
          "layer_indices": [4, 7],
          "pooling_method": "mean"
        },
        'data': {
          "train": {
            "target_sr": 24000,},
          "valid": {
            "target_sr": 24000,},
        },
    },
  "012": {
        "sample_rate": 24000,
        'model': {
          "encoder_type": "mert",
          "sae_method": "top-k",
          "sae_dim": 2048,
          "sparsity": 0.9,
          "freeze": True,
          "layer_indices": [4, 7],
          "pooling_method": "mean"
        },
        'data': {
          "train": {
            "target_sr": 24000,},
          "valid": {
            "target_sr": 24000,},
        },
    },
  "013": {
        "sample_rate": 24000,
        'model': {
          "encoder_type": "mert",
          "sae_method": "top-k",
          "sae_dim": 2048,
          "sparsity": 0.85,
          "freeze": True,
          "layer_indices": [4, 7],
          "pooling_method": "mean"
        },
        'data': {
          "train": {
            "target_sr": 24000,},
          "valid": {
            "target_sr": 24000,},
        },
    },
  "014": {
        "sample_rate": 24000,
        'model': {
          "encoder_type": "mert",
          "sae_method": "top-k",
          "sae_dim": 2048,
          "sparsity": 0.8,
          "freeze": True,
          "layer_indices": [4, 7],
          "pooling_method": "mean"
        },
        'data': {
          "train": {
            "target_sr": 24000,},
          "valid": {
            "target_sr": 24000,},
        },
    },
  "015": {
        "sample_rate": 24000,
        'model': {
          "encoder_type": "mert",
          "sae_method": "top-k",
          "sae_dim": 2048,
          "sparsity": 0.75,
          "freeze": True,
          "layer_indices": [4, 7],
          "pooling_method": "mean"
        },
        'data': {
          "train": {
            "target_sr": 24000,},
          "valid": {
            "target_sr": 24000,},
        },
    },
  # HUBERT
  "016": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.95,
        "freeze": True,
        "layer_indices": [4],
        "pooling_method": "mean"
        },
    },
  "017": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [4],
        "pooling_method": "mean"
        },
    },
  "018": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.85,
        "freeze": True,
        "layer_indices": [4],
        "pooling_method": "mean"
        },
    },
  "019": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.8,
        "freeze": True,
        "layer_indices": [4],
        "pooling_method": "mean"
        },
    },
  "020": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.75,
        "freeze": True,
        "layer_indices": [4],
        "pooling_method": "mean"
        },
  },
  # HUBERT
  "021": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.95,
        "freeze": True,
        "layer_indices": [3, 12],
        "pooling_method": "mean"
        },
    },
  "022": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [3, 12],
        "pooling_method": "mean"
        },
    },
  "023": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.85,
        "freeze": True,
        "layer_indices": [3, 12],
        "pooling_method": "mean"
        },
    },
  "024": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.8,
        "freeze": True,
        "layer_indices": [3, 12],
        "pooling_method": "mean"
        },
    },
  "025": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.75,
        "freeze": True,
        "layer_indices": [3, 12],
        "pooling_method": "mean"
        },
  },
  # WAVLM
  "026": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.95,
        "freeze": True,
        "layer_indices": [1, 12],
        "pooling_method": "mean"
        },
    },
  "027": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.9,
        "freeze": True,
        "layer_indices": [1, 12],
        "pooling_method": "mean"
        },
    },
  "028": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.85,
        "freeze": True,
        "layer_indices": [1, 12],
        "pooling_method": "mean"
        },
    },
  "029": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.8,
        "freeze": True,
        "layer_indices": [1, 12],
        "pooling_method": "mean"
        },
    },
  "030": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.75,
        "freeze": True,
        "layer_indices": [1, 12],
        "pooling_method": "mean"
        },
    },
  "031": {
        'model': {
        "encoder_type": "ast",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.99,
        "freeze": True,
        "layer_indices": [6, 12],
        "pooling_method": "mean"
        },
    },
  "032": {
        'model': {
        "encoder_type": "wavlm",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.99,
        "freeze": True,
        "layer_indices": [1, 12],
        "pooling_method": "mean"
        },
    },
  "033": {
        "sample_rate": 24000,
        'model': {
          "encoder_type": "mert",
          "sae_method": "top-k",
          "sae_dim": 2048,
          "sparsity": 0.99,
          "freeze": True,
          "layer_indices": [4, 7],
          "pooling_method": "mean"
        },
        'data': {
          "train": {
            "target_sr": 24000,},
          "valid": {
            "target_sr": 24000,},
        },
    },
  "034": {
        'model': {
        "encoder_type": "hubert",
        "sae_method": "top-k",
        "sae_dim": 2048,
        "sparsity": 0.99,
        "freeze": True,
        "layer_indices": [3, 12],
        "pooling_method": "mean"
        },
    },
}

