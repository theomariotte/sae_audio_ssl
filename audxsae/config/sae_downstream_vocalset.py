import os
import random
import socket
import string

cluster = socket.gethostname()
slurm = "SLURM_JOB_ID" in os.environ

EXP_ROOT = os.environ["EXP_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]

CKPT_ROOT=os.path.join(EXP_ROOT, "train/SAE/ssl")

if slurm:
  job_id = os.environ["SLURM_JOB_ID"]
else:
  job_id = "".join(random.choices(string.ascii_letters + string.digits, k=8))

common_parameters = {
    #'exp_dir': os.path.join(EXP_ROOT,'train/SAE/ssl'),
    'exp_dir': os.path.join(EXP_ROOT,'train/SAE/sae_downstream'),
    'sample_rate': 16000,
    'num_classes': 10,
    'optim': {
        'epochs': 120,
        'batch_size': 32,  
        'learning_rate': 0.001,
        'patience_early_stop': 200,
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
  "0016": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "001_ast_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "00112": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "001_ast_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "0026": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "002_ast_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "00212": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "002_ast_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "0036": { 
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "003_ast_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "00312": { 
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "003_ast_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "0046": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "004_ast_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "00412": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "004_ast_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "0056": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "005_ast_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "00512": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "005_ast_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "0062": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "006_wavlm_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "00612": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "006_wavlm_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "0072": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "007_wavlm_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "00712": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "007_wavlm_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "0082": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "008_wavlm_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "00812": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "008_wavlm_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "0092": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "009_wavlm_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "00912": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "009_wavlm_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "0102": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "010_wavlm_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "01012": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "010_wavlm_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "0114": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "011_mert_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "0117": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "011_mert_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "0124": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "012_mert_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "0127": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "012_mert_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "0134": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "013_mert_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "0137": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "013_mert_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "0144": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "014_mert_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "0147": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "014_mert_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "0154": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "015_mert_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "0157": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "015_mert_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "0164": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "016_hubert_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "0174": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "017_hubert_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "0184": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "018_hubert_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "0194": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "019_hubert_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "0204": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "020_hubert_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "0213": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "021_hubert_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "02112": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "021_hubert_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "0222": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "022_hubert_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "02212": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "022_hubert_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "0233": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "023_hubert_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "02312": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "023_hubert_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "0243": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "024_hubert_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "02412": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "024_hubert_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "0253": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "025_hubert_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "02512": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "025_hubert_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "0261": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "026_wavlm_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "02612": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "026_wavlm_vocalset_SAE_top-k_2048_95_seed_42/best_model.pth"),
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
  "0271": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "027_wavlm_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "02712": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "027_wavlm_vocalset_SAE_top-k_2048_90_seed_42/best_model.pth"),
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
  "0281": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "028_wavlm_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "02812": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "028_wavlm_vocalset_SAE_top-k_2048_85_seed_42/best_model.pth"),
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
  "0291": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "029_wavlm_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "02912": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "029_wavlm_vocalset_SAE_top-k_2048_80_seed_42/best_model.pth"),
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
  "0301": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "030_wavlm_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
  "03012": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "030_wavlm_vocalset_SAE_top-k_2048_75_seed_42/best_model.pth"),
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
    # AST 99%
  "0316": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "031_ast_vocalset_SAE_top-k_2048_99_seed_42/best_model.pth"),
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
  "03112": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "031_ast_vocalset_SAE_top-k_2048_99_seed_42/best_model.pth"),
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
    # WAVLM 99%
  "0321": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "032_wavlm_vocalset_SAE_top-k_2048_99_seed_42/best_model.pth"),
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
  "03212": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "032_wavlm_vocalset_SAE_top-k_2048_99_seed_42/best_model.pth"),
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
    # HUBERT 99%
  "0343": {
        "sae_index": 0,
        "checkpoint": os.path.join(CKPT_ROOT, "034_hubert_vocalset_SAE_top-k_2048_99_seed_42/best_model.pth"),
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
  "03412": {
        "sae_index": 1,
        "checkpoint": os.path.join(CKPT_ROOT, "034_hubert_vocalset_SAE_top-k_2048_99_seed_42/best_model.pth"),
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
    # MERT 99%
  "0334": {
        'sae_index': 0,
        "checkpoint": os.path.join(CKPT_ROOT, "033_mert_vocalset_SAE_top-k_2048_99_seed_42/best_model.pth"),
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
  "0337": {
        'sae_index': 1,
        "checkpoint": os.path.join(CKPT_ROOT, "033_mert_vocalset_SAE_top-k_2048_99_seed_42/best_model.pth"),
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
}

