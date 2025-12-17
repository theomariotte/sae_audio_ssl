
from IPython.display import display
from IPython.display import HTML
import matplotlib
import random
import numpy as np
from IPython.display import display, HTML
import pandas as pd

pd.set_option('display.max_columns', None)      # Show all columns
pd.set_option('display.max_colwidth', None)     # Show full content of each column
pd.set_option('display.width', None)            # Don't wrap the display
pd.set_option('display.max_rows', None)

def display_colored_top_features(df_res, layer_index, model_type):
    """
    Display a dataframe with top feature labels colorized for a given model type and layer index.

    Args:
        df_res (pd.DataFrame): The full results dataframe.
        layer_index (int): The layer index to filter.
        model_type (str): The model type to filter.
    """
    df_filt = df_res[["model_type", "sparsity", "layer_index", "top_feat_labels"]].loc[
        (df_res["model_type"] == model_type) & (df_res["layer_index"] == layer_index)
    ]

    # Get the full list of feature labels
    all_labels = df_res["feat_labels"].iloc[0]

    # Assign a unique color to each label
    random.seed(42)
    n_labels = len(all_labels)
    colors = matplotlib.cm.get_cmap('tab20', n_labels)
    label2color = {label: matplotlib.colors.rgb2hex(colors(i)) for i, label in enumerate(all_labels)}

    def colorize_labels(label_list):
        if isinstance(label_list, (list, tuple, np.ndarray)):
            return " ".join(
                f'<span style="color:{label2color.get(str(label), "#000")}; font-weight:bold">{label}</span>'
                for label in label_list
            )
        else:
            return f'<span style="color:{label2color.get(str(label_list), "#000")}; font-weight:bold">{label_list}</span>'

    df_filt_disp = df_filt.copy()
    df_filt_disp["top_feat_labels"] = df_filt_disp["top_feat_labels"].apply(colorize_labels)

    display(HTML(df_filt_disp.to_html(escape=False)))

def group_features(feat_list, num_top_feat=5):
    """Group acoustic features considering categories

    Args:
        feat_list (list): list of features to be sorted by category
        num_top_feat (int, optional): Number of features to consider in the sorting. Defaults to 5.

    Returns:
        _type_: _description_
    """
    grouped_feats = {
        "pitch": [],
        "formants": [],
        "quality": [],
        "mfcc": [],
        "spectral": [],
        "rythm": [],
        "loudness": []
    }
    feat_list = feat_list if num_top_feat >= len(feat_list) else feat_list[:num_top_feat]
    for label in feat_list:
        if 'F1' in label or 'F2' in label or 'F3' in label:
            grouped_feats["formants"].append(label)
        elif 'F0' in label:
            grouped_feats["pitch"].append(label)
        elif 'shimmer' in label.lower() or 'jitter' in label.lower() or 'hnr' in label.lower():
            grouped_feats["quality"].append(label)
        elif 'loudness' in label.lower() or 'equivalentsoundlevel' in label.lower():
            grouped_feats["loudness"].append(label)
        elif 'voiced' in label.lower():
            grouped_feats["rythm"].append(label)
        elif 'slope' in label.lower() or 'spectral' in label.lower() or 'alpharatio' in label.lower() or 'hammar' in label.lower():
            grouped_feats["spectral"].append(label)
        elif 'mfcc' in label.lower():
            grouped_feats["mfcc"].append(label)
    
    occurences = {}
    for k,v in grouped_feats.items():
        occurences[k] = len(v)

    return grouped_feats, occurences