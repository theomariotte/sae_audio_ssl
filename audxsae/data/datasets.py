from audxsae.data.sound import ESC_50, UrbanSound8k
from audxsae.data.music import GTZAN, VocalSet
from audxsae.data.speech import TimitDataset, CommonVoiceDataset

def select_dataset(dataset_name: str,
                   **data_kw):
    """Wrapper to select a torch dataset

    Args:
        dataset_name (str): name of the dataset to select
        data_kw (dict): parameters of the dataset
    Returns:
        torch.utils.data.Dataset: pytorch dataset that can be used in a standard torch dataloader.
    """
    if dataset_name.upper() == "ESC50":
        dataset = ESC_50(**data_kw)
    elif dataset_name.upper() == "URBANSOUND8K":
        dataset = UrbanSound8k(**data_kw)
    elif dataset_name.upper() == "GTZAN":
        dataset = GTZAN(**data_kw)
    elif dataset_name.upper() == "TIMIT":
        dataset = TimitDataset(**data_kw)
    elif dataset_name.upper() == "COMMONVOICE":
        dataset = CommonVoiceDataset(**data_kw)
    elif dataset_name.upper() == "VOCALSET":
        dataset = VocalSet(**data_kw)

    return dataset