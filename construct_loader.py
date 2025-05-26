import torch
import torch.utils.data as Data
from typing import List, Tuple, Union

def construct_loader(
    data: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]], 
    batch_size: int, 
    uncom: bool = False
) -> Union[Data.DataLoader, List[Data.DataLoader]]:
    """
    Building a PyTorch Data Loader。

    Parameters.
        data: list or tuple containing the tensor
        batch_size: number of samples in each batch
        uncom: whether to use special uncompressed data format, default is False

    Returns.
        A single DataLoader or a list of DataLoaders.
    """
    if uncom:
        loaders = []
        for i in range(len(data[0])):
            dataset = Data.TensorDataset(data[0][i], data[1][i], data[2][i], data[3])
            loader = Data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True
            )
            loaders.append(loader)
        return loaders

    if len(data) == 2:
        dataset = Data.TensorDataset(data[0], data[1])
        return Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    
    elif len(data) == 6:
        dataset = Data.TensorDataset(*data) 
        return Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
    
    elif len(data) == 8:
        dataset = Data.TensorDataset(*data)
        return Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
    
    elif len(data) == 3:
        loaders = []
        for i in range(len(data[0])):  
            dataset = Data.TensorDataset(data[0][i], data[1][i], data[2][i])
            loader = Data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True
            )
            loaders.append(loader)
        return loaders

    raise ValueError(f"不支持的数据维度: {len(data)}")
