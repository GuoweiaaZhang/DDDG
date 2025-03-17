import torch
import torch.utils.data as Data
from typing import List, Tuple, Union

def construct_loader(
    data: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]], 
    batch_size: int, 
    uncom: bool = False
) -> Union[Data.DataLoader, List[Data.DataLoader]]:
    """
    构建PyTorch数据加载器。

    参数:
        data: 包含张量的列表或元组
        batch_size: 每个批次的样本数
        uncom: 是否使用特殊的未压缩数据格式，默认为False

    返回:
        单个DataLoader或DataLoader列表
    """
    
    # 处理未压缩数据的特殊情况
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

    # 根据输入数据的维度构建不同的数据加载器
    if len(data) == 2:
        dataset = Data.TensorDataset(data[0], data[1])
        return Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    
    elif len(data) == 6:
        dataset = Data.TensorDataset(*data)  # 使用解包操作符简化代码
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
        for i in range(len(data[0])):  # 修正了原代码中的range(data)错误
            dataset = Data.TensorDataset(data[0][i], data[1][i], data[2][i])
            loader = Data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True
            )
            loaders.append(loader)
        return loaders

    raise ValueError(f"不支持的数据维度: {len(data)}")