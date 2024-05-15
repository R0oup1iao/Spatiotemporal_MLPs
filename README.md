# Spatiotemporal_MLPs

交通综合实验

## Step0 创建虚拟环境

```cmd
conda create -n myenv python=3.10
conda create --prefix env python=3.10
python -m venv venv
```

## Step1 安装依赖

### 安装torch

参考<https://pytorch.org/>
例如：

```cmd
pip install torch torchvision torchaudio
```

### 安装其他依赖

```cmd
pip install lightning einops pandas scikit-learn easydict
```

## Step2 基本的MLP训练

更改config.py中的如下参数为False

```python
"if_node": False,
"if_T_i_D": False,
"if_D_i_W": False,
```

之后运行:

```cmd
python train.py
```

## Step3 Spatiotemporal-MLP训练

更改config.py中的如下参数为True

```python
"if_node": True,
"if_T_i_D": True,
"if_D_i_W": True,
```

之后运行:

```cmd
python train.py
```

## Results

| Test metric | MAE         |
| :---        |    :----    |
| **STMLP**      | **1.7959668636322021**     |
| w/o s_emb   | 3.184992790222168       |
| w/o t_emb   | 2.1182820796966553       |
| w/o s_emb&t_emb   | 3.4272847175598145       |
