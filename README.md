# deep_learning_study

This repository is used to storage the code typed by Sunlight-qwq.

I will study pytorch in this repository.

## Install anaconda


Using anaconda, which can be downloaded [here](https://www.anaconda.com/).

Then create a deep learning environment. First we register powershell as a anaconda prompt:

```powershell
conda init
```

After that, we can create a anaconda environment, with basic data-analyzing tools:

```powershell
conda create -n dl python=3.9 anaconda
conda activate dl
```

## Install pytorch (with CUDA)

Under environment `dl`, we can download [pytorch](https://pytorch.org/):

```powershell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

With the volume of pytorch very large (> 1 GB), downloading may cause a long time.

After that, we can start using of the pytorch.

The CUDA is included under anaconda env.

### Check the availability of CUDA

To check if CUDA can be used in `pytorch`, we can use the following code:

```python
>>> import torch
>>> torch.cuda.is_available()
```

If the answer is `True`, it means that CUDA can be successfully used.

## Use jupyter notebook

For `PyTorch` learning, jupyter notebooks are very useful.

Jupyter notebook has been installed during the creation of anaconda environment, where we can start jupyter easily:

```powershell
jupyter lab
```
