# Strip Pooling: Rethinking Spatial Pooling for Scene Parsing (Coming soon!!!)

This repository is a PyTorch implementation for our [CVPR2020 paper](https://arxiv.org/pdf/2003.13328.pdf).

The results reported in our paper are originally based on [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) but the environment settings are complicated. To ease use, we reimplement our work based on [semseg](https://github.com/hszhao/semseg).

### Strip Pooling

![An efficient way to use strip pooling](strip.png)


### Usage

Before training your own models, we recommend you to refer to the usage intruction described [here](https://github.com/hszhao/semseg). Then, you need to update the dataset paths in the configuration files.

Four GPUs with at least 11G memory on each are required for synchronized training.

### Citation

You may want to cite:

```
@inproceedings{hou2020strip,
  title={{Strip Pooling}: Rethinking Spatial Pooling for Scene Parsing},
  author={Hou, Qibin and Zhang, Li and Cheng, Ming-Ming and Feng, Jiashi},
  booktitle={CVPR},
  year={2020}
}
@misc{semseg2019,
  author={Zhao, Hengshuang},
  title={semseg},
  howpublished={\url{https://github.com/hszhao/semseg}},
  year={2019}
}
```
