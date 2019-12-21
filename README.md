## MSN: Morphing and Sampling Network for Dense Point Cloud Completion

[[paper]](http://cseweb.ucsd.edu/~mil070/projects/AAAI2020/paper.pdf) [[data]](https://drive.google.com/drive/folders/1X143kUwtRtoPFxNRvUk9LuPlsf1lLKI7?usp=sharing)

MSN is a learning-based shape completion method which can preserve the known structures and generate dense and evenly distributed point clouds. See our AAAI 2020 [paper](http://cseweb.ucsd.edu/~mil070/projects/AAAI2020/paper.pdf) for more details.

In this project, we also provide an implementation for the Earth Mover's Distance (EMD) of point clouds, which is based on the auction algorithm and only needs $O(n)$ memory.

![](/teaser.png)
*with 32,768 points after completion*


### Usage

#### 1) Envrionment & prerequisites

- Pytorch 1.2.0
- CUDA 10.0
- Python 3.7
- [Visdom](https://github.com/facebookresearch/visdom)
- [Open3D](http://www.open3d.org/docs/release/index.html#python-api-index)

#### 2) Compile

Compile our extension modules:  

    cd emd
    python3 setup.py install
    cd expansion_penalty
    python3 setup.py install
    cd MDS
    python3 setup.py install

#### 3) Download data and trained models

Download the data and trained models from [here](https://drive.google.com/drive/folders/1X143kUwtRtoPFxNRvUk9LuPlsf1lLKI7?usp=sharing).  We don't provide the partial point clouds of the training set due to the large size. If you want to train the model, you can generate them with the [code](https://github.com/wentaoyuan/pcn/tree/master/render) and [ShapeNetCore.v1](https://shapenet.org/). We generate 50 partial point clouds for each CAD model.

#### 4) Train or validate

Run `python3 val.py` to validate the model or `python3 train.py` to train the model from scratch.

### EMD

We provide an EMD implementation for point cloud comparison, which only needs $O(n)$ memory and thus enables dense point clouds  (with 10,000 points or over) and large batch size. It is based on an approximated algorithm (auction algorithm) and cannot guarantee a (but near) bijection assignment. It employs a parameter $\epsilon$ to balance the error rate and the speed of convergence. Smaller $\epsilon$ achieves more accurate results, but needs a longer time for convergence. The time complexity is $O(n^2k)$, where $k$ is the number of iterations. We set a $\epsilon = 0.005, k = 50$ during training and a $\epsilon = 0.002, k = 10000$ during testing. Please refer to`emd/README.md` for more details.

### Citation

If you find our work useful for your research, please cite:
```
@article{liu2019morphing,
  title={Morphing and Sampling Network for Dense Point Cloud Completion},
  author={Liu, Minghua and Sheng, Lu and Yang, Sheng and Shao, Jing and Hu, Shi-Min},
  journal={arXiv preprint arXiv:1912.00280},
  year={2019}
}
```

### License

This project Code is released under the Apache License 2.0 (refer to the LICENSE file for details).