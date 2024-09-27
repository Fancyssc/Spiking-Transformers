# Spiking Transformers Reproduced With Braincog
Here is the current Spiking Transformer code reproduced using [BrainCog](http://www.brain-cog.network/). Welcome to follow the work of BrainCog and utilize the [BrainCog framework](https://github.com/BrainCog-X/Brain-Cog) to create relevant brain-inspired AI endeavors. The works implemented here will also be merged into BrainCog Repo.

### Models
**Spikformer(ICLR 2023)**
[paper link](https://openreview.net/forum?id=frE4fUwz_h) 

<div style="text-align: center;">
    <img src="/img/spikformer.png"  style="width: 60%;">
</div>

**Spike-driven Transformer(Nips 2023)**
[paper link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html)

<div style="text-align: center;">
    <img src="/img/sdv1.png"  style="width: 60%;">
</div>


**Spike-driven Transformer V2(ICLR 2024)**
[paper link](https://openreview.net/forum?id=1SIBN5Xyw7)

<div style="text-align: center;">
    <img src="/img/sdv2.png"  style="width: 60%;">
</div>

**TIM(IJCAI 2024)**
[paper link](https://www.ijcai.org/proceedings/2024/0347.pdf)

The code of **_TIM: An Efficient Temporal Interaction Module for Spiking Transformer_** is originally written by [Braincog](http://www.brain-cog.network/). The official code of TIM can be downloaded in [code link](https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/TIM).
<div style="text-align: center;">
    <img src="/img/TIM.png"  style="width: 60%;">
</div>

## Models in comming soon
**SpikingResFormer**
[Shi, X., Hao, Z., & Yu, Z. (2024). SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks. arXiv preprint arXiv:2403.14302.](https://arxiv.org/abs/2403.14302)

**SGLFormer(Frontiers in Neuroscience)**
[Zhang, H., Zhou, C., Yu, L., Huang, L., Ma, Z., Fan, X., ... & Tian, Y. (2024). SGLFormer: Spiking Global-Local-Fusion Transformer with High Performance. Frontiers in Neuroscience, 18, 1371290.](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1371290/full)

**QKFormer(CVPR2024)**
[Zhou, C., Zhang, H., Zhou, Z., Yu, L., Huang, L., Fan, X., ... & Tian, Y. (2024). QKFormer: Hierarchical Spiking Transformer using QK Attention. arXiv preprint arXiv:2403.16552.](https://arxiv.org/abs/2403.16552)


## Requirments
- Braincog
- einops >= 0.4.1
- timm >= 0.5.4

## Training Examples
Please notice that part of models may not support all the datasets mentioned below. 


### Training on CIFAR10-DVS

```angular2html
python main.py --dataset dvsc10 --epochs 500 --batch-size 16 --seed 42 --event-size 64 --model spikformer_dvs
```

### Training on ImageNet
```angular2html
python main.py --dataset imnet --epochs 500 --batch-size 16 --seed 42 --model spikformer
```
