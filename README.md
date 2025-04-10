# Spiking Transformers Reproduced With [BrainCog](http://www.brain-cog.network/)

**This repo has beed archived, The new edition has been transfered to [here](https://github.com/Fancyssc/SpikingTransformerBenchmark)**
**Thanks for supporting us!**

Spiking Neural Networks (SNNs) are particularly efficient, energy-saving, and biologically interpretable. Models like Spikformer have introduced the attention mechanism from Artificial Neural Networks (ANNs) into SNNs, achieving success across a variety of datasets. The success of **Spiking Transformers** undoubtedly demonstrates the vast potential of SNNs in a wide range of tasks. We provide a customizable solution that allows you to easily [**define your own Spiking Transformer**](#design-your-own-spiking-transformer).



**[BrainCog](https://github.com/BrainCog-X/Brain-Cog)** is a programming framework developed and open-sourced by [Braincog-Lab](http://www.brain-cog.network/) for brain-inspired tasks. With Braincog, it is easy to reproduce and integrate different Spiking Transformer models, enabling developers and researchers to use these open-source models more conveniently and make personalized modifications.

### Implemented Models
Due to the unique characteristics of SNNs, models are typically trained on traditional datasets such as ImageNet and CIFAR10/100, as well as on neuromorphic datasets like **CIFAR10-DVS(DVSC10)**. These datasets are dimensionally incompatible with each other. Therefore, in this phase, we prioritize reproducing the part of these models that is adapted to DVS datasets.

<div align="center">

| Name  | Publication  | Step | Acc@1 on DVSC10 |
| :---: | :---: | :---: | :---: | 
| **Spikformer**| [ICLR 2023](https://openreview.net/forum?id=frE4fUwz_h) | 10/16| 78.9/80.9 |
| **Spikformer V2**|[Arxiv](https://arxiv.org/abs/2401.02020)| / | / | 
| **TIM**| [IJCAI 2024](https://www.ijcai.org/proceedings/2024/0347.pdf)| 10| 81.6 | 
| **Spike-driven Transformer**| [NuerIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html)| 16| 80.0 | 
|**QKFormer** | [NeurIPS 2024](https://arxiv.org/abs/2403.16552v2)| 16 | 84.0|
</div>
Model Results


## Models in comming soon
        
**More models will be updated soon......**


## Requirments

The version of timm should be exactly 0.5.4, unless the code will not work properly.
Other packages may possilbly be need for different models, please install them according to the error message.

```angular2html
- Braincog
- einops >= 0.4.1
- timm >= 0.5.4
...
```
For more information about the runtime environment, please refer to the `requirements.txt`file, or create a virtual environment directly based on this file.


## Details
### Datasets and Hyperparams
The `dataset.py` module in Braincog contains numerous predefined dataset APIs. Since the training scripts and data augmentation methods we use may differ from the original authors, some hyperparameter settings may not be entirely consistent with the original paper.

 However, based on our experiments, the final training results of the model are able to match or even surpass those reported in the original paper.

All self contructed module should follow the basic rules of **Braincog Framework**.

### Design Your Own Spiking Transformer
We offer a customizable approach, allowing you to define your own Spiking Transformer. Customizable options include Patch Embedding, Attention mechanisms, and even neuron types.
#### Neuron as a Param
The threshold, time constant, and other parameters of neurons can be input into the model. Of course, you can also define custom neurons and place them in `st_utils/node.py`.

#### Self-Designed Structure
In `st_utils/layer.py`, we provide commonly used classes for the reproduced models. By freely combining various types of Embedding, Attention, and MLP, you can build your own Transformer.


It is important to note that you only need to align the inputs and outputs of each component or override class methods when necessary.


#### Train Ur Own Model
After constructed, Ur model should be regstered and imported in `main.py`.  


### Training on CIFAR10-DVS
Default Hyperparam
```angular2html
-- num_heads 16
-- emb_dim 256
-- event-size 128
-- mlp ratio 4.
-- attn drop 0.
-- node LIFNode(tau=2.,threshold=0.5,act_func=SigmoidGate)
-- others follow Braincog Defaults...
```
```angular2html
conda activate example-environment

python main.py --dataset dvsc10 --epochs 500 --batch-size 16 --seed 42 --event-size 128 --model spikformer --device 0
```

### Training on ImageNet
Not supported yet. Comming Soon...



