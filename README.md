# Grain Detection

PyTorch based framework for training [ViT](https://arxiv.org/abs/2010.11929)-based models for semantic segmentation of grinding tool images.
The purpose of the predicted segmentation masks is to denote which pixels refer to a grain and which to the binder.

The models which can be used are [SEgmentation TRansformer (SETR)](https://arxiv.org/abs/2012.15840) and [TransUNet](https://arxiv.org/abs/2102.04306).
## Example Results
&nbsp;

<img src='figures\predictions.png' align="left">  

&nbsp;
## Datasets
The base dataset consists of 8 training samples and 1 test sample of size 1270 x 10960 px, which where regularly cropped into patches of 256 x 256 px.

To improve the segmentation accuracy, the dataset was augmented using images which where generated by a [custom diffusion model (insert link)](). For sampling of the feature images and matching segmentation masks two approaches were developed. For more information please visit the corresponding repo.

*The models were also trained on another dataset containing RGB-images of worn 
drilling tools. For legal reasons the data cannot be published*

## Scores

#### Base 
| Model          | IoU | 
| --------------- | --------- | 
| TransUNet | 94,31 %
| SETR-MLA | 90,43 %
| SETR-PUP | 88,52 %

#### Augmented (TransUNet)
| Dataset          | IoU | 
| --------------- | --------- | 
| Base | 94,31 %
| Augmented 1 step | 94,72 %
| Augmented 2 steps | 94,79 %
