
<!-- ABOUT THE PROJECT -->
## Self-supervised 3D anatomy segmentation using self-distilled masked image transformer (SMIT)
### Jue Jiang, Neelam Tyagi, Kathryn Tringale, Christopher Crane, and Harini Veeraraghavan

 Self-supervised 3D anatomy segmentation using self-distilled masked image transformer (SMIT) is a image transformer model constructed using the SWIN transformer backbone. SMIT is pretrained using a large number of unlabeled 3D computed tomography (CT) image sets sourced from insitutional and public data repositories from a variety of diseases including the lung cancer, COVID-19, head and neck cancer, kidney, liver, and other abdominal cancers. SMIT is pretrained using a masked image prediction or dense pixel regression task, as well as self-distillation tasks using patch token and global token distillation. Following pretraining, the network can easily be fine tuned and applied to generating segmentation in other imaging modalities never used in 
pretraining such as magnetic resonance imaging (MRI). Additional details of this work are in: 

https://arxiv.org/abs/2205.10342


 <div align="center">
  <img width="90%" alt="SMIT framework" src="figures/fig.png">
</div>


 ## Jue Jiang, ... Harini Veeraraghavan

This is the official source code for the MICCAI 2022 paper [SMIT](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_53)


<!-- GETTING STARTED -->
## Getting Started


### Install
pip install requirements.txt


<!-- USAGE EXAMPLES -->
## Usage

## for self supervised pretraining
python train_self_supervised.py


## for fine tuning
python fine_tuning_swin_3D.py  --resume_ckpt

## pretrained weight
We offered the pre-trained weight with imagee patch size of 96x96x96, depth= (2, 2, 4, 2), head= (4, 4, 4, 4), window size= (4,4,4).

<!-- ACKNOWLEDGMENTS -->
## Our code refered the following implementation

* [UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR)
* [ibot](https://github.com/bytedance/ibot)

<!-- Citation -->

## Citing SMIT
If you find this repository useful, please consider giving a star :star: and citation:
```
@InProceedings{juejsmit,
  title={Self-supervised 3D Anatomy Segmentation Using Self-distilled Masked Image Transformer (SMIT)},
  author={Jiang, Jue and Tyagi, Neelam and Tringale, Kathryn and Crane, Christopher and Veeraraghavan, Harini},
  journal={International Conference Medical Image Computing and Computer Assisted Intervention, 2022},
  pages={556--566},
  DOI={DOI: 10.1007/978-3-031-16440-8_53},
  year={2022}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



