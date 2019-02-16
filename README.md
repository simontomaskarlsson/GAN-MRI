

## Generative Adversarial Networks for Image-to-Image Translation on Multi-Contrast MR Images - A Comparison of CycleGAN and UNIT  

### [[Arxiv paper]](https://arxiv.org/abs/1806.07777)  


### Code usage  
1. Prepare your dataset under the directory 'data' in the CycleGAN or UNIT folder and
set dataset name to parameter 'image_folder' in model init function.
  * Directory structure on new dataset needed for training and testing:
    * data/Dataset-name/trainA
    * data/Dataset-name/trainB
    * data/Dataset-name/testA
    * data/Dataset-name/testB  

2. Train a model by:
```
python CycleGAN.py
```
or
```
python UNIT.py
```  

3. Generate synthetic images by following specifications under:
  * CycleGAN/generate_images/ReadMe.md
  * UNIT/generate_images/ReadMe.md

### Result GIFs - 304x256 pixel images  
**Left:** Input image. **Middle:** Synthetic images generated during training. **Right:** Ground truth.  
Histograms show pixel value distributions for synthetic images (blue) compared to ground truth (brown).<br/>(An updated image normalization, present in the current version of this repo, has fixed the intensity error seen in these results.) 


#### CycleGAN - T1 to T2
![](./ReadMe/gifs/CycleGAN_T2_hist.gif?)
---


#### CycleGAN - T2 to T1
![](./ReadMe/gifs/CycleGAN_T1_hist.gif)
---


#### UNIT - T1 to T2
![](./ReadMe/gifs/UNIT_T2_hist.gif)
---


#### UNIT - T2 to T1
![](./ReadMe/gifs/UNIT_T1_hist.gif)
---
