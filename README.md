

## Generative Adversarial Networks for Image-to-Image Translation on Multi-Contrast MR Images - A Comparison of CycleGAN and UNIT  

### Article  
[Frontiers in Neuroinformatics - article](https://www.frontiersin.org/journals/neuroinformatics) 
&nbsp;
&nbsp;
&nbsp;

### Code usage  
1. Prepare your dataset under the directory 'data' in the CycleGAN or UNIT folder
  * Directory structure on new dataset needed for training and testing:
    * data/Dataset-name/trainA
    * data/Dataset-name/trainB
    * data/Dataset-name/testA
    * data/Dataset-name/testB  
    &nbsp;
2. Train a model by:
```
python CycleGAN.py
```
or
```
python UNIT.py
```  
&nbsp;
3. Generate synthetic images by following specifications under:
  * CycleGAN/generate_images/ReadMe.rtf
  * UNIT/generate_images/ReadMe.rtf  
  &nbsp;
  &nbsp;
  &nbsp;

### Result GIFs - 304x256 pixel images  
**Left:** Input image. **Middle:** Synthetic images generated during training. **Right:** Ground truth.  
Histograms show pixel value distributions for synthetic images (blue) compared to ground truth (brown).  
&nbsp;
&nbsp;
&nbsp;

#### CycleGAN - T1 to T2
![](./ReadMe/gifs/CycleGAN_T2_hist.gif?)
---
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

#### CycleGAN - T2 to T1
![](./ReadMe/gifs/CycleGAN_T1_hist.gif)
---
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

#### UNIT - T1 to T2
![](./ReadMe/gifs/UNIT_T2_hist.gif)
---
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

#### UNIT - T2 to T1
![](./ReadMe/gifs/UNIT_T1_hist.gif)
---
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
