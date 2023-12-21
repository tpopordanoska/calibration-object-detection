# Calibration in Object Detection

This is the official code repository for ["Beyond Classification: Definition and Density-based Estimation of Calibration in Object Detection"](https://arxiv.org/abs/2312.06645), published at WACV 2024.

In this work, we adapt the definition of classification calibration error to handle the nuances associated with object
detection, and predictions in structured output spaces more generally. Furthermore, we propose a consistent and 
differentiable estimator of the detection calibration error, utilizing kernel density estimation. Our experiments 
demonstrate the effectiveness of our estimator against competing train-time and post-hoc calibration methods, while 
maintaining similar detection performance.

## Getting started

Clone the project, create a conda environment and install the dependencies:
```
git clone https://github.com/tpopordanoska/calibration-object-detection.git
conda env create -f env.yaml
conda activate cal-od
```

Download [COCO](https://cocodataset.org/#home), [Cityscapes](https://www.cityscapes-dataset.com/), 
[Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) and 
[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). Convert the annotations of Cityscapes and Pascal VOC
to COCO-format (e.g. using [Cityscapes-to-COCO](https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion) and 
[VOC-to-COCO](https://github.com/yukkyo/voc2coco/blob/master/voc2coco.py)). Split the original train set into new train 
and validation sets with 90:10 ratio with seeds {5, 10, 21}, using `split_val.py`. The new annotation files should be 
placed in a `splits` subfolder and named `instances_{dataset}_{set}_{seed}.json`, where dataset can be 'coco', cityscapes',
or 'voc', and set can be 'train' or 'val'. The test annotation file should be named `instances_{dataset}_test.json`.

To train the object detectors with the calibration loss, use the `train_net_cal.py` and `train_net_lazy_cal.py` scripts.

## Reference
If you found this work or code useful, please cite:

```
@misc{popordanoska2023beyond,
      title={Beyond Classification: Definition and Density-based Estimation of Calibration in Object Detection}, 
      author={Teodora Popordanoska and Aleksei Tiulpin and Matthew B. Blaschko},
      year={2023},
      eprint={2312.06645},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).