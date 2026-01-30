# SemiEarth

The code implementation of the SemiEarth.
![semiearth](docs/semiearth.png)

### Environment
---
```
pip install -r requirements.txt
```

### Dataset
---
Download the processed [Datasets](https://pan.baidu.com/s/1DD_RUXiIcwBsJyCtPYkzRA). Code = ``yny6``.
Your file structure will be like:
```
├── [Your Dataset Path]
    ├── JPEGImages
        ├── img_001.jpg
        ├── img_002.jpg
        └── ...
    ├── SegmentationClass
        ├── label_001.png
        ├── label_002.png
        └── ...
```

### Pretrained 
---
Download the pretrained [Checkpoints](https://github.com/facebookresearch/dinov2).
Your file structure will be like:
```
├── pretrained
    └── dinov2_small.pth
    └── ...
```

### Training
---
```
sh scripts/train_love_1_100.sh 8 29501 29502 29503 29504 29505 29506 29507 29508
```

### Determined AI
---
```
cd det_create
det e create det_4090.yaml .
```

### Citation
If you find it useful, please consider citing:
```

```