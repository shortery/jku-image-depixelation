# Image Depixelation Project

This is a school project from JKU Linz. It contains CNN trained to predict the original values of pixelated areas within images.
The input images are collected by students and resized to smaller sizes for feasibility. Training and testing sets are created by pixelating randomly sized and randomly positioned rectangle areas within images. CNN with skip connection SimpleCNN and CNN with residual blocks ResidualCNN are trained to restore pixelated areas.

## Structure

```
project
|- architectures.py
|    Classes for network architectures
|- datasets.py
|    Dataset class and dataset helper functions
|- inference.py
|    Function for computing predictions
|- main.py
|    Main file that runs training and inference
|- training.py
|   Function for training and evaluating
|- utils.py
|    Utility functions for conversion to grayscale and pixelating images
```

## Dependencies

Dependencies are specified in requirements.txt

## License

[MIT](https://choosealicense.com/licenses/mit/)
