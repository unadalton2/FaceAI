# FaceAI
A simple autoencoder capable of manipulating faces. Using tensorflow and keras for the creation and training, FaceAI was able to compress imgs from their raw rbg format to 12.5% the size of the origanal. FaceAI maps single values within the compressed img to larger more general shapes and/or colors in the final img. The compression allows the smooth interpolation between faces. Meaning shapes and/or colors smoothly change positions and form. 

# Training
FaceAI was trained using tensorflow and keras. The model was trained for 300 epochs on about 800 datapoints. For training data, the model uses 809 yearbook photos(definetly overfitting). The model’s power comes from overfitting. Overfitting allows the model to reach higher compression rates closing the gap between distant faces.  
# Iterpolation Example
![ImgA](https://github.com/unadalton2/FaceAI/blob/main/interpolation/interpolation50.jpg) ![Gif2](https://github.com/unadalton2/FaceAI/blob/main/Gif2.gif) ![ImgB](https://github.com/unadalton2/FaceAI/blob/main/interpolation/interpolation0.jpg)
