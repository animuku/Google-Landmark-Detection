In this project, I perform experiments on the Google Landmark Dataset(https://www.kaggle.com/c/landmark-recognition-challenge) using various Deep Learning algorithms for image recognition. 

All experiments are performed without GPU acceleration. I have optimized my code to work around this limitation but I can only work on approximately 300 classes. Nevertheless, this is just me playing around with these algorithms and learning more about them.

All the metrics for each of the algorithms will be added once I work on them.

Algorithms used:
1. ResNeXt- 25 classes,5 epochs,loss=1.7205, training_accuracy=0.8182
2. VGG16- 25 classes, 5 epochs, loss=2.4551, training_accuracy=0.2163(also encountered a memory allocation problem)
3. Squeeze and Excitation Net- 25 classes, 5 epochs, loss=0.1822, training_accuracy=0.9408
4. Inception v3
5. AlexNet- 25 classes,5 epochs, loss=2.4608,training_accuracy=0.2295
