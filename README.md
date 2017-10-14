# spacenet-detection

## Problem
In this project we intend to build a neural network for detection and classiﬁcation of objects in high-resolution satellite imagery using SpaceNet dataset. The network will also be able to provide a description of the objects such as building, roads, aircrafts, etc.

## Dataset
We plan to use SpaceNet dataset available on AWS [4]. It is a corpus of satellite imagery and labeled training data developed by DigitalGlobe, Inc. SpaceNet includes thousands of square kilometers of high resolution imagery which includes 8-band multi-spectral data. SpaceNet has been explored to extract geometric features for roads, building footprints, and areas of interest (AOI) using satellite imagery. Currently, the AOIs are Rio De Janeiro, Paris, Las Vegas, Shanghai and Khartoum.

## Evaluation Metric
The ground truth for objects is given as polygons as shown in Fig.1. Thus we intend to use intersection over Union as the evaluation metric for object detection [5]. For classiﬁcation we will use the top 5 error metric.

## References
1. Redmon, Joseph, et al. ”You Only Look Once: Uniﬁed, Real-Time Object Detection [https://arxiv.org/pdf/1506.02640.pdf]
2. Ren, He, et al, ”Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks” [https://arxiv.org/pdf/1506.01497.pdf]
3. Etten, Adam Van. You Only Look Twice - Multi-Scale Object Detection in Satellite Imagery With Convolutional Neural...Medium, The DownLinQ, 7 Nov. 2016, medium.com/the-downlinq/you-only-look-twice-multi-scaleobject-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571.
4. SpaceNet on AWS.Amazon Web Services, Inc., aws.amazon.com/publicdatasets/spacenet/.
5. Hagerty, Patrick. The SpaceNet Metric The DownLinQ Medium. Medium, The DownLinQ, 10 Nov. 2016, medium.com/the-downlinq/the-spacenetmetric-612183cc2ddb.
