# Gradient-Effected Object Recognition Gauge for hive Entrances (GEORGE)

Team member: Kevin Hardin, 001056679

## Background

There is anecdotal evidence from beekeepers that the quantity of pollen being brought back to the hive by worker bees can inform the health of the hive. Many beekeepers believe that the presence and quantity of workers carrying pollen can indicate queenrightedness and healthy brood, as well as the availability of wild nectariferous flowers. The existence of this relationship between pollen and hive status has long been debated, however an official investigation has not yet been conducted.

It is unclear whether the mere presence of workers carrying pollen may be enough to indicate queenrightedness, or if it is the relative pollen flow that informs the overall health of the hive. Beekeepers can visually observe hive entrances; however, it is imprecise and impractical to manually count workers carrying pollen. It is also not possible to visually observe hives at all times of the day and season. The burgeoning use of neural networks for machine vision applications may provide a solution to this problem.

## Proposed method

The Gradient-Effected Object Recognition Gauge for hive Entrances (GEORGE) aims to observe the hive constantly during the day and count the number of workers carrying pollen (“GEORGE” is a ‘backronym’ named in honor of a beekeeping mentor of mine). A weatherproofed Raspberry Pi and PiCamera will be mounted to the front of the hive, looking downward to the entrance (see below).

![A close-up of a device Description automatically generated](fd7080f19a25d902f236c2240d22afff.jpeg)

Figure 1. GEORGE Prototype Hardware

A neural network will be trained to detect honeybees and identify whether they are carrying pollen. Transfer learning and fine-tuning of existing models will be employed to improve the training speed and accuracy of the network. RetinaNet[^1] is being eyed as a potential transfer learning source. The broader category of other Resnet50[^2] architectures and an extension of the Resnet50 architecture, called EfficientDet[^3], may also be good alternatives. RetinaNet is the favorite because of its speed (single-stage detection) and the relative availability of support resources online. Additionally, RetinaNet “heads” can be easily separated between the bounding box and categorizing heads, allowing for training of only resources that require it. Other Resnet50 networks and the EfficientDet model are available from online repositories such as Kaggle.com. All of these options are fine-tunable, which is imperative to improve accuracy. However, this capability may not be used in the end, depending on whether the datasets are large and diverse enough to prevent over-fitting.

[^1]: Lin, Tsung-Yi, et al. “Focal loss for dense object detection.” 2017 IEEE International Conference on Computer Vision (ICCV), 2017, https://doi.org/10.1109/iccv.2017.324.

[^2]: He, Kaiming, et al. “Deep residual learning for image recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, https://doi.org/10.1109/cvpr.2016.90.

[^3]: Tan, Mingxing, et al. “EfficientDet: Scalable and efficient object detection.” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, https://doi.org/10.1109/cvpr42600.2020.01079.

![A diagram of a network Description automatically generated](6a18b3974b28ef67b8f800937e389503.png)

Figure 2. Basic RetinaNet Block Architecture

There are several quality datasets available online for training neural networks on honeybee-related projects. The primary dataset that will be leveraged is one hosted on both Kaggle.com and tensorflow.org called “bee_dataset”[^4], which contains almost 7,500 labeled images of individual bees. Because one of the category labels is “varroa mite”, this dataset will provide the added benefit of allowing GEORGE to identify honeybees afflicted with varroa mites: a common cause of colony collapse disorder. The dataset will be heavily augmented: an attempt will be made to “extract” honeybee subjects from the solid backgrounds of the images, then placed in random orientations, positions, scales, and skew levels on a larger 640x640 canvas to further improve model accuracy and prevent over-fitting. This will also enable the artificial “expansion” of the dataset to an arbitrary number of images by randomly augmenting and saving images such that all are unique.

[^4]: https://www.tensorflow.org/datasets/catalog/bee_dataset

Final, in situ testing of GEORGE will unfortunately need to wait until next spring when honeybees reemerge from their winter hibernation. However, the network will be trained and the final Tensorflow model will be converted into at Tensorflow lite (tflite) model for loading onto the Raspberry Pi. The tflite format should allow for the model to run more efficiently on the limited hardware of the Raspberry Pi.
