# The Tensor Brain


## A Real-World Visual Challenge for Perception and Memory

We propose VRD-E and VRD-EX, two datasets inspired from
[Visual Relationship Detection with Language Priors](https://cs.stanford.edu/people/ranjaykrishna/vrd/)
[(Code)](https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection).
These datasets contains images and annotations from real-world scenarios. They aim testing a machine's perception performance in terms of object recognition,
relationship detection, and instance and semantic memory retrieval.
For a detailed description of the dataset, please refer to our paper
[The Tensor Brain: A Unified Theory of Perception, Memory and Semantic Decoding](https://arxiv.org/abs/2109.13392v2)

An example
![Alt text](example.png?raw=true "Title")

The VRD-E and VRD-EX datasets are augmented versions of the VRD dataset. In the VRD dataset, each visual entity
is labeled as belonging to one out of 100 classes.
Binary statements are annotated with 70 labels with 37,993 binary statements in total.
The training set contains 4000 images and the test set has 1000 images.
The training images contain overall 26,430 bounding boxes, thus on average 6.60 per image.


VRD-E (for VRD-Entity) is derived from VRD with additional labels for each visual entity. First, each entity in each
image obtains an individual entity index (or name).
The 26,430 bounding boxes in the training images describe 26,430 entity indices.
Second, based on concept hierarchies from WordNet, each entity is assigned exactly one basis class B-Class from VRD,
e.g., Dog, one parent class (or  P-Class),
e.g., Mammal, and one grandparent class (or  G-Class), e.g., LivingBeing. At any level, the default class Other is used
for entities that cannot be assigned to a WordNet concept.

In addition, a pretrained attribute classifier is used to label visual entities using
the attribute ontology. Each visual entity obtains exactly one color (including the color Other), and exactly one
activity attribute, e.g., a person can be standing or running. We also introduce the attribute labels Young and Old
which are randomly assigned, such that these can only be predicted for test entities that already occurred in training,
but not for novel entities.

Furthermore, the nonvisual, or hidden, attribute label Dangerous is assigned to all living things and Harmless to all
nonliving things. These labels are designed to be trained exclusively in semantic memory experience, but not in
perceptual training. We intend to use these labels to test the memory effect on label prediction for entities
and for attribute labels.

In summary, every visual entity receives one entity index and 8 attribute labels: entity index, 
B-Class, P-Class, G-Class (Living Being), Age (Young / Old), 
Dangerous / Harmless,  Color,  Activity.


The following table shows statistics of the datasets VRD-E and VRD-EX.

| Dataset | Training Images | Test Images | #BB Train | #Visual Entity | #Binary Statement | #Attribute/Entity Train |
|:-------:|:---------------:|:-----------:|:---------:|:--------------:|:-----------------:|:-----------------------:|
|  VRD-E  |      4000       |    1000     |   26430   |     26430      |       30355       |            8            |
| VRD-EX  |      7737       |    3753     |   50910   |     26430      |       50915       |            8            |


## Leaderboard
Here we list four tasks that are derived from our dataset. You can reproduce the numbers of our model 
using the following scripts. In addition, you can develop your own model and report the performance. To do this, 
you need to save the output in the format defined in the section Evaluation, and follow the instructions there to 
evaluate your model. **If you want to report the scores of your model on the leaderboard,
please don't hesitate to contact hang.li@siemens.com to add your result!**

Table 1 Performance of perception for unary statements (Accuracy)

| Dataset |   Model   | Entity | B-Class | P-Class | G-Class |  Y/O  | Color | Activity | Average |
|:-------:|:---------:|:------:|:-------:|:-------:|:-------:|:-----:|:-----:|:--------:|:-------:|
|  VRD-E  | BTN(ours) |   -    |  80.96  |  87.90  |  94.62  | 49.67 | 68.57 |  83.40   |  77.52  |
| VRD-EX  | BTN(ours) | 88.60  |  95.57  |  96.92  |  98.27  | 93.60 | 94.43 |  97.42   |  96.03  |

Table 2 Performance of perception for binary statements (Hits@k)

| Dataset  | Model  | @10  |  @1  |
|:--------:|:------:|:----:|:----:|
|  VRD-E   |BTN(ours)|92.96|48.43|
|VRD-EX|BTN(ours)|98.96|75.94|

Table 3 Performance of episodic (instance) memory retrieval for unary and binary labels (Accuracy, Hits@k)

|   Model   | s* @50 | B-Class | P-Class | G-Class |  Y/O  | Color | Activity | Binary labels @10 |Binary labels @1|
|:---------:|:------:|:---:|:-------:|:-------:|:-------:|:-----:|:-----:|:--------:|:-------:|
| BTN(ours) | 82.75  | 98.07| 99.51|99.99|94.55|92.93|97.61|97.08|60.90|

Table 4 Performance of semantic memory retrival for binary labels (Hits@k)

|   Model   |  @10  |  @1   |
|:---------:|:-----:|:-----:|
| BTN(ours) | 98.20 | 57.62 |
|  RESCAL   | 89.95 | 57.62 |



## Install Dependencies
The project is developed using python 3.9
```
pip3 install -r requirements.txt
```

## Getting Started
You can use this repository to reproduce the performance of our model which is shown in the above table.
The following instructions walk you through how to download and process the data, train our BTN model, 
and evaluate it on the test set.

You can also propose your own model and test it using our scripts. To do so, follow the instructions
in evaluation.

## Download Data
You can download the [annotations](https://doi.org/10.5281/zenodo.6249038) and
[images](https://zenodo.org/record/6249038/files/images.zip?download=1) and 
unzip it under the './data' directory.
The dataset has the following folders
```
/
/images/train
/images/test
/annotations/annotations_vrde_train.pkl # annoations for vrde training split
/annotations/annotations_vrde_test.pkl # annoations for vrde test split
/annotations/annotations_vrdex_train.pkl # annoations for vrdex training split
/annotations/annotations_vrdex_test.pkl # annoations for vrdex test split
/annotations/index_to_class.pkl # dict of class vocabulary, key(int) is the id, value(str) is the class label
/annotations/index_to_predicate.pkl # dict of predicate vocabulary, key(int) is the id, value(str) is the predicate label
```

The annotation is structured in this way
```
annotations_vrde=dict(
    scenes: dict( 
        # a dictionary of annotations indexed by each image
        scene_id: dict(
            filename: str, filename of the image
            entities: list of int, a list of entity ids from this image
            tuples: list of tuple, a list of (subject, object, predicate) triple annotations,
                specifically, each triple consists of (sub_box, obj_box, predicate, sub_label, obj_label)
        )
        scene_id: ...
    )
    
    entities: dict(
        # a dictionary of annotations indexed by each entity
        entity_id: dict(
            scene_id: the id of the image where the entity appears
            label: list of int, a list of unary labels for the entity
            bounding_box: bounding box of the entity
        )
        entity_id: ...
    )
)
```


## Data Preprocessing
The following script will process the data into a proper format for training a tensor brain model.
First, download the pretrained feature extractor 
[(pretrained-vgg19)](https://zenodo.org/record/6249038/files/vgg19_feature_extractor.pth?download=1) 
and save it under the ./data/ directory. Then run the following script.
It takes as input the downloaded images and annotations data, and outputs feature maps for the entities and images
into the folder defined in the argument. You need to run this script for each of the data split
(vrde_train, vrde_test, vrdex_train, vrdex_test)
```
cd data
python preprocessing.py --annotation_file annotations/annoations_vrde_train.pkl \
    --image_folder images/vrde/train/ \
    --output_folder vrde_reprs/train  \
    --extractor vgg19_feature_extractor.pth
```

## Training
Run the following code to train a BTN from scratch 
```
cd tensorbrain
python train.py perception_config
```
The input argument is the name of the 
_training_config_ defined in _config.py_. The options include 
[_perception_config_, _simple_perception_config_, _tkg_config_, _skg_config_] 
It saves the model.pth in the output directory. Training takes about 4 hours
for a perception model on a single GPU with 16GB RAM.


## Evaluation
If you want to evaluate our tensor brain model, run the following command
```
cd tensorbrain
python eval.py perception_test_config
```
The input argument is the name of the 
_test_config_ defined in _config.py_.  The options include
[_perception_test_config_, _simple_perception_test_config_, _tkg_test_config_, _skg_test_config_]. 
This script output the numbers that are required to fill in the above tables.


To evaluate your own proposed model, please prepare a _result.pkl_ file following the structure provided here.
```python
# Table 1 perception of unary labels
dict(
    entity_id1=dict(S_rank=int, S_cls=int, S_cat=int, S_liv=int, S_age=int, S_col=int, S_act=int, S_ins=int),
    entity_id1=dict(S_rank=int, S_cls=int, S_cat=int, S_liv=int, S_age=int, S_col=int, S_act=int, S_ins=int),
    ...
)

# Table 2 perception of unary labels
dict(
    tuple(scene_id1, subject_id1, object_id1)=dict(P_rank=int),
    tuple(scene_id2, subject_id2, object_id2)=dict(P_rank=int),
    ...
)
# X_rank stores the rank of the index of the ground truth label in the predictions sorted by the probabilities.
```
Optionally, we provide a function to create a template that you can use to easily fill in your
predictions. The following command outputs a pickle file following the structure as defined above.
You can keep the keys and replace the values with your own predictions and save it under _result.pkl_.
```python
cd data
python submission_template.py annotations/annotations_vrde_test.pkl
```

After you have prepared the _result.pkl_ file, run the following script to calculate those metrics
for the leaderboard. It will save these numbers in the _log.txt_ file. You can then contact us to add them
to the leaderboard.
```
cd tensorbrain
python eval_custom.py /path/to/result.pkl
```


## Citation
If you find our work helpful in your research, please cite us
```latex
@article{tresp2021tensor,
  title={The Tensor Brain: A Unified Theory of Perception, Memory and Semantic Decoding},
  author={Tresp, Volker and Sharifzadeh, Sahand and Li, Hang and Konopatzki, Dario and Ma, Yunpu},
  journal={arXiv preprint arXiv:2109.13392},
  year={2021}
}
```
