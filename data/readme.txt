This data is released along with the following paper:
[1] "Visual Relationship Detection with Language Priors", 
Cewu Lu*, Ranjay Krishna*, Michael Bernstein, Li Fei-Fei, European Conference on Computer Vision, 
(ECCV 2016), 2016 (oral). (* = indicates equal contribution)

CITATION
===============================================================================================
If you use this data, please use the following citation:
@InProceedings{lu2016visual,
  title = {Visual Relationship Detection with Language Priors},
  author = {Lu, Cewu and Krishna, Ranjay and Bernstein, Michael and Fei-Fei, Li},
  booktitle = {European Conference on Computer Vision},
  year = {2016},
}

USING THE DATA
===============================================================================================
1. Images:
   The images used can be downloaded from:
   http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
   We follow the same training and test splits.

2. Annotation: 
   Annotations_train.json and annotations_test.json contains our annotations for the 100 object categories and 70 predicates.
   They are in the following format:

    [...
    FILENAME: [...
        {'subject': {'category': CATEGORY_ID, 'bbox': [YMIN, YMAX, XMIN, XMAX]},
         'predicate': PREDICATE_ID,
         'object': {'category': CATEGORY_ID, 'bbox': [YMIN, YMAX, XMIN, XMAX]},
        }
        ...]
    ...]
    
    
    For example:
    
    [...
    '3008054318_8dfe57c0d0_b.jpg':
        {'subject': {'category': 2, 'bbox': [120, 635, 109, 924]},
         'predicate': 3,
         'object': {'category': 1, 'bbox': [463, 762, 5, 1019]},
        }
        ...]
    ...]

   Our bounding box format is [ymin, ymax, xmin, xmax]. 
   PREDICATE_ID and OBJECT_ID are indices to the arrays in 'objects.json' and 'predicates.json', explained below.

3. Object and Predicate Categories
    The object and predicate categories are listed in 'objects.json' and 'predicates.json'

    Let's print out all the object categories:
    >> objects = json.load(open('objects.json'))
    >> for i in range(100):
    ...  print "%dth object category: %s" % (i, objects[i])
    1th object category: person
    2th object category: sky
    3th object category: building
    4th object category: truck
    5th object category: bus
    6th object category: table
    7th object category: shirt
    8th object category: chair
    9th object category: car
    10th object category: train
    11th object category: glasses
    12th object category: tree
    13th object category: boat
    14th object category: hat
    15th object category: trees
    16th object category: grass
    17th object category: pants
    18th object category: road
    19th object category: motorcycle
    20th object category: jacket
    21th object category: monitor
    22th object category: wheel
    23th object category: umbrella
    24th object category: plate
    25th object category: bike
    26th object category: clock
    27th object category: bag
    28th object category: shoe
    29th object category: laptop
    30th object category: desk
    31th object category: cabinet
    32th object category: counter
    33th object category: bench
    34th object category: shoes
    35th object category: tower
    36th object category: bottle
    37th object category: helmet
    38th object category: stove
    39th object category: lamp
    40th object category: coat
    41th object category: bed
    42th object category: dog
    43th object category: mountain
    44th object category: horse
    45th object category: plane
    46th object category: roof
    47th object category: skateboard
    48th object category: traffic light
    49th object category: bush
    50th object category: phone
    51th object category: airplane
    52th object category: sofa
    53th object category: cup
    54th object category: sink
    55th object category: shelf
    56th object category: box
    57th object category: van
    58th object category: hand
    59th object category: shorts
    60th object category: post
    61th object category: jeans
    62th object category: cat
    63th object category: sunglasses
    64th object category: bowl
    65th object category: computer
    66th object category: pillow
    67th object category: pizza
    68th object category: basket
    69th object category: elephant
    70th object category: kite
    71th object category: sand
    72th object category: keyboard
    73th object category: plant
    74th object category: can
    75th object category: vase
    76th object category: refrigerator
    77th object category: cart
    78th object category: skis
    79th object category: pot
    80th object category: surfboard
    81th object category: paper
    82th object category: mouse
    83th object category: trash can
    84th object category: cone
    85th object category: camera
    86th object category: ball
    87th object category: bear
    88th object category: giraffe
    89th object category: tie
    90th object category: luggage
    91th object category: faucet
    92th object category: hydrant
    93th object category: snowboard
    94th object category: oven
    95th object category: engine
    96th object category: watch
    97th object category: face
    98th object category: street
    99th object category: ramp
    100th object category: suitcase

    Let's print out all the predicate categories:
    >> predicates = json.load(open('predicates.json'))
    >> for i in range(70):
    ...  print "%dth predicate category: %s" % (i, predicates[i])
    1th predicate category: on
    2th predicate category: wear
    3th predicate category: has
    4th predicate category: next to
    5th predicate category: sleep next to
    6th predicate category: sit next to
    7th predicate category: stand next to
    8th predicate category: park next
    9th predicate category: walk next to
    10th predicate category: above
    11th predicate category: behind
    12th predicate category: stand behind
    13th predicate category: sit behind
    14th predicate category: park behind
    15th predicate category: in the front of
    16th predicate category: under
    17th predicate category: stand under
    18th predicate category: sit under
    19th predicate category: near
    20th predicate category: walk to
    21th predicate category: walk
    22th predicate category: walk past
    23th predicate category: in
    24th predicate category: below
    25th predicate category: beside
    26th predicate category: walk beside
    27th predicate category: over
    28th predicate category: hold
    29th predicate category: by
    30th predicate category: beneath
    31th predicate category: with
    32th predicate category: on the top of
    33th predicate category: on the left of
    34th predicate category: on the right of
    35th predicate category: sit on
    36th predicate category: ride
    37th predicate category: carry
    38th predicate category: look
    39th predicate category: stand on
    40th predicate category: use
    41th predicate category: at
    42th predicate category: attach to
    43th predicate category: cover
    44th predicate category: touch
    45th predicate category: watch
    46th predicate category: against
    47th predicate category: inside
    48th predicate category: adjacent to
    49th predicate category: across
    50th predicate category: contain
    51th predicate category: drive
    52th predicate category: drive on
    53th predicate category: taller than
    54th predicate category: eat
    55th predicate category: park on
    56th predicate category: lying on
    57th predicate category: pull
    58th predicate category: talk
    59th predicate category: lean on
    60th predicate category: fly
    61th predicate category: face
    62th predicate category: play with
    63th predicate category: sleep on
    64th predicate category: outside of
    65th predicate category: rest on
    66th predicate category: follow
    67th predicate category: hit
    68th predicate category: feed
    69th predicate category: kick
    70th predicate category: skate on
