# TRAINING RESULTS paper replication:

Outputs to check: 
- test image 65

### DeeplabV3+: 

training 600 epochs with weights = [1, 15, 3, 9, 3]

       Test metric             DataLoader 0

    test_iou_class_0        0.9268433451652527
    test_iou_class_1        0.45537710189819336
    test_iou_class_2        0.3844251334667206
    test_iou_class_3        0.21111884713172913
    test_iou_class_4         0.900140106678009
        test_loss           0.27782541513442993
      test_mean_iou         0.5755809545516968

--------------------------------------------------------

training 600 epochs with weights = [1, 1, 1, 1, 1]


       Test metric             DataLoader 0

    test_iou_class_0        0.9581848978996277
    test_iou_class_1        0.4558284878730774
    test_iou_class_2        0.5127352476119995
    test_iou_class_3        0.1423523873090744
    test_iou_class_4        0.9216744303703308
        test_loss           0.10860651731491089
      test_mean_iou          0.59815514087677

--------------------------------------------------------

training 600 epochs with weights = [1, 2, 1, 4, 1]

       Test metric             DataLoader 0

    test_iou_class_0        0.9609159827232361
    test_iou_class_1        0.5261569619178772
    test_iou_class_2        0.4874131381511688
    test_iou_class_3        0.31306973099708557
    test_iou_class_4        0.9391894340515137
        test_loss           0.11618368327617645
      test_mean_iou         0.6453490257263184