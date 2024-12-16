# TRAINING RESULTS paper replication:

### DeeplabV3+: 
val_mean_iou=0.400, train_loss_epoch=0.185, train_mean_iou_epoch=0.491

Total training time for 600 epochs, using batch size of 36: 3506.692738056183 seconds

     Test metric             DataLoader 0
──────────────────────────────────────────────────────────────────────────────

    test_iou_class_0        0.6811823844909668
    test_iou_class_1        0.08742057532072067
    test_iou_class_2        0.21952080726623535
    test_iou_class_3       0.002845690120011568
    test_iou_class_4        0.3153667151927948
        test_loss           0.7325242757797241
      test_mean_iou         0.2612672448158264




#### Training on 60 epochs with no loss weights: 

      Test metric             DataLoader 0
──────────────────────────────────────────────────────────────────────────────────

    test_iou_class_0        0.8867560625076294
    test_iou_class_1        0.08901149034500122
    test_iou_class_2        0.18198399245738983
    test_iou_class_3       0.023161649703979492
    test_iou_class_4        0.7540621757507324
        test_loss           0.35497865080833435
      test_mean_iou         0.3869950771331787