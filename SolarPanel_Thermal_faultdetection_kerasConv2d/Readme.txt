"SolarPanelThermalFaultDetection.h5" is trained file for Solar thermal Images for Fault Detection with accuracy(training and validation) nearly 0.99.

"CrackDetectionModel.h5" is trained on artificially generated images representing Surface crack detection.



---------------------------------------------------------------------------------------------------------------------
PYTHON Files

1.)Cnn1.py file is common model to train differnt datasets
    Once the .h5 file is gerented, comment out lines (9-57) to run inference on new images
2.) "SolarPanelThermalFaultDetection.py" is Cnn-2d model using keras lib, comment out line 90 and 92 once .hf file is made to run inference

3.)Cnn2_withStructure.py is same as above python file (modified)

------------------------------------------------------------------------------------------------------------------------

Dataset Folder

dataset1 contains training_set and test_set subfolders, each having 2 folders representing 2 classes of fault and nofault

image1,image2,.... are to run inference on new solar panel images, just make sure you commented out training partin model.