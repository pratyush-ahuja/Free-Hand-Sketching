# Free-Hand-Sketching
##Dataset
Dataset can be downloaded from here-https://github.com/googlecreativelab/quickdraw-dataset. 

##Procedure
1. Get the dataset for the emojis in qd_emo file and place the .npy files in /data folder.
2. Run LoadData.py which will load the data from the /data folder and store the features and labels in pickel files.
3. Run QD_trainer.py which will load data from pickle and augment it. After this, the training process begins.
4. Run QuickDrawApp.py which will use use the webcam to get what you have drawn.
5. For altering the model, check QD_trainer.py.
