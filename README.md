# Arcadia
Arcadia is a software used for lab environment for identifying the lab components to a non-technical person. Built using Yolo v11 model with custom data set specifically for Edge Computing Lab components like Raspberry pi, Arduino, various sensors,etc. 
I have Excluded the dataset's images beacause of it's big size, however i mentioned the folder structure for it. 


The Folder structure goes as : 
  Root(Project) Folder:
  |-->Val Folder: 
  |  |->images Folder
  |  |->labels Folder
  |-->Train Folder: 
  |  |->images Folder
  |  |->labels Folder



Packages: 
labelImg: For Annotation i used the lableImg Package Tool 
ultralytics : Used for building yolo model 
wikipedia : Used for Fetching the information about lab componet 
opencv-python : Used for detecting the compononet live 
pyttsx3 : Used to tell the information about component gathered from wikipedia. 
