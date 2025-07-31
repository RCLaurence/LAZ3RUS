# LAZ3RUS 
LAZ3RUS is a python enabled code for transinformation of a captured point cloud of a weld geometry in 3D through the use of a laser scanner into stl, stp and meshed ABAQUS input files with FreeCAD for future analysis. 

## Insolation 
### Prerequisites
For proper functionality of LAZ3RUS instillations of [Python 12](https://www.python.org/downloads/release/python-3120/) and [FreeCAD 1.0](https://www.freecad.org/downloads.php) are required.
###Version 
The current version of LAZ3RUS (1.0) can be run as either a python script (automatic_bead_to_finite_element_mesh.py) with hard coded values for necessary fitting thresholds or through the GUI (GUI.py). 



## Running the LAZ3RUS GUI 
Upon launching the [LAZ3RUS GUI](https://github.com/RCLaurence/LAZ3RUS/blob/main/GUI.py) you will be met with the following user interface.
First the setting must be sorted by clicking the settings button. This will allow the loading of the [settings.yaml](https://github.com/RCLaurence/LAZ3RUS/blob/main/settings.yaml) file. 
This file contains the path to the working directory and the path to the FreeCAD 1.0 installation. 

![1](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/1.png)
