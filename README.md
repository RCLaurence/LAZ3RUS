# LAZ3RUS 
LAZ3RUS is a python enabled code for transinformation of a captured point cloud of a weld geometry in 3D through the use of a laser scanner into stl, stp and meshed ABAQUS input files with FreeCAD for future analysis. 

## Instillation 
### Prerequisites
For proper functionality of LAZ3RUS instillations of [Python 12](https://www.python.org/downloads/release/python-3120/) and [FreeCAD 1.0](https://www.freecad.org/downloads.php) are required.

### Version 
The current version of LAZ3RUS (1.0) can be run as either a python script (automatic_bead_to_finite_element_mesh.py) with hard coded values for necessary fitting thresholds or through the GUI (GUI.py). 



## Running the LAZ3RUS GUI 
### Applying user settings
Upon launching the [LAZ3RUS GUI](https://github.com/RCLaurence/LAZ3RUS/blob/main/GUI.py) you will be met with the following user interface.
First the setting must be sorted by clicking the **Settings** button. This will allow the loading of the [settings.yaml](https://github.com/RCLaurence/LAZ3RUS/blob/main/settings.yaml) file. 
This file contains the path to the working directory and the path to the FreeCAD 1.0 installation. 

![1](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/1.png)

### Loading data
Next load the laser scan data of the bead using the **Load** button. This data should be in an three column, x, y, z format (.xyz). Example bead data can is available [here](https://github.com/RCLaurence/LAZ3RUS/blob/main/Data/bead_0_no_transformation_sampled.zip).
![2](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/2.png)

### Apply transformation matrices
With the data loaded, an xy plot of the scanned area will appear with the recorded height represented by a colour plot. This data is in the scanners coordinate system. To both correct for any miss alignment of the scanner and transform the data into the welding cells coordinate system transformation matrices must be applied to the data.
The procedure for creating these matrices is outlined in the accompanying publication, **you will need to generate your own for your own welding cell**. In this case two are required, the first to correct for the skew and the second to move into the weld cell coordinate system. Example matrices can be found [here](https://github.com/RCLaurence/LAZ3RUS/tree/main/Data). 
Press the **Transform** button to open the transinformation matrix dialogue box. As there are two matrices **load** them one after another, the dialogue box will retain the combined matrix. To apply the matrix close the dialogue box with the **x**. 

![3](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/3.png)

## Preparing the bead for fitting
### selecting the area of interest

To select the area of interest, i.e. the bead, of the now transformed data press the *R* key to activate the bounding box tool. With the tool active drag and select an area around bead to be fitted. The press the **Crop** button to crop the data. 

![4]https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/4.png)

### Remove scanning artifacts
After cropping the data to the bead change the view to show the y,z plane. With the bounding box too still active, draw around the area of interest again to remove any scanning artifacts. Once the area is selected again press the **crop** button.  


![5]https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/5.png)

### Flatten the plate around the bead









