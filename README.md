# LAZ3RUS 
LAZ3RUS is a python enabled code for transinformation of a captured point cloud of a weld geometry in 3D through the use of a laser scanner into stl, stp and meshed ABAQUS input files with FreeCAD for future analysis. 

The use of LAZ3RUS along with all the welding setup steps needed for geometry capture are covered in detail in the accompanying JoVE publication: [Automatic Laser-based Geometry Capture for Finite Element Analysis of Weld Beads](https://app.jove.com/t/68654/automatic-laser-based-geometry-capture-for-finite-element-analysis)

## Citing LAZ3RUS
IF you use LAZ3RUS in your research please cite as: RC Laurence, MJ Roy& J Li. (2025). RCLaurence/LAZ3RUS: LAZ3RUS (v0.1). Zenodo. https://doi.org/10.5281/zenodo.15175021

## Instillation 
### Prerequisites
For proper functionality of LAZ3RUS instillations of [Python 12](https://www.python.org/downloads/release/python-3120/) and [FreeCAD 1.0](https://www.freecad.org/downloads.php) are required.

### Version 
The current version of LAZ3RUS (1.1) can be run as either a python script (automatic_bead_to_finite_element_mesh.py) with hard coded values for necessary fitting thresholds or through the GUI (GUI.py). 



## Running the LAZ3RUS GUI 
### Applying user settings
Upon launching the [LAZ3RUS GUI](https://github.com/RCLaurence/LAZ3RUS/blob/main/GUI.py) you will be met with the following user interface.
First the setting must be sorted by clicking the **Settings** button. This will allow the loading of the [settings.yaml](https://github.com/RCLaurence/LAZ3RUS/blob/main/settings.yaml) file. 
This file contains the path to the working directory and the path to the FreeCAD 1.0 installation. 

![1](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/1.png)

### Loading data
Next load the laser scan data of the bead using the **"Load"** button. This data should be in an three column, x, y, z format (.xyz). Example bead data can is available [here](https://github.com/RCLaurence/LAZ3RUS/blob/main/Data/bead_0_no_transformation_sampled.zip).
![2](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/2.png)

### Apply transformation matrices
With the data loaded, an xy plot of the scanned area will appear with the recorded height represented by a colour plot. This data is in the scanners coordinate system. To both correct for any miss alignment of the scanner and transform the data into the welding cells coordinate system transformation matrices must be applied to the data.
The procedure for creating these matrices is outlined in the accompanying publication, **you will need to generate your own for your own welding cell**. In this case two are required, the first to correct for the skew and the second to move into the weld cell coordinate system. Example matrices can be found [here](https://github.com/RCLaurence/LAZ3RUS/tree/main/Data). 
Press the **Transform** button to open the transinformation matrix dialogue box. As there are two matrices **"Load"** them one after another, the dialogue box will retain the combined matrix. To apply the matrix close the dialogue box with the **"x"**. 

![3](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/3.png)

## Preparing the bead for fitting
### Selecting the area of interest

To select the area of interest, i.e. the bead, of the now transformed data press the **"R"** key to activate the bounding box tool. With the tool active drag and select an area around bead to be fitted. The press the **"Crop"** button to crop the data. 

![4](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/4.png)

### Remove scanning artifacts
After cropping the data to the bead change the view to show the y,z plane. With the bounding box too still active, draw around the area of interest again to remove any scanning artifacts. Once the area is selected again press the **"Crop"** button.  


![5](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/5.png)

### Flatten the plate around the bead
Return to the to down view. With the bounding box tool still active select the **"Invert selection"** tick box and draw around the bead area. Next press the **"Level"** button to flatten the plate. 

![6](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/6.png)

### Orientate the bead for fitting
To help insure a reliable fit, the bead is orientated to be parallel to the y axis. This is done by the selecting the **"Rotate"** tick box and pressing the **"Orient"** button. 

![7](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/7.png)

### Find the height of the plate
With the bounding box tool and the **"Invert selection"** tick box active draw around the bead a press the **"select"** button. This will find the height of the plate denoted in the box **"p="**. This can be manually adjusted if desired. 

![8](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/8.png)

## Fit the Bead
Use the **"Fit bead** button to fit the bead.

![9](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/9.png)

### Confirm the fit is appropriate 
The fit is now complete and appears as a black line across the centre. The extremities of the bead to create the 3D representation can be seen in grey. The **"slider bar"** can be used to make the data more transparent to better view the fitting underneath. 

![10](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/10.png)

### Create the 3D representations
To create the stl, stp and inp files first chose the size of the base plate using the **"w" "h" "t"** options within the **"FreeCAD""** section. Then press the **"Run"** command. This with prompt a save dialogue option where you can name your files. If you want the final output to be in the same coordinate system as the welding cell as opposed to the bead coordinate system used for fitting select the **"Invert transform"** option prior to committing to the run. 

![11](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/11.png)

### Veiw the result 
The final stl can be veiw by pressing the **"Load STL"** button. The stl, stp and inp files have also now been created. 

![12](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/12.png)

## Bring the geometry into ABAQUS
Using the **"File > Import > Model"** option within ABAQUS CAE the meshed inp file brought into ABAQUS where it can act as the basis for future FEA analysis of the weld. 

![13](https://github.com/RCLaurence/LAZ3RUS/blob/main/Images/13.png)








