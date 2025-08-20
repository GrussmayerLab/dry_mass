# About
This repository is a fork of the phase retrieval script created in the grussmayer lab for the initial quantitative phase retrieval.
Afterwards it uses cellpose 4.0 to segment cells from the phase maps.
Refractive index is recovered in a maybe non-optimal way (thickness map is a bit simplified), better would be an accurate 3D segmentation which I was not able to configure on the example data I had. 
From the refractive index map and the found segmentation masks a dry-mass is calculated. 
Outputs are the segmented cells as png and tif & a csv file with parameters as columns, cells as rows. 
 
# Installation
Via command line:
```sh 
git clone https://github.com/GrussmayerLab/dry_mass.git
```
clones into the repository and gives access to the source code on your local machine
Follow the steps outlined in install.txt or install the environment via the requirements.txt file. 
If you have a GPU available and want to use it, use a python version lower than 3.13 as torch seemed incomatible at the point of creation. 

# Processing
Currently, the workflow only works for a single stack at once for demonstration purposes. 
For that, use 'Main analysis workflow.ipynb'
Batch fitting will soon be implemented. 

# License (tomographic phase retrieval)  
Copyright © 2018 Adrien Descloux - adrien.descloux@epfl.ch, École Polytechnique Fédérale de Lausanne, LBEN/LOB, BM 5.134, Station 17, 1015 Lausanne, Switzerland.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
