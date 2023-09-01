## Low light image enhancement using terminal application
# Used some important libraries
***
This code is a Python script that implements a low-light image enhancement technique using two methods: LIME (Low-light IMage Enhancement) and DUAL. The script provides a graphical user interface (GUI) built with the Kivy library, allowing users to select images and apply the enhancement technique.

## Steps:

1. # Importing Libraries:
   - The code begins by importing the necessary libraries, including argparse, RawTextHelpFormatter, glob, os, tqdm, cv2, kivy, and shutil. These libraries provide various functionalities for command-line argument parsing, file operations, image processing, GUI development, and more.

2. # Loading the GUI Layout:
   - The code uses the Kivy Builder module to load the GUI layout defined in Kivy language format. The layout includes buttons for selecting images and converting them, as well as a scrollable grid for displaying the selected images and their enhanced versions.

3. # Defining the MainScreen Class:
   - The MainScreen class is a subclass of the BoxLayout widget provided by Kivy. It represents the main screen of the GUI and contains methods for selecting images and converting them.

4. # Image Selection:
   - The select_images() method is called when the "Select Images" button is pressed. It opens a file chooser dialog using the Plyer library and allows the user to select one or multiple image files. The selected files are then copied to a destination directory.

5. # Main Image Enhancement Process:
   - The main(args) function is defined outside the MainScreen class. It takes the command-line arguments parsed by argparse and performs the low-light image enhancement process.
   - The function first loads the images from the specified folder and stores them in a list.
   - It creates a directory for saving the enhanced image files if it doesn't already exist.
   - It iterates over the images, enhances each image using the enhance_image_exposure() function from the enhancement module, and saves the enhanced image to the destination directory.
   - During the enhancement process, the function also adds the enhanced images to the GUI grid for display.

6. # Argument Parsing:
   - Inside the if _name_ == "_main_": block, the script uses the argparse library to parse command-line arguments.
   - The arguments include the folder path to the test images, gamma correction parameter, weight for balancing illumination refinement, method selection (LIME or DUAL), and various other parameters related to the enhancement process.
   - The parsed arguments are passed to the main() function for further processing.

7. # Defining the MyApp Class:
   - The MyApp class is a subclass of the App class provided by Kivy. It represents the application and defines the build() method, which returns an instance of the MainScreen class.

8. # Running the Application:
   - Finally, the script checks if it is being run directly (not imported as a module) and creates an instance of the MyApp class, which starts the GUI application by calling its run() method.

9. # Enhancement Module (enhancement.py):
   - The enhancement module contains functions and utilities for the low-light image enhancement technique.
   - Functions include creating a spatial affinity kernel, computing smoothness weights, fusing multiple exposure images, refining illumination maps, correcting underexposure, and enhancing image exposure.
   - The functions utilize various techniques such as image filtering, Laplacian matrix computation, solving linear systems, gamma correction, and more.

10. # Utility Module (utils.py):
   - The utils module provides a helper function, get_sparse_neighbor(), which returns the sparse neighbors of a given pixel position in an image grid. It is used by the refinement process in the enhancement module to compute the spatially inhomogeneous Laplacian matrix.
***


# Libraries used... 
1. import argparse
2. from argparse import RawTextHelpFormatter
3. import glob
4. from os import makedirs
5. from os.path import join, exists, basename, splitext
6. from tqdm import tqdm
7. import cv2
8. from kivy.app import App
9. from kivy.lang import Builder
10. from kivy.uix.boxlayout import BoxLayout
11. from kivy.uix.image import Image
12. from plyer import filechooser
13. import shutil
14. import os 

