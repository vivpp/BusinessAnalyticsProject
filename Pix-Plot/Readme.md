# This file provides a link to the data required for pix-plot

## Acknowlegement - Pix-Plot is the visualiser used to generate the 2D clustering visualisation presented in the paper
link to Pix-Plot GitHub : https://github.com/YaleDHLab/pix-plot

link to my own implementation of pix-plot: https://drive.google.com/open?id=1UaI0yrfahMqGWd3myzuU4et-rEyY9zf1


## Running my implementation on your own system

My own implementation contains all neccessary files needed which have been pre-processed.

Should you wish to run run the pre-processing stage please refer to the process.py file found in the utils folder

Given you have the requirements installed below
all you need to do is download the whole of the `pix-plot-AMGAN-output` folder to your system and in the terminal navigate to the directory where pix-plot has been saved using:

`cd pix-plot-AMGAN-output`

then depending on your python version either run: 

for python 3 - `python -m http.server 5000`

for python 2 - `python -m SimpleHTTPServer 5000`

and open the webserver url shown in the terminal in a browser

## Requirements for pix-plox (from pix-plot GitHub):
Pix-plot requires the following to be installed:
- h5py==2.8.0rc1
- numpy==1.14.3
- Pillow==4.1.1
- psutil==5.2.2
- scikit-learn==0.19.1
- six==1.11.0
- tensorflow==1.8.0
- umap-learn==0.2.3

Pix-plot also requires ImageMagick with jpg support for image resizing. This can be done using the following command:

`brew uninstall imagemagick && brew install imagemagick`

Browser to use with pix-plot: WebGL-enabled e.g. google chrome or mozilla firefox
