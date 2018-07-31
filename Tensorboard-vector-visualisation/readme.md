This folder contains code and relevant links for the vector only visualisation presented in the paper

## Due to GitHub file size limitations (only 25MB) please refer to the following link for all data and code files:
## https://drive.google.com/open?id=1dKibMcNjlZ-CmfglknR8XMPFskVpO3h7

- jpg_numpy.py  -- file converst all of AMGAN outputs, which have been seperated, into numpy arrays
- script_combinenumpys.py -- combines all of the seperate numpy arrays made by jpg_numpy.py into one numpy array
- convert_numpy file to csv.ipynb -- converts the combined numpyfile to csv file
- out3.csv is the converted numpy array to csv file
- tensorflow_TSNE_plot4.py - setsup the configuration for the tensboard visualiser for the vector points
- project-tensorboard/log-4 -contains appropriate files needed for visualisation of vectors to work 

- to run tensorboard for the seperate vectors download this folder and use `cd custom_tsne_plotting` in the terminal
- next use `tensorboard --logdir=project-tensorboard/log-4 --port=6006` to activate tensorboard
- use the URL that appears in tensorboard to access the visualisation

## Acknowledgement
Source code for tensorboard implementation: https://medium.com/@vegi/visualizing-higher-dimensional-data-using-t-sne-on-tensorboard-7dbf22682cf2
