# Installation instructions

The exercises in this course use Python with several additional tools and packages.

The most convenient way to use Python is with the Conda environment. This will create a sort of sandbox where your specific Python version and packages are available without affecting the main installation of your machine.


## Install Miniconda

The first thing we need is a recent installation of [Miniconda](https://docs.conda.io/en/latest/miniconda.html). This is a minimal installation of Anaconda without all the packages pre-installed. You can find the latest Windows installer [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe). Make sure you use version 3.7 of Python.

Use all the default settings of the installer. If it asks for who to install it, leave it at "Just Me" and do not select "All Users". It works a lot easier this way.


## Setup environment

Start an "Anaconda Prompt". We will now create a new Conda environment with the name `dlscratch` that contains Python (3.7), Jupyter, Matplotlib, Pandas, and TensorFlow. First create the environment with most packages using the following command.

    conda create -n dlscratch python==3.7.4 jupyter==1.0.0 matplotlib==3.1.1 pandas==0.25.1

All required packages will be downloaded (about 280 MB) and then installed. It can take a few minutes.

Once it's finished we can check if it works by activating the environment.

    conda activate dlscratch

Next up is TensorFlow. We need the latest and greatest version of TensorFlow, 2.0, which is not yet available as a Conda package. So we have to install it via `pip`.  Make sure you are still in the `dlscratch` Conda environment and type the following command.

    pip install tensorflow==2.0.0

This will download and install the latest version of TensorFlow and its dependencies (about 60 MB in total).

When this is done you can close the current prompt.


## Test environment

### Windows

If you are using Windows, you should now have a new Start menu entry called `Jupyter Notebook (dlscratch)`. Start this and a console window will start that displays a URL that will also be opened automatically. So a browser should pop up with a website that allows browsing your home folder. If it doesn't show up, then open the URL that is noted in the console window.

### Linux

If you are using Linux then you have to open a command prompt and browse to a nice folder in your home dir. Now activate the right Conda environment and start Jupyter Notebook.

    conda activate dlscratch
    jupyter notebook .

It will display a URL that will also be opened automatically. So a browser should pop up with a website that allows browsing your home folder. If it doesn't show up, then open the URL that is noted in the console window.

### Test

Go to a folder you like and click on `New` -> `Python 3`. A new tab will open that shows you a notebook with an empty cell (starts with `In [ ]`). Enter the following lines in that cell.

    import numpy
    import pandas
    import matplotlib
    import tensorflow
    print('OK')

Now hit `CTRL+ENTER` or the `>| Run` button on the toolbar, the label will change to `In [*]` while it is busy and when it is done it should change to `In [1]` and `OK` will be printed below the cell. If no warnings are reported, then it all works fine.

Close the tab and hit the `Quit` button in the folder browse page. You can now close this browser tab as well and you are all set up for this course.
