# Using the BART toolbox on an HPC cluster

This is how I was able to get the [Berkeley Advanced Reconstruction Toolbox (BART)](https://mrirecon.github.io/bart/) to run on the [Narval](https://docs.alliancecan.ca/wiki/Narval/en) cluster made available through the [Digital Research Alliance of Canada](https://www.alliancecan.ca/en)

_This allowed me to run the BART toolbox in an interactive node shell. I have not tested it using slurm jobs._

### Why this way?
* This method creates a container image locally before copying and running it on the cluster. It may be difficult to install BART directly on the cluster (See [BLAS and LAPACK](https://docs.alliancecan.ca/wiki/BLAS_and_LAPACK) libraries on the cluster.) 
* Another workaround would be to use a debian:testing container (as opposed to ubuntu:20.04). But, python packages may not work correctly.
* This method uses [Apptainer](https://docs.alliancecan.ca/wiki/Apptainer), the docker software supported on the Compute Canada clusters.
 
### Overview :
1. Creating a container image on the local machine.
2. Installing bart via this container image.
3. Creating a Python file that imports bart.
4. Copying and running the image.

## Creating a container image on the local machine

Check if you have [apptainer installed](https://apptainer.org/docs/admin/main/installation.html) on your local machine.

````commandline
apptainer --version
````

Create a new directory where you want to store your image and bart file. Create a definition file (.def) in the new directory.

````commandline
mkdir bart-container
cd bart-container/
nano container1.def 
````

We can now create a definition file for our container image.

````
#container 1.def

BootStrap: docker
from: ubuntu:20.04

%post
        apt-get update -y
        apt-get install -y python3
        apt-get install -y python3-pip
        apt-get install -y make wget gcc libfftw3-dev liblapacke-dev libpng-dev
        pip install numpy
        pip install h5py
        pip install matplotlib
        pip install SimpleITK
%runscript
        echo "Hello from the container!"
````

This .def file defines an ubuntu:20.04 container. It installs Python, pip, and some packages required for BART. You can also install the Python libraries you would need for your program. This file installs numpy, h5py, matplotlib, and SimpleITK libraries.


Now, we can build an image from this definition file.

````commandline
apptainer build container1.sif container1.def
````

Now, the apptainer build function will run. After a few moments, your container image (.sif) will be created.

You can test your container image by executing the runscript.

```commandline
apptainer run container1.sif
```


## Installing BART via this container image

We can now open the container as an interactive shell. We will mount the container with the current directory.

```commandline
mkdir workdir
apptainer shell -W /home/timothy/bart-container/workdir/ -C -B /home/timothy/bart-container container1.sif
```

* -W sets the working directory.
* -C tells Apptainer not to mount the default directory.
* -B asks to specifically mount the provided directory.
  
This will open an interactive shell. Here, you can download packages or install other libraries.

```commandline
Apptainer> python3 --version
```

Now, we will install BART, following the [installation documentation](https://mrirecon.github.io/bart/installation.html)

```commandline
Apptainer> wget https://github.com/mrirecon/bart/archive/v0.8.00.tar.gz
Apptainer> xzvf v0.8.00.tar.gz
Apptainer> cd bart-0.8.00
Apptainer> make 
Apptainer> exit
```

This will install bart: version 0.8.00 in a directory called bart-0.8.00. 

## Creating a Python file that imports bart
Before we copy our container to the cluster, let's create a test python file.

```commandline
cd ~/bart-container/
nano test-import.py
```
```python
#test-import.py

import os
import sys
path = os.environ["TOOLBOX_PATH"] + "/python/";
sys.path.append(path)

try:
    import bart
    print ('Success, BART library imported!')
except ImportError as err:
    print ('Error: ', err)
```

The first four lines ensure that the 'TOOLBOX_PATH' is set to the base directory.

## Copying and running the image

Now, we can copy our directory (with the container image and the BART files) to the cluster.

```commandline
cd 
scp /home/timothy2/bart-container/ timothy2@narval.computecanada.ca:/home/timothy2/scratch/
```

Now, connect to your cluster.
```commandline
ssh timothy2@narval.computecanada.ca
cd /home/timothy2/scratch/bart-container/
```

If you want, you can connect to one of the compute nodes on your cluster.
```commandline
salloc --nodes=1 --time=00:05:00 --mem=4G
```

Now, we can open an interactive container shell.
```commandline
module load apptainer
apptainer shell -W /home/timothy2/scratch/bart-container/workdir/ -C -B /home/timothy2/scratch/bart-container/ container1.sif
```

Before we can run our Python program, we must run the BART startup script.
```commandline
Apptainer> cd /home/timothy2/scratch/bart-container/bart-0.8.00/
Apptainer> . startup.sh 
```

Now, we can run our Python program.
```commandline
Apptainer> cd ..
Apptainer> python3 test-import.py
```
If the BART library was able to import, you should see the success message.
