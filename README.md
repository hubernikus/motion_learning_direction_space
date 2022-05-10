# Orientation Learning Using Direction Space

```sh
git clone --recurse-submodules https://github.com/hubernikus/motion_learning_direction_space.git
```
(Make sure submodules are there if various_tools librarys is not installed.)

Go to file directory:
```sh
cd motion_learning_direction_space 
``` 

# Setup 
Make sure you use python version 3.10, e.g., setup with virtual environment:

``` bash
python3.10 -m venv .venv
```

Activate the environment
``` bash
source .venv/bin/activate
```

## Install Packages
Setup environment and install packages
``` bash
pip install -e . && pip install -r requirments.txt
```
(TODO: include installing of the requirements in `setup.py`)


## Error Handling
You've got missing modules? Maybe you forgot to add the submodules, add them with:
``` sh
git submodule update --init --recursive
```

