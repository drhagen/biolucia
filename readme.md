# Biolucia
This is a prototype simulation engine for biological models. At the moment, it is very crude and intended for experimentation, not for production use.

## Installation
(Thanks to Kevin Shi @xpspectre)

### Windows
1. Install [Anaconda with Python 3.5](https://www.continuum.io/downloads) package.
2. Make sure the right python and conda executables are on your path. I did the rest of this in the console emulator [cmder](http://cmder.net/).
3. Make a new conda env: `conda create --name biolucia numpy scipy matplotlib sympy` 
4. Activate the new conda env: `activate biolucia`  Make sure the python and pip executables point to the right ones.
5. Install more packages: `pip install aenum typing funcparserlib`

### Linux
It's the standard virtualenv and install from the following requirements.txt:
```
aenum==1.4.5
cycler==0.10.0
funcparserlib==0.3.6
matplotlib==1.5.1
mpmath==0.19
numpy==1.11.1
pyparsing==2.1.5
python-dateutil==2.5.3
pytz==2016.4
scipy==0.17.1
six==1.10.0
sympy==1.0
typing==3.5.2.2
```