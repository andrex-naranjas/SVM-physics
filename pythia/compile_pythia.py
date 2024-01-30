#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------
 Authors: A. Ramirez-Morales (andres.ramirez.morales@cern.ch)
          T. Cisneros-Perez
---------------------------------------------------------------
"""
import os
import urllib.request

PYTHIAVERS="pythia8310"

def main(arg1):
    """
    Function to build a shared library for Pythia:
    - Download the Pythia code from pythia official website
    - Compile the code with appropiate flags
    - Create the shared library
    - Clean up
    """  
    print("Fetching Pythia...")
    os.chdir(arg1)
    url = "https://pythia.org/download/pythia83/" + PYTHIAVERS + ".tgz"
    urllib.request.urlretrieve(url, "pythia.tar.gz")
    os.system('tar xzvf pythia.tar.gz')
    os.chdir(arg1 + PYTHIAVERS)
    os.system('./configure --with-python-config=python3-config')
    os.system('make')
    # os.system('export PYTHONPATH=$(PREFIX_LIB):$PYTHONPATH')
    os.chdir("..")
    print("Cleaning up...")
    os.system("rm -rf pythia.tar.gz")
    os.system("rm -rf " + PYTHIAVERS + "/")

    
if __name__ == "__main__":
    main("./")
