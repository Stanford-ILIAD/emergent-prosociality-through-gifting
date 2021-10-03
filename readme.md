# Gifting 

This directory is supplementary material for the IJCAI 2021 paper:

Woodrow Z. Wang\*, Mark Beliaev\*, Erdem Bıyık\*, Daniel A. Lazar, Ramtin Pedarsani, Dorsa Sadigh. "Emergent Prosociality in Multi-Agent Games Through Gifting"

All relevant citations for methods used are found in the paper's list of references.

## Requirements

We recommend using package manager [pip](https://pip.pypa.io/en/stable/) as well as 
[cuda](https://developer.nvidia.com/cuda-toolkit) to install the relative packages:

**NOTE:** 

**(1)** open-ai's gym and pytorch need to be compatible, the following versions used were chosen 
    for this reason.

**(2)** one might need to build pytorch from source when working on an intel-cpu with Linux

**pip:**

- gym-0.17.2  [gym](https://gym.openai.com/)

**conda:**

- python-3.8.5 [python](https://www.python.org/downloads/release/python-385/)
- numpy-1.19.1 [numpy](https://numpy.org/devdocs/release/1.19.1-notes.html)
- pytorch 1.6.0 [pytorch](https://pytorch.org/get-started/previous-versions/)

## Usage

This directory gives the code responsible for the two main elements in our analysis:

**(1) Computing Basins of Attraction**

To model the dynamics of stag hunt with and without gifting, as done in **Section 4.2**, one should look at the file: **compute_boa.py**.
All relevant information and explanation can be found in the file. 

**(2) Running a DQN algorithm on Stag Hunt with gifting**

As done in most experiemnts in **Section 5**, we run DQN on various environments all deriving from open-ai's 
basic "gym environment" structure. These environments can be found in **/games/**. The DQN structure and algorithm was written for this experiment using pytorch. All relevant files needed to run DQN are given in **/utils/**. We show how to run a DQN algorithm on Stag Hunt with and without gifting, and varying levels of risk in **example.py**. By default **example.py** will run for only 16 seeds. Nonetheless this is enough to show that higher risk settings benefit greatly from additional gifting actions. Parameters used in **example.py** correspond to the ones described in **Section 5.2**, but can  be changed for different experiment settings, such as varying the gift value.



