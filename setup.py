# -*- coding: utf-8 -*-
"""


███╗   ███╗ █████╗ ██╗   ██╗        ████████╗██╗  ██╗███████╗         █████╗  █████╗ ██████╗ ███████╗        
████╗ ████║██╔══██╗╚██╗ ██╔╝        ╚══██╔══╝██║  ██║██╔════╝        ██╔══██╗██╔══██╗██╔══██╗██╔════╝        
██╔████╔██║███████║ ╚████╔╝            ██║   ███████║█████╗          ██║  ╚═╝██║  ██║██║  ██║█████╗          
██║╚██╔╝██║██╔══██║  ╚██╔╝             ██║   ██╔══██║██╔══╝          ██║  ██╗██║  ██║██║  ██║██╔══╝          
██║ ╚═╝ ██║██║  ██║   ██║              ██║   ██║  ██║███████╗        ╚█████╔╝╚█████╔╝██████╔╝███████╗        
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝              ╚═╝   ╚═╝  ╚═╝╚══════╝         ╚════╝  ╚════╝ ╚═════╝ ╚══════╝        

██████╗ ███████╗        
██╔══██╗██╔════╝        
██████╦╝█████╗          
██╔══██╗██╔══╝          
██████╦╝███████╗        
╚═════╝ ╚══════╝        

 ██╗       ██╗██╗████████╗██╗  ██╗        ██╗   ██╗ █████╗ ██╗   ██╗
 ██║  ██╗  ██║██║╚══██╔══╝██║  ██║        ╚██╗ ██╔╝██╔══██╗██║   ██║
 ╚██╗████╗██╔╝██║   ██║   ███████║         ╚████╔╝ ██║  ██║██║   ██║
  ████╔═████║ ██║   ██║   ██╔══██║          ╚██╔╝  ██║  ██║██║   ██║
  ╚██╔╝ ╚██╔╝ ██║   ██║   ██║  ██║           ██║   ╚█████╔╝╚██████╔╝
   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝           ╚═╝    ╚════╝  ╚═════╝ 
"""

from setuptools import setup, find_packages
import sys
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Determine Python version
py_version = sys.version_info
if py_version >= (3, 13):
    req_file = "requirements/requirements-py313.txt"
    nlp_file = "requirements/requirements-nlp-py313.txt"
    img_file = "requirements/requirements-img-py313.txt"
elif py_version >= (3, 12):
    req_file = "requirements/requirements-py312.txt"
    nlp_file = "requirements/requirements-nlp-py312.txt"
    img_file = "requirements/requirements-img-py312.txt"
elif py_version >= (3, 11):
    req_file = "requirements/requirements-py311.txt"
    nlp_file = "requirements/requirements-nlp-py311.txt"
    img_file = "requirements/requirements-img-py311.txt"
elif py_version >= (3, 10):
    req_file = "requirements/requirements-py310.txt"
    nlp_file = "requirements/requirements-nlp-py310.txt"
    img_file = "requirements/requirements-img-py310.txt"
elif py_version >= (3, 9):
    req_file = "requirements/requirements-py39.txt"
    nlp_file = "requirements/requirements-nlp-py39.txt"
    img_file = "requirements/requirements-img-py39.txt"
elif py_version >= (3, 8):
    req_file = "requirements/requirements-py38.txt"
    nlp_file = "requirements/requirements-nlp-py38.txt"
    img_file = "requirements/requirements-img-py38.txt"
else:  # Python 3.7
    req_file = "requirements/requirements-py37.txt"
    nlp_file = "requirements/requirements-nlp-py37.txt"
    img_file = "requirements/requirements-img-py37.txt"

# Load core requirements
def load_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

install_requires = load_requirements(req_file)
utils_requires = load_requirements("requirements/requirements-utils.txt")
nlp_requires = load_requirements(nlp_file)
img_requires = load_requirements(img_file)
dev_requires = load_requirements("requirements/requirements-dev.txt")

setup(
    name="scomp_link",
    version="0.1.0",
    author="scomp-link contributors",
    description="The Astromech arm for your Python data projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scomp_link",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "nlp": nlp_requires,
        "img": img_requires,
        "utils": utils_requires,
        "dev": dev_requires,
        "all": nlp_requires + img_requires + utils_requires,
    },
)
