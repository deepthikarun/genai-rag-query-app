import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s')

project_name = "FastRAG"

list_of_files = [
    ".github/workflows/.gitkeep" ,  # for CI/CD deploymnet create a yaml file inside this directory,
                                     #inititially keep a .gitkeep empty file to upload to git, else empty folder is not uploaded
                                     # and github actions fails
     "main.py",
     "requirements.txt",
     ".env",
     "rag_chain.py",
     "data/",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        logging.info(f"Creating empty file: {filepath}")
        with open(filepath, "w") as f:
            pass
    
    else:
        logging.info(f"File {filename} already exists")


