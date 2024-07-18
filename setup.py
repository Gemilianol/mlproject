from setuptools import find_packages, setup
from typing import List

E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]: #Returns a list of string
    '''
    This function returns a list of requirements.
    '''
    requirements = []
    with open (file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        
        if E_DOT in requirements:
            requirements.remove(E_DOT)
        
        return requirements
    
setup(
    name = "mlproject",
    version = "0.0.1",
    author = "Emiliano",
    author_email="glabonia@mail.utdt.edu",
    packages=find_packages(),
    requires = get_requirements("requirements.txt")
)