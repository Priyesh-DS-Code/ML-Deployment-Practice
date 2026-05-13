from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    requirement=[]
    with open(file_path) as library_requirement:
        requirement=library_requirement.readlines()
        requirement=[req.replace("\n", "") for req in requirement]
        if HYPHEN_E_DOT in requirement:
            requirement.remove(HYPHEN_E_DOT)
            
    return requirement

setup(
    name="ML Deployment",
    version='0.0.1',
    author="Priyesh Kumar Kashyap",
    author_email="Priyeshkashyap47@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)