from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = "Memintelli",
    version = "0.0.0.1",
    author = "Odysseia",
    author_email = "1548384176@qq.com",
    description = "Memristive Intelligient Computing Simulator",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = "MIT",
    license_files = "LICENSE",
    url = "https://github.com/zzzzzzzzzzw/Memintelli",
    zip_safe = False,
    packages=find_packages(include=["pimpy", "NN_layers", "NN_models", "utils"]),
    #install_requires = requirements,
)