from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name = "memintelli",
    version = "0.0.2",
    author = "Odysseia",
    author_email = "1548384176@qq.com",
    description = "Memristive Intelligient Computing Simulator",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = "MIT",
    license_files = "LICENSE.txt",
    url = "https://github.com/zzzzzzzzzzw/Memintelli",
    zip_safe = False,
    packages=find_packages(include=["memintelli", "memintelli.*"]),
)