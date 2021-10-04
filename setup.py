from setuptools import setup, find_packages

setup(
    name="hyperbox",  # you should change "src" to your project name
    version="0.0.1",
    description="Hyperbox: An easy-to-use NAS framework.",
    author="marsggbo",
    # replace with your own github project link
    install_requires=["pytorch-lightning>=1.2.0", "hydra-core>=1.0.6"],
    packages=find_packages(),
)