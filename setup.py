from setuptools import setup, find_packages

setup(
    name="hyperbox",  # you should change "src" to your project name
    version="1.0.0",
    description="Hyperbox: An easy-to-use NAS framework.",
    author="marsggbo",
    url="https://github.com/marsggbo/hyperbox",
    # replace with your own github project link
    install_requires=["pytorch-lightning>=1.5", "hydra-core>=1.1"],
    packages=find_packages(),
)