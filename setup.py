from setuptools import setup, find_packages


setup(
    name="hyperbox",  # you should change "src" to your project name
    version="1.2.0",
    description="Hyperbox: An easy-to-use AutoML framework.",
    author="marsggbo",
    url="https://github.com/marsggbo/hyperbox",
    # replace with your own github project link
    install_requires=["pytorch-lightning>=1.6", "hydra-core>=1.2"],
    packages=find_packages(),
    include_package_data=True,
)