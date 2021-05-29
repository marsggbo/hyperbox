from setuptools import setup, find_packages


setup(
    name="hyperbox", # Replace with your own username
    version="0.1",
    author="marsggbo",
    author_email="marsggbo@foxmail.com",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "projects")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)