#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from setuptools import find_packages, setup

setup(
    name="hyperbox-app",
    version="0.1",
    packages=find_packages(include=["hyperbox_app"]),
    entry_points={"console_scripts": ["hyperbox_app = hyperbox_app.main:main"]},
    author="marsggbo",
    author_email="marsggbo@foxmail.com",
    include_package_data=True,
)