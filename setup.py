from setuptools import setup, find_packages

required_modules  = []
with open('requirements.txt') as f:
    required = f.read().splitlines()
for module in required:
    if not module.startswith('#'):
        required_modules.append(module)

setup(
    name="hyperbox",  # you should change "src" to your project name
    version="1.4.3",
    description="Hyperbox: An easy-to-use NAS framework.",
    author="marsggbo",
    url="https://github.com/marsggbo/hyperbox",
    # replace with your own github project link
    install_requires=required_modules,
    packages=find_packages(),
    include_package_data=True,
)
