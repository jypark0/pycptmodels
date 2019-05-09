import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycptmodels",
    version="0.0.1",
    author="Jung Yeon Park",
    author_email="jpark0@gmail.com",
    description="Equipment models for clustered photolithography tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jypark0/pycptmodels",
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    setup_requires=["pytest-runner"],
    tests_requires=["pytest"]
)
