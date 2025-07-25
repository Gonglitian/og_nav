from setuptools import setup, find_packages

setup(
    name="og_nav",
    version="1.0.0",
    description="A modular navigation system for robot navigation in OmniGibson environments.",
    author="Litian Gong",
    author_email="gonglitian2002@gmail.com",
    url="https://github.com/Gonglitian/og_nav",
    packages=find_packages(),
    install_requires=[
        "omnigibson",
        "matplotlib"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={"og_nav": ["configs/*.yaml", "assets/*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
        ]
    },
)