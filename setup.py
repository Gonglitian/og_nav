from setuptools import setup, find_packages

setup(
    name="og_nav",
    version="1.0.0",
    description="A modular navigation system for robot navigation in OmniGibson environments.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-repo/og_nav",
    packages=find_packages(),
    install_requires=[
        "omnigibson",
        "torch",
        "numpy",
        "opencv-python",
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
            # 可选: 添加命令行入口
        ]
    },
)