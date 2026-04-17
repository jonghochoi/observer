from setuptools import setup, find_packages

setup(
    name="observer",
    version="0.1.0",
    description="Automated evaluation pipeline for dexterous manipulation RL policies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pyyaml",
        "matplotlib",
    ],
    extras_require={
        "tracking": ["wandb", "tensorboard"],
        "tactile":  ["opencv-python"],
    },
    entry_points={
        "console_scripts": [
            "observer=eval_runner:main",
        ],
    },
)
