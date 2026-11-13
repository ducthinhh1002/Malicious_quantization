from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name="grow-watermark",
    version="0.1.0",
    description="GROW: Watermark Generation with Progressive Guidance for Diffusion Models (CVPR)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pengcheng Luo, Zexi Jia, Yijia Zhong, Jinchao Zhang, Jie Zhou",
    url="https://github.com/<your-handle>/GROW",
    packages=find_packages(exclude=("scripts", "configs", "assets", "docs")),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
