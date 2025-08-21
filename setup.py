from setuptools import setup, find_packages

setup(
    name="homeostasis",
    version="0.1.0",
    description="An open-source framework for self-healing systems",
    author="Homeostasis Contributors",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.4.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0.0",
        "requests>=2.31.0",
        "cryptography>=41.0.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "jsonschema>=4.0.0",
        "semver>=3.0.0",
        "PyJWT>=2.8.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "httpx>=0.25.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ]
    },
    python_requires=">=3.8",
)