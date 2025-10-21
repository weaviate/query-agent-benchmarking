from setuptools import setup, find_packages

setup(
    name="query_agent_benchmarking",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "weaviate-client[agents]>=4.15.4",
        "weaviate-agents>=1.0.0",
        "pandas>=2.3.1",
        "datasets>=4.0.0",
        "ir-datasets>=0.5.11",
        "pip>=25.2",
        "setuptools>=80.9.0",
        "wheel>=0.45.1",
        "twine>=6.2.0",
    ],
)