from setuptools import find_packages, setup

setup(
    name="dspy_medical_ai",
    version="2.0.0",
    author="DSPy Medical AI Team",
    author_email="team@dspy-medical.ai",
    description="Self-optimizing medical AI system powered by DSPy",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "dspy>=2.5.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "pydantic>=2.9.0",
        "pydantic-settings>=2.6.0",
        "langchain>=0.3.26",
        "langchain-pinecone>=0.2.8",
        "langchain-openai>=0.3.24",
        "sentence-transformers>=4.1.0",
        "python-dotenv>=1.1.0",
        "pypdf>=5.6.1",
    ],
    entry_points={
        "console_scripts": [
            "dspy-medical=app:main",
        ],
    },
)