from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as f:
    readme = f.read()


setup(
      name="trvae-pytorch",
      version="0.0.1",
      author="Oscar Davalos",
      author_email="odavalos2@ucmerced.edu",
      description="A repackaging of trvaep",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="MIT",
      url="https://github.com/odavalos/trvae-pytorch",
      download_url="https://github.com/odavalos/trvae-pytorch",
      packages=find_packages(),
      keywords=['Single-cell RNASeq', 'scRNA-Seq','Integration', 'Neural Networks', 'Autoencoders', 'Tabular Data'],
      install_requires=[
                        "get_version",
                        "scanpy",
                        "adjustText",
                        "torch",
                        "torchvision",
                        "numpy",
                        "scipy",
                        "seaborn",
                        "matplotlib",
                        "adjustText",
                        "pandas"
                        ]
)
