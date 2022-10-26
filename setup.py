from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop


class CocosPostInstall(install):
    """This downloads the tree-sitter grammars at the end of the installation process."""

    def run(self):
        install.run(self)
        print("Downloading tree-sitter grammars... (this may take a while)", flush=True)
        from cocos.tokenizecode import CodeParser

        t = CodeParser()  # just to download the grammars
        print("Done!")


class CocosPostDevelop(develop):
    """This downloads the tree-sitter grammars at the end of the installation process."""

    def run(self):
        develop.run(self)

        print("Downloading tree-sitter grammars... (this may take a while)", flush=True)
        from cocos.tokenizecode import CodeParser

        t = CodeParser()  # just to download the grammars
        print("Done!")


setup(
    name="cocos",
    packages=["cocos"],
    version="1.0",
    python_requires=">=3.9",
    description="",
    author="Johannes Villmow",
    author_email="johannes.villmow@hs-rm.de",
    url="https://github.com/villmow/coling-cocos",
    license="MIT License",
    entry_points={
        "console_scripts": [
            "cocos-convert = cocos.utils:cli_convert_checkpoint_to_huggingface",
        ],
    },
    # FIXME filter required!
    install_requires=[
        "datasets",
        "tokenizers==0.12.0",  # 0.12.0
        "transformers==4.18.0",  # v4.18.0
        "scikit-learn",
        "scipy",
        "hydra-core",
        "rich",
        "pytorch_lightning==1.5.8",
        "wandb",
        "pytorch_metric_learning",
        "gdown",
        "requests",
        "tree-sitter==0.20.4",
        "matplotlib",
        "seaborn",
    ],
    keywords=[
        "source code",
        "machine learning",
        "ml4code",
        "code search",
        "contextualized code search",
    ],  # Keywords that define your package best
    classifiers=[
        "Development Status :: 4 - Beta",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",  # Define that your audience are developers
        "Topic :: Software Development",
        "Topic :: Software Development :: Documentation",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT",
        "Programming Language :: Python :: 3",
    ],
    cmdclass={
        "install": CocosPostInstall,
        "develop": CocosPostDevelop,
    },
)
