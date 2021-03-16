import os
import subprocess
from setuptools import setup
from setuptools import find_namespace_packages
from setuptools.command.develop import develop as develop_orig

os.environ["IN_SETUP"] = "1"  # noqa: E402

from research_daps import __version__, __basedir__  # noqa: E402


def git_crypt_lock():
    """Run git-crypt lock on repo.

    This is a security precaution to avoid unencrypted configs being packaged
    and distributed.
    """
    result = subprocess.run(["git-crypt", "lock"], capture_output=True)

    if result.returncode and (
        result.stderr != b"Error: this repository is already locked.\n"
    ):
        raise OSError(result.returncode, result.stderr)


class develop(develop_orig):
    def run(self):
        super().run()
        subprocess.run(["sh", "bin/post-install-hook.sh"])


git_crypt_lock()

version = "".join(v for v in __version__ if (v.isnumeric() or v == "."))

with open(f"{__basedir__}/requirements.txt") as f:
    required = f.read().splitlines()

exclude = ["docs", "tests"]
common_kwargs = dict(
    version=version,
    license="MIT",
    install_requires=required,
    long_description=open("README.md").read(),
    url="https://github.com/nestauk/research_daps",
    author="nesta",
    author_email="software_development@nesta.org.uk",
    maintainer="nesta",
    maintainer_email="software_development@nesta.org.uk",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)

setup(
    name="research_daps",
    cmdclass={"develop": develop},
    packages=find_namespace_packages(where=".", exclude=exclude),
    **common_kwargs,
)
