import io
import os
import sys
from shutil import rmtree
import numpy as np

from setuptools import find_packages, setup, Command, find_namespace_packages

NAME = "pydmclab"
DESCRIPTION = "Package to facilitate DFT calculations and analysis"
URL = "https://github.umn.edu/bartel-group/pydmclab"
EMAIL = "cbartel@umn.edu"
AUTHOR = "Chris Bartel"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "1.2.0"

REQUIRED = ["numpy>=1.25.0", "mp-api>=0.39", "pymatgen>=2023.0.0"]
EXTRAS = {
    "mlp": ["chgnet"],
    "defects": ["pymatgen-analysis-defects"],
    "phonons": ["phonopy"],
}
here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_namespace_packages(include=["pydmclab.*", "pydmclab.**.*"]),
    package_data={"pydmclab.data.data": ["*.json", "*.yaml", "*.csv", "*.dat"]},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        "upload": UploadCommand,
    },
    include_dirs=[np.get_include()],
)
