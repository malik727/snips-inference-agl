import io
import os

from setuptools import setup, find_packages

packages = [p for p in find_packages()
            if "tests" not in p and "debug" not in p]

root = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(root, "snips_inference_agl", "__about__.py"),
             encoding="utf8") as f:
    about = dict()
    exec(f.read(), about)

required = [
    "deprecation>=2.0,<3.0",
    "future>=0.16,<0.18",
    "numpy>=1.22.0,<1.22.4",
    "num2words>=0.5.6,<0.6",
    "pyaml>=17.0,<20.0",
    "requests>=2.0,<3.0",
    "scipy>=1.8.0,<1.9.0",
    "threadpoolctl>=2.0.0",
    "scikit-learn==0.24.2",
    "sklearn-crfsuite>=0.3.6,<0.4",
    "snips-nlu-parsers>=0.4.3,<0.4.4",
    "snips-nlu-utils>=0.9.1,<0.9.2",
]

setup(name=about["__title__"],
      description=about["__summary__"],
      version=about["__version__"],
      author=about["__author__"],
      author_email=about["__email__"],
      license=about["__license__"],
      url=about["__github_url__"],
      project_urls={
          "Source": about["__github_url__"],
          "Tracker": about["__tracker_url__"],
      },
      install_requires=required,
      classifiers=[
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      keywords="nlu nlp language machine learning text processing intent",
      packages=packages,
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
      include_package_data=True,
      entry_points={
          "console_scripts": [
              "snips-inference=snips_inference_agl.cli:main"
          ]
      },
      zip_safe=False)
