 from setuptools import setup, find_packages


setup(name='scoringrule_networks',
      version='1.0.0',
      description='Scoring rule implementations for Tensorflow',
      long_description=readme(),
      url='https://github.com/DaanR/scoringrule_networks',
      author='Daan Roordink',
      author_email='daan.roordink@gmail.com',
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'tensorflow',
          'keras',
          'tensorflow-probability'
      ],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
      ],
      include_package_data=False, #For now
      packages=find_packages(),
