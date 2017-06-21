from setuptools import setup

setup(name='primfilters',
      version='0.1',
      description='Python package with image filters',
      author='Ella Bahry',
      license='GPLv2',
      author_email='bellonetatgmaildotcom',
      url='',
      packages=['primfilters'],
      install_requires=[
          'numpy',
          'scikit-image',
          'scipy',
          'joblib'
      ]
     )