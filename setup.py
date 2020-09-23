from distutils.core import setup
import setuptools

setup(name='christmass',
      version='0.1.0',
      author="Na'ama Hallakoun",
      author_email='naama.hallakoun@weizmann.ac.il',
      description='Assign stellar mass and metallicity based on HR-diagram location',
      long_description=open('README.md').read(),
      url='https://github.com/naamach/christmass',
      license='LICENSE.txt',
      packages=setuptools.find_packages(),
      install_requires=['astropy', 'configparser', 'numpy', 'scipy', 'tqdm'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Astronomy']
      )
