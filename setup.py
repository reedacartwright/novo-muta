from distutils.core import setup

DISTNAME = 'novo-muta'
MAINTAINER = 'Aaron Baker'
MAINTAINER_EMAIL = 'aabaker99 [at] gmail [dot] com'
DESCRIPTION = ('A set of Python modules for detecting de novo mutations in' + 
               ' a set of genome sequencing data')
LICENSE = 'new BSD' 
URL = 'https://github.com/aabaker99/novo-muta' 
VERSION = '0.0.1' 
LONG_DESCRIPTION = open('README.md').read()
PACKAGES = ['family']
REQUIRES = ['numpy', 'scipy']

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      long_description=LONG_DESCRIPTION,
      packages=PACKAGES,
      requires=REQUIRES)
