from distutils.core import setup, Extension
import numpy

def main():

    setup(name="rsnn_backend",
          version="1.0.0",
          description="C library functions for RSNN",
          author="Clayton Seitz",
          author_email="cwseitz@uchicago.edu",
          ext_modules=[Extension("rsnn._rsnn", ["rsnn/_rsnn/rsnn.c"],
                       include_dirs = [numpy.get_include(), '/usr/include/gsl'],
                       library_dirs = ['/usr/lib/x86_64-linux-gnu'],
                       libraries=['m', 'gsl', 'gslcblas'])])


if __name__ == "__main__":
    main()
