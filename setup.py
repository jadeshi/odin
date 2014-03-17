u"""
setup.py: Install ODIN
"""

import os, sys,re
from os.path import join as pjoin
from glob import glob

#try:
    #from setuptools import Extension, setup
#except:

from distutils.extension import Extension
from distutils.core import setup

from Cython.Distutils import build_ext

import numpy

import subprocess
from subprocess import CalledProcessError

# ------------------------------------------------------------------------------
# HEADER
# 

VERSION     = "0.0.1"
ISRELEASED  = False
__author__  = "TJ Lane"
__version__ = VERSION

metadata = {
    'name': 'odin',
    'version': VERSION,
    'author': __author__,
    'author_email': 'tjlane@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://github.com/tjlane/odin',
    'download_url': 'https://github.com/tjlane/odin',
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'pyyaml', 'mdtraj', 
                         'nose', 'cython>=0.16', 'tables'],
    'dependency_links' : ['https://github.com/kif/fabio/tarball/master#egg=fabio-0.1.3'],
    'platforms': ['Linux', 'OSX'],
    'zip_safe': False,
    'test_suite': "nose.collector",
    'description': "Code for Structure Determination",
    'long_description': """ODIN is a simulation toolpackage for producing
models of biomolecular structures consistent with a large set of experimental
data."""}


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS -- path finding, git, python version, readthedocs
# 

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    
    
def print_warning(string):
    print bcolors.WARNING + string + bcolors.ENDC
    

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None
    

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


def git_version():
    """
    Return the git revision as a string.
    Copied from numpy setup.py
    """
    
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


# ------------------------------------------------------------------------------
# GPU FUNCTION WRAPPING -- nvcc support
# python distutils doesn't have NVCC by default
# 

def locate_cuda():
    """
    Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDA_HOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDA_HOME env variable is in use
    if 'CUDA_HOME' in os.environ:
        home = os.environ['CUDA_HOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            print_warning('The nvcc binary could not be located in your $PATH. '
                          'add it to your path, or set $CUDA_HOME.')
            return False
            
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    print "CUDA config:", cudaconfig
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            print_warning('The CUDA %s path could not be located in %s' % (k, v))
            return False
    return cudaconfig
    
CUDA = locate_cuda()
if CUDA == False:
    CUDA_SUCCESS = False
else:
    CUDA_SUCCESS = True

def customize_compiler_for_nvcc(self):
    """
    Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


# ------------------------------------------------------------------------------

metadata['packages']     = ['odin', 'odin.scripts', 'odin.smfret']
metadata['package_dir']  = {'odin' :         'src/python',
                            'odin.scripts' : 'scripts',
                            'odin.smfret'  : 'src/python/smfret'}

metadata['ext_modules']  = []    
metadata['scripts']      = [s for s in glob('scripts/*') if not s.endswith('__.py')]
metadata['data_files']   = [('reference', glob('./reference/*'))]
metadata['cmdclass']     = {'build_ext': custom_build_ext}

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    setup(**metadata) # ** will unpack dictionary 'metadata' providing the values as arguments
