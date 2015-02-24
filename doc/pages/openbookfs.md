openbookfs {#openbookfs_page}
==============================

The openbook filesystem is a fuse-based file synchronizer. Once the fuse
filesystem is mounted file synchronization happens in the background. We're
planning to use a Bayou-like peer-to-peer synchronization strategy, however
at this point in time we're developing the system to use a dedicated server. 

Authors
--------
Joshua Bialkowski
Gabe Ayers
Bruno Alvisio

Doxygen
--------

You can find doxygen documentation [here][1]. Below are
build instructions for Ubuntu 12.04 (though they are quite generic). 

[1]: http://cheshirekow.github.io/
    
@section Build Build Instructions

This instructions are for a more-or-less fresh install of Ubuntu
12.04 as of 25 Mar 2013.

@subsection Build_packages Packages

Install the following packages from the repositories:
  1.  build-essential
  2.  cmake
  3.  doxygen (optional)
  4.  git-core
  5.  mercurial
  6.  sqlite3
  7.  libcrypto++-dev
  8.  libtclap-dev
  9.  libyaml-dev
  10. libboost-filesystem-dev
  11. libfuse-dev
  12. libsqlite3-dev
  13. libsigc++-dev

@verbatim
  $ sudo apt-get install build-essential cmake doxygen mercurial \
    sqlite3 libcrypto++-dev libtclap-dev libyaml-dev \
    libboost-filesystem-dev  libfuse-dev libsqlite3-dev\
    libsigc++-dev
@endverbatim

@subsection Build_protobuf Google Protocol Buffers

Download protocol buffers from http://code.google.com/p/protobuf/downloads/list

Configure, make, and install protocol buffers.

@verbatim
  $ wget http://protobuf.googlecode.com/files/protobuf-2.4.1.tar.gz
  $ tar xvzf protobuf-2.4.1.tar.gz
  $ cd protobuf-2.4.1
  $ ./configure
  $ make
  $ sudo make install
@endverbatim

@subsection Build_yamlcpp YAML-cpp

Checkout the source using mercurial. Build and install. Yaml-CPP has gone
through an API change since it's use in this code, so checkout the
hash 0316abdc76a8.

@verbatim
   $ hg clone https://code.google.com/p/yaml-cpp/
   $ hg checkout -r 0316abdc76a8
   $ mkdir build/yaml-cpp
   $ cd build/yaml-cpp
   $ cmake ../../../yaml-cpp
   $ make
   $ sudo make install
@endverbatim

@subsection Build_soci SOCI

Download version 3.1.0 from http://sourceforge.net/projects/soci/files/.
Build and install the library

@verbatim
  $ wget http://iweb.dl.sourceforge.net/project/soci/soci/soci-3.1.0/soci-3.1.0.zip
  $ unzip soci-3.1.0.zip
  $ mkdir build/soci-3.1.0
  $ cd build/soci-3.1.0
  $ cmake ../../soci-3.1.0
  $ make
  $ sudo make install
@endverbatim

@subsection Build_re2 RE2

Checkout RE2 from the mercurial repository at https://re2.googlecode.com/hg
I'm using revision 2d252384c5e8. Build and install the library.

@verbatim
   $ hg clone https://re2.googlecode.com/hg re2
   $ hg checkout -r 2d252384c5e8
   $ cd re2
   $ sudo make install
@endverbatim

@subsection Build_cppthreads cpp-pthreads

@subsection Build_openbookfs Openbook FS

Checkout the code from https://github.com/cheshirekow/codebase.git
Build and install the library.


@verbatim
  $ git clone https://github.com/cheshirekow/codebase.git cheshirekow
  $ mkdir build/cheshirekow
  $ cd build/cheshirekow
  $ cmake ../../cheshirekow
  $ make -j 
  $ sudo make install
@endverbatim

