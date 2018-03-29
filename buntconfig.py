"""
Buntstrap configuration for building mpblocks
"""

from buntstrap import config

apt_sources = config.get_bootstrap_sources('amd64', 'xenial')
apt_http_proxy = config.get_apt_cache_url()
apt_include_essential = False
apt_include_priorities = ['required']
apt_packages = [
    'apt',
    'build-essential',
    'cmake',
    'libboost-all-dev',
    'libeigen3-dev',
    'libglfw-dev',
    'libglew-dev',
    'libgoogle-glog-dev',
    'libgtkmm-3.0-dev',
    'libtinyxml2-dev',
    'libyaml-cpp-dev',
    'libcrypto++-dev',
    'libtclap-dev',
    'libsoci-dev',
    'oracle-java8-installer',
    'oracle-java8-set-default'
]

binds = [
    '/dev/urandom',
    '/etc/resolv.conf',
]

# buntstrap --config <thisfile> -- ${PWD}/mpblocks-build
# sudo mount -t proc procfs mpblocks-build/proc
# uchroot --binds ${PWD}/mpblocks:src /etc/resolv.conf /dev/urandom /tmp/.X11-unix -- mpblocks-build
# export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-oracle/jre/lib/amd64/jli
# DISPLAY=:0
