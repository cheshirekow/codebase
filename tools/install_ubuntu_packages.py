#!/usr/bin/python

from __future__ import print_function
import argparse
import os
import subprocess
import sys

PACKAGE_LIST = [
  'cmake',
  'libboost-dev',
  'libboost-filesystem-dev',
  'libboost-random-dev',
  'libboost-system-dev',
  'libcairomm-1.0-dev',
  'libcrypto++-dev',
  'libeigen3-dev',
  'libfcgi-dev',
  'libfontconfig1-dev',
  'libfreetype6-dev',
  'libfuse-dev',
  'libgflags-dev',
  'libglew-dev',
  'libglib2.0-dev',
  'libgtkmm-3.0-dev',
  'libgl1-mesa-dev',
  'libglfw-dev',
  'libgoogle-glog-dev',
  'libpangomm-1.4-dev',
  'libprotobuf-dev',
  'libsigc++-2.0-dev',
  'libtclap-dev',
  'libtinyxml2-dev',
  'libx11-dev',
  'libxcomposite-dev',
  'libxdamage-dev',
  'libxext-dev',
  'libxfixes-dev',
  'libxrandr-dev',
  'libxrender-dev',
  'libyaml-cpp-dev',
  'openjdk-7-jdk',
  'protobuf-compiler',
]

def main():
  if os.getuid() != 0:
    print('Please re-run as root')
    sys.exit(1)

  parser = argparse.ArgumentParser('Installs ubuntu package dependencies')
  parser.add_argument('-u', '--update', action='store_true',
                      help='if specified, will run apt-get update first')
  args = parser.parse_args()

  if args.update:
    subprocess.check_call(['apt-get', 'update'])
  subprocess.check_call(['apt-get', 'install', '-y'] + PACKAGE_LIST)

if __name__ == '__main__':
  main()
