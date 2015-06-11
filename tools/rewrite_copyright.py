#!/usr/bin/python
from __future__ import print_function
import argparse
import datetime
import os
import re
import sys
import tempfile

EXTENSION_PATTERN = re.compile(r'.*\.((h)|(cc))')
COPYRIGHT_PATTERN = re.compile(r'Copyright\s*\(C\)\s*(\d{4})')
COMMENT_START_PATTERN = re.compile(r'/\*')
COMMENT_END_PATTERN = re.compile(r'\*/')

COPYRIGHT_STRING = """/*
 *  Copyright (C) {copyright_year} Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of {project_name}.
 *
 *  {project_name} is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  {project_name} is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with {project_name}.  If not, see <http://www.gnu.org/licenses/>.
 */
"""

def get_path_to_src(start_path):
  head, tail = os.path.split(start_path)
  if not head :
    sys.stderr.write('Failed to find path to <project>/src\n')
    sys.exit(1)
  if tail == 'src':
    return start_path
  else:
    return get_path_to_src(head)  

def get_src_relative_path(filename):
  abs_path = os.path.abspath(filename)
  src_path = get_path_to_src(abs_path)
  return os.path.relpath(abs_path, src_path)

def get_first_path_component(path):
  head = path
  while head:
    head, tail = os.path.split(head)
  return tail

def get_subproject_name(filename):
  src_rel_path = get_src_relative_path(filename)
  return get_first_path_component(get_src_relative_path(filename))


def rewrite_file(dirpath, filename):
  filepath = '{}/{}'.format(dirpath,filename)

  if not EXTENSION_PATTERN.match(filepath):
    print('Skipping ' + filepath)
    return
  print('Rewriting ' + filepath);

  subproject_name = get_subproject_name(filepath)
  copyright_year = datetime.datetime.now().year

  out_file = tempfile.NamedTemporaryFile(prefix=filepath, delete=False)
  
  with open(filepath) as in_file:
    # chomp whitespace
    for line in in_file:
      if line.strip():
        break;
    
    # if this is the start of a comment the start chomping unti we get to 
    # the end of the comment, assuming this is a copyright comment
    if COMMENT_START_PATTERN.search(line):
      for line in in_file:
        match = COPYRIGHT_PATTERN.search(line)
        if match:
          copyright_year = match.group(1)
        if COMMENT_END_PATTERN.search(line):
          out_file.write(COPYRIGHT_STRING.format(copyright_year=copyright_year, 
                                                 project_name=subproject_name))
          break
  
    # otherwise, add the copyright comment
    else:
      out_file.write(COPYRIGHT_STRING.format(copyright_year=copyright_year, 
                                             project_name=subproject_name))
      out_file.write(line)
    
    # and copy te rest of the file
    for line in in_file:
      out_file.write(line)

    out_file.close()
    in_file.close()
    os.rename(out_file.name, filepath)

def rewrite_files_in_directory(startpath, recurse):
  for (dirpath, dirnames, filenames) in os.walk(startpath):
    for filename in filenames:
      rewrite_file(dirpath, filename)
    if not recurse:
      break

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Rewrite copyright comments.')
  parser.add_argument('-r','--recursive', action='store_true', default=False,
                      help='traverse directories recusively')
  parser.add_argument('files', nargs='*', help="Files to rename")

  args = parser.parse_args()
  for request in args.files:
    if os.path.isdir(request):
      rewrite_files_in_directory(os.path.abspath(request), args.recursive)
    elif os.path.isfile(request):
      head, tail = os.path.split(os.path.abspath(request))
      rewrite_file(head, tail)
 
