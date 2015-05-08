#!/usr/bin/python

"""
A script to do some top level formatting of source files. Will check that the
copyright block is correct for the given project. If not, will re-write the
copyright block. Will also check inclusion guard style and fix it if not
correct. 
"""

from __future__ import print_function
import argparse
import os.path
import tempfile
import re
import datetime


COPYRIGHT_FMT = (
"""/*
 *  Copyright (C) {year} Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of {project}.
 *
 *  {project} is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  {project} is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with {project}.  If not, see <http://www.gnu.org/licenses/>.
 */""")

COPYRIGHT_REGEX = re.compile(r'Copyright\s*\(C\)\s*(\d{4})')
HEADER_FILE_REGEX = re.compile(r'\.hp{0,2}$')

def GetProjectPath(query_path):
  project_path = query_path
  while not os.path.exists(os.path.join(project_path, '.git')):
    next_path = os.path.normpath(os.path.join(project_path, os.pardir))
    if os.path.samefile(next_path, project_path):
      raise Exception("Failed to find project root from {}".format(query_path))
    project_path = next_path
  return os.path.abspath(project_path)

def GetProjectRelativePath(query_path):
  project_path = GetProjectPath(query_path)
  return os.path.relpath(os.path.abspath(query_path), project_path)

def GetProjectName(query_path):
  return GetProjectRelativePath(query_path).split(os.sep)[1]

def GetInclusionGuard(query_path):
   return re.subs(r'\W', '_', GetProjectRelativePath(query_path)).upper()

def FormatFile(file_path):
  temp_file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.tmp', 
                                          prefix='cppformat', dir='/tmp', 
                                          delete=False)

  print("Writing to temporary file {}".format(temp_file.name))
  target_file = open(file_path, 'r')
  
  # chomp any initial blank lines
  for line_content in target_file:
    if line_content.strip():
      break

  # check if this is a comment block
  if line_content.strip().startswith('/*'):
    # consume the whole block
    comment_content = line_content
    for line_content in target_file:
      comment_content += line_content      
      if r'*/' in line_content:
        break;

    # Check if this is already a copyright block. If it is, extracchet date and
    # write to target file. Otherwise, write a copyright block to the new file
    # as well as the comment we just consumed
    match = COPYRIGHT_REGEX.search(comment_content)
    if match:
      temp_file.write(COPYRIGHT_FMT.format(year=match.group(1), 
                                           project=GetProjectName(file_path)))
    else:
      print('No copyright found')
      temp_file.write(COPYRIGHT_FMT.format(year=datetime.date.today().year, 
                                           project=GetProjectName(file_path)))
      temp_file.write(comment_content)
    temp_file.write('\n')

  for line in target_file:
    temp_file.write(line)
     
  temp_file_path = temp_file.name
  temp_file.close()
  target_file.close()

  os.rename(temp_file_path, file_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Reformat the file')
  parser.add_argument('file_paths', help='The file to format', nargs='+')
  args = parser.parse_args()

  for file_path in args.file_paths:
    if os.path.isfile(file_path):
      FormatFile(file_path)
