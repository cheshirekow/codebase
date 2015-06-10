#!/usr/bin/python
from __future__ import print_function
import argparse
import os

def get_lowercase_name(uppercase_name):
  if uppercase_name[0].isupper():
    uppercase_name = uppercase_name[0].lower() + uppercase_name[1:]
  lowercase_name = ''
  for char in uppercase_name:
    if char.isupper():
      lowercase_name += '_' + char.lower()
    else:
      lowercase_name += char
  return lowercase_name

def rename_file(dirpath, filename):
  lowercase_name = get_lowercase_name(filename)
  if lowercase_name != filename:
     print('{dirpath}/{} -> {dirpath}/{}'.format(
           filename, lowercase_name, dirpath=dirpath))
     os.rename('{}/{}'.format(dirpath,filename),
               '{}/{}'.format(dirpath,lowercase_name))

def rename_files_in_directory(startpath, recurse):
  for (dirpath, dirnames, filenames) in os.walk(startpath):
    for filename in filenames:
      rename_file(dirpath, filename)
    if not recurse:
      break


    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Rename files to lowercase.')
  parser.add_argument('-r','--recursive', action='store_true', default=False,
                      help='traverse directories recusively')
  parser.add_argument('files', nargs='*', help="Files to rename")

  args = parser.parse_args()
  for request in args.files:
    if os.path.isdir(request):
      rename_files_in_directory(request, args.recursive)
    elif os.path.isfile(request):
      rename_file('.', request)
 
