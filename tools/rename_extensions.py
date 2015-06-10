#!/usr/bin/python
from __future__ import print_function
import argparse
import os

def get_new_name(old_name):
  if old_name[-4:] == '.hpp':
    return old_name[0:-4] + '_impl.h'
  elif old_name[-4:] == '.cpp':
    return old_name[0:-4] + '.cc'
  else:
    return old_name

def rename_file(dirpath, filename):
  new_name = get_new_name(filename)
  if new_name != filename:
     print('{dirpath}/{} -> {dirpath}/{}'.format(
           filename, new_name, dirpath=dirpath))
     os.rename('{}/{}'.format(dirpath,filename),
               '{}/{}'.format(dirpath,new_name))

def rename_files_in_directory(startpath, recurse):
  for (dirpath, dirnames, filenames) in os.walk(startpath):
    for filename in filenames:
      rename_file(dirpath, filename)
    if not recurse:
      break


    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Rename file extensions.')
  parser.add_argument('-r','--recursive', action='store_true', default=False,
                      help='traverse directories recusively')
  parser.add_argument('files', nargs='*', help="Files to rename")

  args = parser.parse_args()
  for request in args.files:
    if os.path.isdir(request):
      rename_files_in_directory(request, args.recursive)
    elif os.path.isfile(request):
      rename_file('.', request)
 
