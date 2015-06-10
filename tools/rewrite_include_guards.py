#!/usr/bin/python
from __future__ import print_function
import argparse
import os
import re
import sys
import tempfile

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

def get_header_guard(filename):
  src_rel_path = get_src_relative_path(filename)
  return re.sub('\W', '_', src_rel_path.upper()) + '_'

def rewrite_file(dirpath, filename):
  filepath = '{}/{}'.format(dirpath,filename)

  if not filename.endswith('.h'):
    print('Skipping ' + filepath)
    return
  print('Rewriting ' + filepath);

  header_guard = get_header_guard(filepath)
  out_file = tempfile.NamedTemporaryFile(prefix=filepath, delete=False)

  ifndef_found = False
  define_found = False
  endif_found  = False

  with open(filepath) as in_file:
    for line in in_file:
      if not ifndef_found and line.startswith('#ifndef'):
        ifndef_found = True
        out_file.write('#ifndef {}\n'.format(header_guard))
        continue
      if not define_found and line.startswith('#define'):
        define_found = True
        out_file.write('#define {}\n'.format(header_guard))
        continue
      if not endif_found and line.startswith('#endif'):
        endif_found = True
        out_file.write('#endif  // {}\n'.format(header_guard))
        continue
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
  parser = argparse.ArgumentParser(description='Rewrite inclusion guards.')
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
 
