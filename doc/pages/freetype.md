C++ Bindings for freetype {#cpp_freetype_page}
==========

cpp_freetype is a C++ wrapper around the FreeType2 library. It is a simple,
thin wrapper that doesn't do much on it's own. I wouldn't consider it
very useful except to vain C++ programs who want the font config objects
to be nicely namespaced and to eliminate the FT_ macros from their code.

cpp_freetype mostly just provides objects which wrap freetype pointers
with the methods of the underlying object.

Note-to-self: You probably want to link against this library using
Link Time Optimization (LTO) (i.e. the -flto switch with gcc). It can
inline functions that are declared in this library in the code that
uses it.

As an example, here is a simple program which loads a font file and
displays some information from it, and then laods the outline for the
character 'A' and prints the list of path segments:

@include test/cpp_freetype/tutorial/main.cpp

The output of this program looks like the following

@verbatim
user@machine:~/path/to$ ./test /usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-R.ttf
Some info about the font:
      filepath: /usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-R.ttf
        family: Ubuntu
         style: Regular
  n fixed size: 0
    n charmaps: 3
      scalable: yes
      n glyphs: 1264
  units per EM: 1000
      charmaps:
                unic
                armn
                unic

Set charmap to index: 2
for char A :
    ascii: 65
    index: 36
   format: outl
 contours: 2
   points: 25


Contour: 0

   (549,0)  on
   (532,45)  off  quadradic
   (502,132)  off  quadradic
   (486,177)  on
   (172,177)  on
   (109,0)  on
   (8,0)  on
   (48,110)  off  quadradic
   (118,297)  off  quadradic
   (185,465)  off  quadradic
   (251,618)  off  quadradic
   (287,693)  on
   (376,693)  on
   (412,618)  off  quadradic
   (478,465)  off  quadradic
   (545,297)  off  quadradic
   (615,110)  off  quadradic

Contour: 1

   (458,257)  on
   (426,344)  off  quadradic
   (363,507)  off  quadradic
   (329,582)  on
   (294,507)  off  quadradic


josh@Nadie:~/Codes/cpp/builds/cppfreetype/test$

@endverbatim

