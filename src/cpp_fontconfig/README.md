C++ Bindings for fontconfig {#cpp_fontconfig_page}
==========

cpp_fontconfig is a very thin c++ wrapper around libfontconfig. It wraps
only the public API and takes advantage of the fact that most of the
library exposes opaque pointers.

Most of the classes in cpp_fontconfig are wrappers around these opaque
pointers. As such, you should treat objects of these classes as if they
were, in fact, pointers themselves. It is probably better to pass around
copies and/or references to these objects rather than pointers to
these objects.

As a simple demonstration, here is a program which will output a system
font file which matches a requested family name.

@include src/cpp_fontconfig/tutorial.cpp

Running the tutorial application yields the following:

@verbatim
user@machine:~/Codes/cpp/builds/fontconfig/test/tutorial$ ./tutorial Ubuntu
Font found for query [Ubuntu] at /usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-R.ttf

user@machine:~/Codes/cpp/builds/fontconfig/test/tutorial$ ./tutorial UbuntuMono
Font found for query [UbuntuMono] at /usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf

user@machine:~/Codes/cpp/builds/fontconfig/test/tutorial$ ./tutorial Times
Font found for query [Times] at /usr/share/fonts/X11/Type1/n021003l.pfb

user@machine:~/Codes/cpp/builds/fontconfig/test/tutorial$ ./tutorial Arial
Font found for query [Arial] at /usr/share/fonts/truetype/msttcorefonts/Arial.ttf

user@machine:~/Codes/cpp/builds/fontconfig/test/tutorial$ ./tutorial Sans
Font found for query [Sans] at /usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf
@endverbatim








