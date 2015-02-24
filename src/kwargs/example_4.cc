#include <iostream>
#include "kwargs/kwargs.h"

constexpr uint64_t c_tag = kw::Hash("c_tag");
constexpr uint64_t d_tag = TAG(d_tag);

// global symbols used as keys in list of kwargs
kw::Key<c_tag> c_key = KW(c_tag);
kw::Key<d_tag> d_key;


// a function taking kwargs parameter pack
template <typename... Args>
void foo(int a, int b, Args... kwargs) {
  // first, we construct the parameter pack from the parameter pack
  kw::ParamPack<Args...> params(kwargs...);

  std::cout << "foo:\n--------"
            << "\na: " << a
            << "\nb: " << b
  // We can attempt to retrieve a key while providing a default fallback value.
  // If c_key is in kwargs then this will return the value associated with
  // that key, and will have the correct type. Note that the type of the default
  // parameter in this case is const char*.
            << "\nc: " << params.Get(KW(c_tag),"null");
  // We can also do stuff conditionally based on whether or not arg exists in
  // the param pack. We still need to provide a default value, since we need to
  // know the return type of the Get function when the key is not in kwargs.
  // In this case, the default value wont ever be used at runtime.
  if( params.Contains(KW(d_tag)) ) {
    std::cout << "\nd: " << kw::Get(params,KW(d_tag),0);
  }

  std::cout << "\n\n";
}

int main( int argc, char** argv )
{
  std::cout << "tag_c : " << kw::Tag("c_tag").Hash() << "\n"
            << "tag_d : " << kw::Tag("d_tag").Hash() << "\n";
  foo(1, 2);
  foo(1, 2, KW(c_tag)=3);
  foo(1, 2, KW(c_tag)=3, KW(d_tag)=4);
  foo(1, 2, KW(d_tag)=4);
}
