/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of kwargs.
 *
 *  kwargs is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  kwargs is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with kwargs.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @author Josh Bialkowski <josh.bialkowski@gmail.com>
 */
#include <iostream>
#include <kwargs/v1.h>

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
            << "\nc: " << params.GetAs("c_tag",0);
  // We can also do stuff conditionally based on whether or not arg exists in
  // the param pack. We still need to provide a default value, since we need to
  // know the return type of the Get function when the key is not in kwargs.
  // In this case, the default value wont ever be used at runtime.
  if( params.Contains("d_tag") ) {
    std::cout << "\nd: " << params.GetAs("d_tag",0);
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
