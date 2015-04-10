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

// Tags are just numeric constants that allow us to gerate a unique type for
// all of our keys. We can use enums, as in the previous example, or we can
// can use straight up literals. Though the enums are a lot more readable.
kw::Key<1> c_key;

// We can utilize namespaces to keep our tag names from clobbering each other.
namespace test {

enum Tags {
  c_tag,
  d_tag
};

kw::Key<c_tag> c_key;
kw::Key<d_tag> d_key;

}  // namespace test

template <typename... Args>
void foo(int a, int b, Args... kwargs) {
  kw::ParamPack<Args...> params(kwargs...);

  std::cout << "foo:\n-------"
            << "\na: " << a
            << "\nb: " << b
            // If we want to retrieve a parameter by it's tag value, and not
            // it's key name, we can do that to, but this is pretty low-level.
            << "\nc: " << kw::Get<0>(params,"null")
            // Generally, using the key name is more readable.
            << "\nd: " << kw::Get(params,test::d_key,"null")
            << "\n\n";
}

/// usage of foo in various different ways
int main( int argc, char** argv )
{
  foo(1, 2);
  // foo() unpacks paremeter tagged with "0" so we can use any kw::Key<0> to
  // assign it.
  foo(1, 2, kw::Key<0>() = 3);
  // Generally speaking, using a declared Key is more readable though, and
  // more in the spirit of python-like kwargs.
  foo(1, 2, test::c_key = 3);
  // Unfortunately this opens up the avenue for abuse by using keys other than
  // the intended key. For instance we can use c_key from the root namespace,
  // which in fact packs the parameter with Key<1>, not Key<0> which is what
  // would be packed with test::c_key... So we need to be careful and use
  // the right keys.
  foo(1, 2, c_key = 3);
}

