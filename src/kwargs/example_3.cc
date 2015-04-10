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

enum Keys {
  c_tag,
  d_tag
};

kw::Key<c_tag> c_key;
kw::Key<d_tag> d_key;

// This is a dummy class which just prints to cout whenever it is c'tor-ed or
// d'tor-ed to demonstrate the amount of copying that may happen
class Test {
 public:
  Test(const Test& other) { Construct(); }
  Test() { Construct(); }

  ~Test() {
    std::cout << "Destroyed object " << obj_id_ << "\n";
  }

  int GetId() const { return obj_id_; }

 private:
  void Construct() {
    obj_id_ = n_objs_++;
    std::cout << "Created object " << obj_id_ << "\n";
  }
  int obj_id_;
  static int n_objs_;
};

int Test::n_objs_ = 0;

// when we print a test object, print it's ID
std::ostream& operator<<( std::ostream& out, const Test& test ){
  out << test.GetId();
  return out;
}

// when we print a test object pointer, print the object's id
std::ostream& operator<<( std::ostream& out, const Test* test ){
  out << "*" << test->GetId();
  return out;
}

template <typename... Args>
void foo(int a, int b, Args... kwargs) {
  kw::ParamPack<Args...> params(kwargs...);

  std::cout << "foo:\n----"
            << "\na: " << a
            << "\nb: " << b
            << "\ne: " << kw::Get(params,c_key,"null")
            << "\nf: " << kw::Get(params,d_key,"null")
            << "\n\n";
}

int main( int argc, char** argv )
{
  // Note that 5 test objects are created. The object that shows up in foo()
  // is in fact the 4th copy of the original object!
  foo(1, 2, c_key=Test());

  // We can avoid unnecessary copies by forcing the parameter to be passed
  // by const ref. In this case, only one Test object is actually made.
  foo(1, 2, d_key=kw::ConstRef(Test()));

  Test test_obj;
  // We can use ConstRef on non-temporaries too. No additional object's are
  // created, and foo() see's a reference to test_obj itself.
  foo(1, 2, d_key=kw::ConstRef(test_obj));

  // For non-temporaries, we can use non-const ref as well. Again no additional
  // object's are created, and foo() see's a reference to test_obj itself.
  foo(1, 2, d_key=kw::Ref(test_obj));

  // Though, in general, if foo() needs to modify the object it may be better
  // to pass a pointer. In this case foo() get's a non-const pointer to
  // test_obj.
  foo(1, 2, d_key=&test_obj);
}
