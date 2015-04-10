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
#include <string>
#include <kwargs/is_call_possible.h>


struct Foo {
  Foo(int) {}
  Foo(double) {}
  void operator()(double) const {}
  void operator()(std::string) const {}
  void operator()(const std::string&, int) const {}
};

void foo(double) {
}

using namespace kw;
static_assert(is_call_possible<Foo, void(double)>::value, "here");
static_assert(is_call_possible<Foo, void(int)>::value, "here");
static_assert(is_call_possible<Foo, void(const char *)>::value, "here");
static_assert(is_call_possible<Foo, void(const char *, int)>::value, "here");
static_assert(!is_call_possible<Foo, void(void *)>::value, "here");

static_assert(CanConvertTo<int,double>::value, "here");
static_assert(CanConvertTo<double,double>::value, "here");
static_assert(CanConvertTo<int,Foo>::value, "here");
static_assert(CanConvertTo<double,Foo>::value, "here");
static_assert(!CanConvertTo<std::string,Foo>::value, "here");

int main() {
  return 1;
  foo((foo(1.0), 1.0));
}

