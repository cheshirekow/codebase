#include <string>
#include "kwargs/is_call_possible.h"


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

