#include <gtest/gtest.h>

#include <array>
#include <list>
#include <set>
#include <vector>

// Adapted from
// https://groups.google.com/forum/#!msg/comp.lang.c++.moderated/T3x6lvmvvkQ/mfY5VTDJ--UJ

template <typename Type>
class has_push_back {
  class yes {
    char m;
  };

  class no {
    yes m[2];
  };

  struct BaseMixin {
    void push_back() {}
  };

  struct Base : public Type, public BaseMixin {};

  template <typename T, T t>
  class Helper {};

  template <typename U>
  static no deduce(U*, Helper<void (BaseMixin::*)(), &U::push_back>* = 0);
  static yes deduce(...);

 public:
  static const bool result = sizeof(yes) == sizeof(deduce((Base*)(0)));
};

namespace details {

template <typename type>
class void_exp_result {};

template <typename type, typename U>
U const& operator, (U const&, void_exp_result<type>);

template <typename type, typename U>
U& operator, (U&, void_exp_result<type>);

template <typename src_type, typename dest_type>
struct clone_constness {
  typedef dest_type type;
};

template <typename src_type, typename dest_type>
struct clone_constness<const src_type, dest_type> {
  typedef const dest_type type;
};

}  // namespace details

template <typename type, typename call_details>
struct can_push_back {
 private:
  class yes {};
  class no {
    yes m[2];
  };

  struct derived : public type {
    using type::push_back;
    no push_back(...) const;
  };

  typedef typename details::clone_constness<type, derived>::type derived_type;

  template <typename T, typename due_type>
  struct return_value_check {
    static yes deduce(due_type);
    static no deduce(...);
    static no deduce(no);
    static no deduce(details::void_exp_result<type>);
  };

  template <typename T>
  struct return_value_check<T, void> {
    static yes deduce(...);
    static no deduce(no);
  };

  template <bool has, typename F>
  struct impl {
    static const bool value = false;
  };

  template <typename arg1, typename r>
  struct impl<true, r(arg1)> {
    static const bool value =
        sizeof(return_value_check<type, r>::deduce(
            (((derived_type*)0)->push_back(*(arg1*)0),
             details::void_exp_result<type>()))) == sizeof(yes);
  };

  // specializations of impl for 2 args, 3 args,..
 public:
  static const bool value =
      impl<has_push_back<type>::result, call_details>::value;
};

int main() {
  static_assert(!can_push_back<std::array<int, 3>, void(int)>::value,
                "std::array<array> can push");
  static_assert(can_push_back<std::list<int>, void(int)>::value,
                "std::list<int> can't push it's own type");
  static_assert(!can_push_back<std::set<int>, void(int)>::value,
                "std::set<int> can push");
  static_assert(can_push_back<std::vector<int>, void(int)>::value,
                "std::vector<int> can't push it's own type");
  //  static_assert(is_container<Foo, void(int)>::value, "");
  //  static_assert(is_container<Foo, void(const char*)>::value, "");
  //  static_assert(!is_container<Foo, void(void*)>::value, "");
}
