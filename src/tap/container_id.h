#pragma once

#include <iterator>
#include <type_traits>
#include "common.h"

// Adapted from
// https://groups.google.com/forum/#!msg/comp.lang.c++.moderated/T3x6lvmvvkQ/mfY5VTDJ--UJ

namespace tap {

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

template <typename T>
struct is_cstyle_array : public std::false_type {
  enum { value = 0 };
};

template <typename T, std::size_t SIZE>
struct is_cstyle_array<T(&)[SIZE]> : public std::true_type {
  enum { value = 1 };
};

template <typename T>
struct is_pointer : public std::false_type {
  enum { value = 0 };
};

template <typename T>
struct is_pointer<T*> : public std::true_type {
  enum { value = 1 };
};

template <typename Type>
class has_push_back {
  // clang-format off
  class yes { char m; };
  class no { yes m[2]; };
  // clang-format on

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

template <typename Type>
class has_insert {
  // clang-format off
  class yes { char m; };
  class no { yes m[2]; };
  // clang-format on

  struct BaseMixin {
    void insert() {}
  };

  struct Base : public Type, public BaseMixin {};

  template <typename T, T t>
  class Helper {};

  template <typename U>
  static no deduce(U*, Helper<void (BaseMixin::*)(), &U::insert>* = 0);
  static yes deduce(...);

 public:
  static const bool result = sizeof(yes) == sizeof(deduce((Base*)(0)));
};

template <typename type, typename call_details>
struct can_insert {
 private:
  class yes {};
  class no {
    yes m[2];
  };

  struct derived : public type {
    using type::insert;
    no insert(...) const;
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
            (((derived_type*)0)->insert(*(arg1*)0),
             details::void_exp_result<type>()))) == sizeof(yes);
  };

  // specializations of impl for 2 args, 3 args,..
 public:
  static const bool value = impl<has_insert<type>::result, call_details>::value;
};

template <typename Type>
class has_begin {
  // clang-format off
  class yes { char m; };
  class no { yes m[2]; };
  // clang-format on

  struct BaseMixin {
    void insert() {}
  };

  struct Base : public Type, public BaseMixin {};

  template <typename T, T t>
  class Helper {};

  template <typename U>
  static no deduce(U*, Helper<void (BaseMixin::*)(), &U::begin>* = 0);
  static yes deduce(...);

 public:
  static const bool result = sizeof(yes) == sizeof(deduce((Base*)(0)));
};

template <typename Type>
class has_end {
  // clang-format off
  class yes { char m; };
  class no { yes m[2]; };
  // clang-format on

  struct BaseMixin {
    void insert() {}
  };

  struct Base : public Type, public BaseMixin {};

  template <typename T, T t>
  class Helper {};

  template <typename U>
  static no deduce(U*, Helper<void (BaseMixin::*)(), &U::end>* = 0);
  static yes deduce(...);

 public:
  static const bool result = sizeof(yes) == sizeof(deduce((Base*)(0)));
};

template <typename T, typename = void>
struct is_iterator {
  static constexpr bool value = false;
};

template <typename T>
struct is_iterator<
    T, typename std::enable_if<
           !std::is_same<typename std::iterator_traits<T>::value_type,
                         void>::value>::type> {
  static constexpr bool value = true;
};

template <typename T, bool has_begin_, bool has_end_>
struct has_iterators_impl {
  static const bool value = false;
};

template <typename T>
struct has_iterators_impl<T, true, true> {
  static const bool value = true;
};

template <typename T>
struct has_iterators {
  static const bool value =
      has_iterators_impl<T, has_begin<T>::result, has_end<T>::result>::value;
};

template <class T, class R = void>
struct enable_if_type {
  typedef R type;
};

template <class T, class Enable = void>
struct get_value_type : std::false_type {
  typedef T value_type;
};

template <class T>
struct get_value_type<T, typename enable_if_type<typename T::value_type>::type>
    : std::true_type {
  typedef typename T::value_type value_type;
};

}  // namespace tap
