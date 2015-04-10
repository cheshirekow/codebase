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
#ifndef KWARGS_IS_CALL_POSSIBLE_H_
#define KWARGS_IS_CALL_POSSIBLE_H_

namespace kw {

/**
 * From:
 * https://groups.google.com/forum/embed/#!topic/comp.lang.c++.moderated/T3x6lvmvvkQ
 */

/// Template metafunction, ::result evaluates to true if Type has a function
/// call operator member.
template<typename Type>
class has_member {
  class yes {
    char m;
  };

  class no {
    yes m[2];
  };

  /// Very simply, a class which has a function call operator.
  struct DummyFnCallOperator {
    void operator()() {}
  };

  /// This class derives from both the argument of the meta function, and a
  /// class which we know has a function call operator.
  struct HasAtLeastOneFnCallOperator :
      public Type,
      public DummyFnCallOperator {
  };

  /// This template only exists for a pair of template parameters where the
  /// type of the second parameter is equivalent to the first parameter
  template<typename T, T t>
  class Helper {};

  /// SFINAE. The first overload for deduce takes two arguments, the first is
  /// a pointer to an object. This parameter appears to be used only for type
  /// deduction. The second argument is a pointer to a Helper<> object.
  /// However, since the helper template only exists if &U::operator() is the
  /// same type as void (BaseMixin::*)() then this type only exists under the
  /// condition that Base has exactly one available overload for operator(),
  /// and it is the one that comes from BaseMixin.
  template<typename U>
  static no deduce(
      U*, Helper<void (DummyFnCallOperator::*)(), &U::operator()>* = 0);

  /// If the first definition of deduce has a substitution failuer then this is
  /// the one that the compiler finds.
  static yes deduce(...);

 public:
  static const bool result =
      sizeof(yes) == sizeof(deduce((HasAtLeastOneFnCallOperator*) (0)));
};


namespace details {

// If the operator matching the desired inputs returns void, then when
// we take a void expression and comma-operator it with one of these guys,
// the result will be one of these guys. This will lead to the overload
// resolution of a function taking this type as a parameter
template<typename type>
class void_exp_result {};

template<typename type, typename U>
U const& operator,(U const&, void_exp_result<type>);

template<typename type, typename U>
U& operator,(U&, void_exp_result<type>);

/// Template meta function, copy's "const" from src_type to dest_type
template<typename src_type, typename dest_type>
struct clone_constness {
  typedef dest_type type;
};

/// Template meta function, copy's "const" from src_type to dest_type.
/// Specialization for when src_type is const.
template<typename src_type, typename dest_type>
struct clone_constness<const src_type, dest_type> {
  typedef const dest_type type;
};

}  // namespace details


template <typename type, typename call_details>
struct is_call_possible
{
private:
  class yes {};
  class no {
    yes m[2];
  };

  /// A class which will definitely have a function call operator. If `type`
  /// has a function call operator then `derived` will also have that same
  /// operator available; If `type` does not have a function call operator,
  /// then the only function call operator available to `derived` is the one
  /// returning a `no`.
  struct derived : public type {
    using type::operator();
    no operator()(...) const;
  };

  /// Same as `derived` but with the same const-ness as `type`.
  typedef typename details::clone_constness<type, derived>::type derived_type;

  /// Provides a member function deduce() which accepts any parameters and whose
  /// return type is always `no` except for when the single parameter matches
  /// `DesiredReturnType`
  template<typename T, typename DesiredReturnType>
  struct return_value_check {
    // This overload will be resolved by the compiler if the function
    // call operator matching the desired inputs returns anything that is
    // convertable to `DesiredReturnType`.
    static yes deduce(DesiredReturnType);

    // If the operator matching the desired input arguments returns anything
    // other than void which is not convertable to a DesiredReturnType, then
    // this is the overload that will be resolved by the compiler.
    static no deduce(...);

    // If no operator matching the desired input arguments is found, then
    // the operator() from derived that returns `no` will be the one that
    // get's resolved below.
    static no deduce(no);

    // If the operator matching the desired inputs returns void, then when
    // we take a void expression and comma-operator it with one of
    static no deduce(details::void_exp_result<type>);
  };

  /// If the desired return type is void, then we will ignore the return
  /// value of the function call anyway, so it doesn't matter what the
  /// return value is... Any return value will be compatable.
  template<typename T>
  struct return_value_check<T, void> {
    static yes deduce(...);
    static no deduce(no);
  };

  template <typename Mid, typename... Tail>
  struct Dispatcher {
    template <typename... Head>
    static inline auto Dispatch( Head ... head)
      -> decltype(Dispatcher<Tail...>::Dispatch(head..., (*(Mid*)0))) {
      Dispatcher<Tail...>::Dispatch(head..., (*(Mid*) 0));
    }
  };

  template <typename Tail>
  struct Dispatcher<Tail> {
    template <typename... Head>
    static inline auto Dispatch(Head... head)
      -> decltype(((derived_type*) 0)->operator()(head..., (*(Tail*)0))){
      ((derived_type*) 0)->operator()(head..., (*(Tail*)0));
    }
  };

  /// Default template, HasFnCallOperator is false so obviously no function
  /// call is possible.
  template<bool HasFnCallOperator, typename F>
  struct impl {
    static const bool value = false;
  };

  template<typename... args, typename r>
  struct impl<true, r(args...)> {
    typedef Dispatcher<args...> CallDispatcher;
    // The thing we're taking the `sizeof` here is a function call, so the
    // `sizeof` call will evaluate to the `sizeof` the function's return type.
    //
    // The function being called is the deduce method of
    // `return_value_check<type,r>`. The method resolved by this call will
    // be the one that returns a `yes` object only if the parameter of
    // `deduce()` is compatible with `r`.
    //
    // The parameter of `deduce` is a comma separated tuple of the return
    // type of the function call operator and a `void_exp_result` object. If
    // the function call operator returns void then this tuple resolves to
    // be just the latter argument. Otherwise, the comma operator overload above
    // passes through the return value of the function call operator.
    static const bool value = sizeof(return_value_check<type, r>::deduce(
        (CallDispatcher::Dispatch(), details::void_exp_result<
            type>()))) == sizeof(yes);
  };

  // specializations of impl for 2 args, 3 args,..
 public:
  static const bool value = impl<has_member<type>::result, call_details>::value;
};

template <typename SourceType, typename TargetType>
struct CanConvertTo
{
private:
  class Yes { char x; };
  class No {
    Yes m[2];
  };

  // if SourceType is convertable to TargetType then this overload will be
  // resolved when we call Deduce(SourceType)
  static Yes Deduce(TargetType);

  // If it is not convertible, then this overload will be resolved.
  static No  Deduce(...);

public:
  static constexpr bool value = sizeof(Deduce(*(SourceType*)0)) == sizeof(Yes);
};

}  // namespace kw

#endif // KWARGS_IS_CALL_POSSIBLE_H_
