#include <gtest/gtest.h>

#include <array>
#include <list>
#include <set>
#include <vector>
#include "container_id.h"

using namespace tap;

int main() {
  static_assert(!has_push_back<std::array<int, 3>>::result,
                "std::array shouldn't have push_back");
  static_assert(has_push_back<std::list<int>>::result,
                "std::list should have push_back");
  static_assert(!has_push_back<std::set<int>>::result,
                "std::set shouldn't have push_back");
  static_assert(has_push_back<std::vector<int>>::result,
                "std::vector should have push_back");

  static_assert(!can_push_back<std::array<int, 3>, void(int)>::value,
                "std::array<array> shouldn't be able to push back");
  static_assert(can_push_back<std::list<int>, void(int)>::value,
                "std::list<int> should be able to push back");
  static_assert(!can_push_back<std::set<int>, void(int)>::value,
                "std::set<int> shouldn't be able to push back");
  static_assert(can_push_back<std::vector<int>, void(int)>::value,
                "std::vector<int> should be able to push back");

  static_assert(!has_insert<std::array<int, 3>>::result,
                "std::array shouldn't have insert");
  static_assert(has_insert<std::list<int>>::result,
                "std::list should have insert");
  static_assert(has_insert<std::set<int>>::result,
                "std::set should have insert");
  static_assert(has_insert<std::vector<int>>::result,
                "std::vector should have insert");

  static_assert(!can_insert<std::array<int, 3>, void(int)>::value,
                "std::array<array> shouln't be able to insert");
  static_assert(
      !can_insert<std::list<int>, void(int)>::value,
      "std::list<int> shouldn't be able to insert without an iterator");
  static_assert(can_insert<std::set<int>, void(int)>::value,
                "std::set<int> should be able to insert");
  static_assert(
      !can_insert<std::vector<int>, void(int)>::value,
      "std::vector<int> shouldn't be able to insert without an iterator");

  static_assert(has_begin<std::array<int, 3>>::result,
                "std::array<array> should have begin");
  static_assert(has_begin<std::list<int>>::result,
                "std::list<int> should have begin");
  static_assert(has_begin<std::set<int>>::result,
                "std::set<int> should have_begin");
  static_assert(has_begin<std::vector<int>>::result,
                "std::vector<int> should have begin");

  static_assert(has_end<std::array<int, 3>>::result,
                "std::array<array> should have begin");
  static_assert(has_end<std::list<int>>::result,
                "std::list<int> should have begin");
  static_assert(has_end<std::set<int>>::result,
                "std::set<int> should have_begin");
  static_assert(has_end<std::vector<int>>::result,
                "std::vector<int> should have begin");

  static_assert(is_iterator<std::array<int, 3>::iterator>::value,
                "std::array<array> iterator should be iterator");
  static_assert(is_iterator<std::list<int>::iterator>::value,
                "std::list<int> iterator should be iterator");
  static_assert(is_iterator<std::set<int>::iterator>::value,
                "std::set<int> iterator should be iterator");
  static_assert(is_iterator<std::vector<int>::iterator>::value,
                "std::vector<int> iterator should be iterator");

  static_assert(has_iterators<std::array<int, 3>>::value,
                "std::array<array> iterator should be iterator");
  static_assert(has_iterators<std::list<int>>::value,
                "std::list<int> iterator should be iterator");
  static_assert(has_iterators<std::set<int>>::value,
                "std::set<int> iterator should be iterator");
  static_assert(has_iterators<std::vector<int>>::value,
                "std::vector<int> iterator should be iterator");

  static_assert(
      std::is_same<typename get_value_type<std::vector<int>>::value_type,
                   int>::value,
      "value type of a vector<int> should be int");
  static_assert(
      std::is_same<typename get_value_type<int>::value_type, int>::value,
      "value type of a int should be int");

  static_assert(is_cstyle_array<int(&)[3]>::value, "int[3] is a cstyle array");
  static_assert(!is_cstyle_array<int>::value, "int* is not a cstyle array");
}
