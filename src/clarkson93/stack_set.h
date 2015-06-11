/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of clarkson93.
 *
 *  clarkson93 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  clarkson93 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with clarkson93.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef CLARKSON93_STATIC_STACK_H_
#define CLARKSON93_STATIC_STACK_H_

#include <cassert>
#include <list>
#include <type_traits>

#include <clarkson93/bit_member.h>

namespace clarkson93 {

/// Maintain a stack of bitset-able objects. When items are pushed onto the
/// stack they are added to the set, when they are popped from the stack they
/// are removed from the set
template <typename T, typename SetEnum = void>
class StackSet : public std::list<T> {
 public:
  StackSet(SetEnum which_set = static_cast<SetEnum>(0))
      : which_set_(which_set) {
    // TODO(josh): figure out how to do this since we don't know the size of
    // the enum list
    // static_assert(std::is_base_of<BitMember<SetEnum>, T>::value,
    //               "Items in a StackSet must derive from BitMember");
  }

  ~StackSet() {}

  /// Change the bit-field for the set that items in this stack are added to.
  /// Only valid if the stack is empty.
  void ChangeSetBit(SetEnum which_set) {
    assert(this->empty());
    which_set_ = which_set;
  }

  /// Return the element at the top of the stack, remove it from the stack, and
  /// mark it as not part of the set
  T Pop() {
    assert(!this->empty());
    T return_me = this->back();
    return_me.RemoveFrom(which_set_);
    this->pop_back();
    return return_me;
  }

  /// Put an element in the top of the stack.
  template <class... Args>
  void Push(Args&&... args) {
    this->emplace_back(args...);
    this->back().AddTo(which_set_);
  }

  /// Check if an element is a member of the set in this stack
  bool IsMember(const T& obj) const {
    return obj.isMemberOf(which_set_);
  }

 private:
  SetEnum which_set_;
};

/// Maintain a stack of bitset-able objects. When items are pushed onto the
/// stack they are added to the set, when they are popped from the stack they
/// are removed from the set,
/**
 *  TODO(josh): No need to template on both element type and set enumerator
 *  type... just use partial specialization to get enum from base class.
 */
template <typename T, typename SetEnum>
class StackSet<T*, SetEnum> : public std::list<T*> {
 public:
  StackSet(SetEnum which_set = static_cast<SetEnum>(0))
      : which_set_(which_set) {}

  ~StackSet() {
    Clear();
  }

  /// Change the bit-field for the set that items in this stack are added to.
  /// Only valid if the stack is empty.
  void ChangeSetBit(SetEnum which_set) {
    assert(this->empty());
    which_set_ = which_set;
  }

  /// Return the element at the top of the stack, remove it from the stack, and
  /// mark it as not part of the set
  T* Pop() {
    assert(!this->empty());
    T* return_me = this->back();
    return_me->RemoveFrom(which_set_);
    this->pop_back();
    return return_me;
  }

  /// Put an element in the top of the stack.
  void Push(T* obj) {
    obj->AddTo(which_set_);
    this->push_back(obj);
  }

  /// Remove all objects from the stack and also remove them from the set
  /// representing stack membership.
  void Clear() {
    for (T* obj : *this) {
      obj->RemoveFrom(which_set_);
    }
    this->clear();
  }

  /// Check if an element is a member of the set in this stack
  bool IsMember(const T* obj) const {
    return obj->IsMemberOf(which_set_);
  }

 private:
  SetEnum which_set_;
};

/// A simple stack data structure with a slightly more convenient interface
template <typename T>
struct Stack : public std::list<T> {
  T Pop() {
    assert(!this->empty());
    T return_me = this->back();
    this->pop_back();
    return return_me;
  }

  template <class... Args>
  void Push(Args&&... args) {
    this->emplace_back(args...);
  }
};

}  // namespace clarkson93

#endif  // CLARKSON93_STATIC_STACK_H_
