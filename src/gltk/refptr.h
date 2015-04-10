/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of gltk.
 *
 *  gltk is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gltk is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with gltk.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Feb 3, 2013
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */

#ifndef GLTK_REFPTR_H_
#define GLTK_REFPTR_H_

#include <gltk/refcounted.h>

namespace gltk {

/// pointer to a reference counted object, self-destruct when reference
/// count is zero.
/**
 *  This is a smart pointer for intrusive reference counting (as opposed to,
 *  for instance, shared_ptr).
 */
template<class Obj>
class RefPtr {
 private:
  Obj* ptr_;

  /// Increase reference count if we're not pointing to a nullptr
  void Reference() {
    if (ptr_) {
      static_cast<RefCounted*>(ptr_)->Reference();
    }
  }

  /// Decrease reference count and, if this is the last reference, then
  /// destroy the object
  void Dereference() {
    if (ptr_) {
      if (static_cast<RefCounted*>(ptr_)->Dereference()) {
        delete ptr_;
      }
    }
  }

 public:
  template<class Other> friend class RefPtr;

  RefPtr(Obj* ptr = nullptr)
      : ptr_(ptr) {
    Reference();
  }

  ~RefPtr() {
    Dereference();
  }

  void Unlink() {
    Dereference();
    ptr_ = nullptr;
  }

  int RefCount() const {
    return static_cast<const RefCounted*>(ptr_)->GetRefCount();
  }

  RefPtr<Obj>& operator=(const RefPtr<Obj>& other) {
    Dereference();
    ptr_ = other.ptr_;
    Reference();
    return *this;
  }

  Obj* operator->() {
    return ptr_;
  }

  const Obj* operator->() const {
    return ptr_;
  }

  Obj& operator*() {
    return *ptr_;
  }

  const Obj& operator*() const {
    return *ptr_;
  }

  operator bool() const {
    return ptr_;
  }
};

}  // namespace gltk

#endif  // GLTK_REFPTR_H_
