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

/// pointer ot a reference counted object, auto destruct when reference
/// count is zero
template<class Obj>
class RefPtr {
 private:
  Obj* m_ptr;

  void Reference() {
    if (m_ptr)
      static_cast<RefCounted*>(m_ptr)->Reference();
  }

  void Dereference() {
    if (m_ptr)
      if (static_cast<RefCounted*>(m_ptr)->Dereference())
        delete m_ptr;
  }

 public:
  template<class Other> friend class RefPtr;

  RefPtr(Obj* ptr = 0)
      : m_ptr(ptr) {
    Reference();
  }

  ~RefPtr() {
    Dereference();
  }

  void Unlink() {
    Dereference();
    m_ptr = 0;
  }

  int RefCount() const {
    return static_cast<const RefCounted*>(m_ptr)->GetRefCount();
  }

  RefPtr<Obj>& operator=(RefPtr<Obj> other) {
    Dereference();
    m_ptr = other.m_ptr;
    Reference();
    return *this;
  }

  Obj* operator->() {
    return m_ptr;
  }

  const Obj* operator->() const {
    return m_ptr;
  }

  Obj& operator*() {
    return *m_ptr;
  }

  const Obj& operator*() const {
    return *m_ptr;
  }

  operator bool() const {
    return m_ptr;
  }

};

}  // namespace gltk

#endif // GLTK_REFPTR_H_
