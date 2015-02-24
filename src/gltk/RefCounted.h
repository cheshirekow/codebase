/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
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
 *  @file   /home/josh/Codes/cpp/gltk/src/RefCounted.h
 *
 *  @date   Feb 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef GLTK_REFCOUNTED_H_
#define GLTK_REFCOUNTED_H_

namespace gltk {

/// base class for objects which are reference counted
class RefCounted
{
    private:
        int m_refCount;

    public:
        /// initializes reference count to 0
        RefCounted();

        /// increase reference count by 1
        void reference();

        /// decrease reference count by 1, return true if reference count
        /// is zero
        bool dereference();

        int getRefCount() const;
};


} // namespace gltk


#endif // REFCOUNTED_H_
