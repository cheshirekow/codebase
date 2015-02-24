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
 *  @file   /home/josh/Codes/cpp/gltk/src/RefPtr.h
 *
 *  @date   Feb 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef GLTK_REFPTR_H_
#define GLTK_REFPTR_H_

#include <gltk/RefCounted.h>

namespace gltk {

/// pointer ot a reference counted object, auto destruct when reference
/// count is zero
template< class Obj >
class RefPtr
{
    private:
        Obj* m_ptr;

        void reference()
        {
            if(m_ptr)
                static_cast<RefCounted*>(m_ptr)->reference();
        }

        void dereference()
        {
            if(m_ptr)
                if( static_cast<RefCounted*>(m_ptr)->dereference() )
                    delete m_ptr;
        }



    public:
        template <class Other> friend class RefPtr;

        RefPtr( Obj* ptr=0 ):
            m_ptr(ptr)
        {
            reference();
        }

        ~RefPtr()
        {
            dereference();
        }

        void unlink()
        {
            dereference();
            m_ptr = 0;
        }

        int refCount() const
        {
            return static_cast<const RefCounted*>(m_ptr)->getRefCount();
        }

        RefPtr<Obj>& operator=( RefPtr<Obj> other )
        {
            dereference();
            m_ptr = other.m_ptr;
            reference();
            return *this;
        }

        Obj* operator->()
        {
            return m_ptr;
        }

        const Obj* operator->() const
        {
            return m_ptr;
        }

        Obj& operator*()
        {
            return *m_ptr;
        }

        const Obj& operator*() const
        {
            return *m_ptr;
        }

        operator bool() const
        {
            return m_ptr;
        }

};


} // namespace gltk




#endif // REFPTR_H_
