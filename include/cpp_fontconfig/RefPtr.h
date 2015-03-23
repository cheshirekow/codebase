/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of cppfontconfig.
 *
 *  cppfontconfig is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cppfontconfig is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cppfontconfig.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   include/cppfontconfig/RefPtr.h
 *
 *  @date   Feb 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef CPPFONTCONFIG_REFPTR_H_
#define CPPFONTCONFIG_REFPTR_H_

#include <cpp_fontconfig/AssignmentPair.h>
#include <cpp_fontconfig/CPtr.h>
#include <iostream>

namespace fontconfig {


/// object which acts like a c-pointer, but when dereferenced returns a
/// delegate object which adds methods to the pointer
/**
 *  @note   If the pointer is a reference counted object then the reference
 *          count is managed by this class and so long as pointer safety
 *          is not subverted all reference counted objects will be freed
 *          automatically when their references are all destroyed.
 */
template< class Traits >
class RefPtr
{
    public:
        typedef typename Traits::cobjptr    cobjptr;
        typedef typename Traits::Delegate   Delegate;
        typedef typename Traits::Storage    Storage;

    private:
        Storage m_ptr;

        /// increase reference count by one, see specializations
        void reference(){}

        /// decrease reference count by one, see specializations
        void dereference(){}


    public:
        /// create a RefPtr from the specified cobj
        /**
         *  @param ptr      the c-obj pointer to wrap
         *  @param doRef    whether or not to increase the reference count
         *
         *  @note since fontconfig sometimes gives us a pointer which already has
         *        a reference count of 1, @p reference defaults to false.
         */
        RefPtr( cobjptr ptr=0, bool doRef=false ):
            m_ptr(ptr)
        {
            if(doRef)
                reference();
        }

        /// copy construct a pointer, increasing the reference count
        RefPtr( const RefPtr<Traits>& other ):
            m_ptr(other.m_ptr)
        {
            reference();
        }

        /// when the RefPtr is destroyed the reference count of the pointed-to
        /// object is decreased
        ~RefPtr()
        {
            dereference();
        }

        /// dereference the stored object and turn this into a null pointer
        void unlink()
        {
            dereference();
            m_ptr = 0;
        }

        /// return the stored pointer, subverting reference safety, see
        /// specializations if Storage is not the same as cobjptr
        cobjptr subvert()
        {
            return m_ptr;
        }

        /// return the stored pointer, subverting reference safety, see
        /// specializations if Storage is not the same as cobjptr
        const cobjptr subvert() const
        {
            return m_ptr;
        }

        /// assignment operator, decreases reference count of current object,
        /// increases reference count of copied pointer
        RefPtr<Traits>& operator=( const RefPtr<Traits>& other )
        {
            dereference();
            m_ptr = other.m_ptr;
            reference();
            return *this;
        }

        /// assignment operator, decreases reference count of current object,
        /// increases reference count of copied pointer
        RefPtr<Traits>& operator=( cobjptr ptr )
        {
            dereference();
            m_ptr = ptr;
            reference();
            return *this;
        }

        /// returns a delegate object which exposes member functions of
        /// the underlying object
        Delegate operator->()
        {
            return Delegate(m_ptr);
        }

        /// returns a delegate object which exposes member functions of
        /// the underlying object
        const Delegate operator->() const
        {
            return Delegate(m_ptr);
        }

        /// returns a delegate object which acts exactly like a c-pointer but
        /// cannot be copied and so reference counting cannot be subverted
        CPtr<Traits> operator*()
        {
            return CPtr<Traits>(m_ptr);
        }

        /// returns a delegate object which acts exactly like a c-pointer but
        /// cannot be copied and so reference counting cannot be subverted
        ConstCPtr<Traits> operator*() const
        {
            return ConstCPtr<Traits>(m_ptr);
        }

        /// exposes the boolean interpretation of the underlying pointer
        operator bool() const
        {
            return m_ptr;
        }

        /// can be paired with other objects for multiple (tuple) returns
        template <typename T2>
        LValuePair< RefPtr<Traits>,T2 > operator,( T2& other )
        {
            return LValuePair< RefPtr<Traits>,T2 >(*this,other);
        }

};


} // namespace cppfontconfig




#endif // REFPTR_H_
