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
 *  @file   include/cppfontconfig/StrList.h
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_STRLIST_H_
#define CPPFONTCONFIG_STRLIST_H_

#include <fontconfig/fontconfig.h>
#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/RefPtr.h>
#include <cpp_fontconfig/StrSet.h>

namespace fontconfig
{

class StrList;

/// used during enumeration to safely and correctly walk the list of strings
/// even while that list is edited in the middle of enumeration.
/**
 *  String iterators are not reference counted object and the StrList class is
 *  mearly a container for the pointer. It is safe to copy an StrList but
 *  be sure to only call destroy on one of the copies.
 *
 *  Also, since StrList is a wrapper for the pointer, you should probably
 *  only allocate an StrList on the stack.
 */
class StrListDelegate
{
    private:
        FcStrList* m_ptr;

        /// wrap constructor
        /**
         *  wraps the pointer with this interface, does nothing else, only
         *  called by RefPtr<Atomic>
         */
        explicit StrListDelegate(FcStrList* ptr):
            m_ptr(ptr)
        {}

        /// not copy-constructable
        StrListDelegate( const StrListDelegate& other );

        /// not assignable
        StrListDelegate& operator=( const StrListDelegate& other );

    public:
        friend class RefPtr<StrList>;

        StrListDelegate* operator->(){ return this; }
        const StrListDelegate* operator->() const { return this; }

        /// get next string in iteration
        /**
         *  Returns the next string in set.
         */
        Char8_t* next ();

        /// destroy a string iterator
        /**
         *  Destroys the enumerator list.
         */
        void done();
};


struct StrList
{
    typedef StrListDelegate Delegate;
    typedef FcStrList*      Storage;
    typedef FcStrList*      cobjptr;

    /// create a string iterator
    /**
     *  Creates an iterator to list the strings in set.
     */
    static RefPtr<StrList> create (RefPtr<StrSet> set);
};




} // namespace fontconfig 

#endif // STRLIST_H_
