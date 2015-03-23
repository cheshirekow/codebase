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
 *  @file   include/cppfontconfig/Blanks.h
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_BLANKS_H_
#define CPPFONTCONFIG_BLANKS_H_

#include <fontconfig/fontconfig.h>
#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/RefPtr.h>

namespace fontconfig
{

class Blanks;

/// holds a list of Unicode chars which are expected to be blank
/**
 *  unexpectedly
 *  blank chars are assumed to be invalid and are elided from the charset
 *  associated with the font.
 *
 *  Blanks structures are not reference counted. It is safe to pass around
 *  copies of this object, however you must remember to call destroy on
 *  one and only one copy when you're done with it
 *
 *  It contains only one data member which is a pointer
 *  and the copy constructor will simply copy that pointer so there is no
 *  reason to allocate a Blank on the heap.
 */
class BlanksDelegate
{
    private:
        FcBlanks* m_ptr;

        /// wrap constructor
        /**
         *  wraps the pointer with this interface, does nothing else, only
         *  called by RefPtr<Atomic>
         */
        explicit BlanksDelegate(FcBlanks* ptr):
            m_ptr(ptr)
        {}

        /// not copy-constructable
        BlanksDelegate( const BlanksDelegate& other );

        /// not assignable
        BlanksDelegate& operator=( const BlanksDelegate& other );

    public:
        friend class RefPtr<Blanks>;

        BlanksDelegate* operator->(){ return this; }
        const BlanksDelegate* operator->() const { return this; }

        /// Destroys an FcBlanks object, freeing any associated memory.
        /**
         *  @see FcBlanksDestroy
         */
        void destroy();

        /// Add a character to an FcBlanks
        /**
         *  Adds a single character to an FcBlanks object, returning FcFalse if
         *  this process ran out of memory.
         *
         *  @see FcBlanksAdd
         */
        bool add( Char32_t ucs4 );

        /// Query membership in an FcBlanks
        /**
         *  Returns whether the specified FcBlanks object contains the
         *  indicated Unicode value.
         *  @see FcBlanksIsMember
         */
        bool isMember( Char32_t ucs4 );

};


/// traits class for FcBlanks. holds a list of Unicode chars which are
/// expected to be blank
/**
 *  unexpectedly
 *  blank chars are assumed to be invalid and are elided from the charset
 *  associated with the font.
 *
 *  Blanks structures are not reference counted. It is safe to pass around
 *  copies of this object, however you must remember to call destroy on
 *  one and only one copy when you're done with it
 *
 *  It contains only one data member which is a pointer
 *  and the copy constructor will simply copy that pointer so there is no
 *  reason to allocate a Blank on the heap.
 */
struct Blanks
{
    typedef BlanksDelegate   Delegate;
    typedef FcBlanks*        Storage;
    typedef FcBlanks*        cobjptr;

    /// Creates an empty FcBlanks oject
    /**
     *  @see FcBlanksCreate
     */
    static RefPtr<Blanks> create();
};

} // namespace fontconfig 

#endif // BLANKS_H_
