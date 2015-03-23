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
 *  @file   include/cppfontconfig/CharSet.h
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_CHARSET_H_
#define CPPFONTCONFIG_CHARSET_H_

#include <fontconfig/fontconfig.h>
#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/RefPtr.h>
#include <unistd.h>

namespace fontconfig
{

class CharSet;


/// An CharSet is an abstract type that holds the set of encoded Unicode
/// chars in a font. Operations to build and compare these sets are provided.
/**
 *  The underlying FcCharSet object is reference counted, so this CharSet
 *  wrapper provides a copy constructor which increments the reference count,
 *  and a destructor which decrements the reference count
 */
class CharSetDelegate
{
    private:
        FcCharSet* m_ptr;

        /// wrap constructor
        /**
         *  wraps the pointer with this interface, does nothing else, only
         *  called by RefPtr<Atomic>
         */
        explicit CharSetDelegate(FcCharSet* ptr):
            m_ptr(ptr)
        {}

        /// not copy-constructable
        CharSetDelegate( const CharSetDelegate& other );

        /// not assignable
        CharSetDelegate& operator=( const CharSetDelegate& other );

    public:
        friend class RefPtr<CharSet>;

        CharSetDelegate* operator->(){ return this; }
        const CharSetDelegate* operator->() const { return this; }

        static const unsigned int MAP_SIZE = 256/32;

        /// Add a character to a charset
        /**
         *  FcCharSetAddChar adds a single Unicode char to the set, returning
         *  FcFalse on failure, either as a result of a constant set or from
         *  running out of memory.
         */
        bool addChar (Char32_t ucs4);

        /// Remove a character from a charset
        /**
         *  FcCharSetDelChar deletes a single Unicode char from the set,
         *  returning FcFalse on failure, either as a result of a constant set
         *  or from running out of memory.
         */
        bool delChar (Char32_t ucs4);

        /// Compare two charsets
        /**
         *  Returns whether a and b contain the same set of Unicode chars.
         */
        bool equal (const RefPtr<CharSet>& other) const;

        /// Intersect charsets
        /**
         *  Returns a set including only those chars found in both a and b.
         */
        RefPtr<CharSet> intersect (const RefPtr<CharSet>& other);

        /// Add charsets
        /**
         *  Returns a set including only those chars found in either a or b.
         */
        RefPtr<CharSet> createUnion(const RefPtr<CharSet>& other);

        /// Subtract charsets
        /**
         *  Returns a set including only those chars found in a but not b.
         */
        RefPtr<CharSet> subtract (const RefPtr<CharSet>& other);

        /// Merge charsets
        /**
         *  Adds all chars in b to a. In other words, this is an in-place
         *  version of FcCharSetUnion. If changed is not NULL, then it returns
         *  whether any new chars from b were added to a. Returns FcFalse on
         *  failure, either when a is a constant set or from running out of
         *  memory.
         */
        bool merge (const RefPtr<CharSet>& other, bool& changed);
        bool merge (const RefPtr<CharSet>& other);

        /// Check a charset for a char
        /**
         *  Returns whether fcs contains the char ucs4.
         */
        bool hasChar (Char32_t ucs4) const;

        /// Count entries in a charset
        /**
         *  Returns the total number of Unicode chars in a.
         */
        Char32_t count () const;

        /// Intersect and count charsets
        /**
         *  Returns the number of chars that are in both a and b.
         */
        Char32_t intersectCount (const RefPtr<CharSet>& other);

        /// Subtract and count charsets
        /**
         *  Returns the number of chars that are in a but not in b.
         */
        Char32_t subtractCount (const RefPtr<CharSet>& other);

        /// Test for charset inclusion
        /**
         *  Returns whether a is a subset of b.
         */
        bool isSubset (const RefPtr<CharSet>& other) const;

        /// Start enumerating charset contents
        /**
         *  Builds an array of bits marking the first page of Unicode coverage
         *  of a. Returns the base of the array. next contains the next page
         *  in the font.
         */
        Char32_t firstPage (
                Char32_t        map[MAP_SIZE],
                Char32_t        *next);

        /// Continue enumerating charset contents
        /**
         *  Builds an array of bits marking the Unicode coverage of a for page
         *  *next. Returns the base of the array. next contains the next page
         *  in the font.
         */
        Char32_t nextPage (
               Char32_t     map[MAP_SIZE],
               Char32_t     *next);
};

/// Traits class for a charset.
/// An CharSet is an abstract type that holds the set of encoded Unicode
/// chars in a font. Operations to build and compare these sets are provided.
/**
 *  The underlying FcCharSet object is reference counted, so this CharSet
 *  wrapper provides a copy constructor which increments the reference count,
 *  and a destructor which decrements the reference count
 */
struct CharSet
{
    typedef CharSetDelegate Delegate;
    typedef FcCharSet*      Storage;
    typedef FcCharSet*      cobjptr;

    /// Create an empty character set
    /**
     *  FcCharSetCreate allocates and initializes a new empty character
     *  set object.
     */
    static RefPtr<CharSet> create (void);
};

/// CharSets are reference counted
template <> void RefPtr<CharSet>::reference();

/// CharSets are reference counted
template <> void RefPtr<CharSet>::dereference();


} // namespace fontconfig 

#endif // CHARSET_H_
