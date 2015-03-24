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
 *  @file   include/cppfontconfig/FontSet.h
 *
 *  \date   Jul 22, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_FONTSET_H_
#define CPPFONTCONFIG_FONTSET_H_

#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/Pattern.h>

namespace fontconfig
{

class FontSet;

/// An FcFontSet simply holds a list of patterns; these are used to return the
/// results of listing available fonts.
/**
 *  An FcFontSet contains a list of FcPatterns. Internally fontconfig uses
 *  this data structure to hold sets of fonts. Externally, fontconfig returns
 *  the results of listing fonts in this format. 'nfont' holds the number of
 *  patterns in the 'fonts' array; 'sfont' is used to indicate the size of
 *  that array.
 */
class FontSetDelegate
{
    private:
        FcFontSet* m_ptr;

        /// wrap constructor
        /**
         *  wraps the pointer with this interface, does nothing else, only
         *  called by RefPtr<Atomic>
         */
        explicit FontSetDelegate(FcFontSet* ptr):
            m_ptr(ptr)
        {}

        /// not copy-constructable
        FontSetDelegate( const FontSetDelegate& other );

        /// not assignable
        FontSetDelegate& operator=( const FontSetDelegate& other );

    public:
        friend class RefPtr<FontSet>;

        FontSetDelegate* operator->(){ return this; }
        const FontSetDelegate* operator->() const { return this; }


        /// Destroy a font set
        /**
         *  Destroys a font set. Note that this destroys any referenced
         *  patterns as well.
         */
        void destroy ();

        /// Add to a font set
        /**
         *  Adds a pattern to a font set. Note that the pattern is not copied
         *  before being inserted into the set. Returns FcFalse if the pattern
         *  cannot be inserted into the set (due to allocation failure).
         *  Otherwise returns FcTrue.
         */
        bool add(RefPtr<Pattern> font);

        /// Print a set of patterns to stdout
        void print();

};


/// An FcFontSet simply holds a list of patterns; these are used to return the
/// results of listing available fonts.
/**
 *  An FcFontSet contains a list of FcPatterns. Internally fontconfig uses
 *  this data structure to hold sets of fonts. Externally, fontconfig returns
 *  the results of listing fonts in this format. 'nfont' holds the number of
 *  patterns in the 'fonts' array; 'sfont' is used to indicate the size of
 *  that array.
 */
struct FontSet
{
    typedef FontSetDelegate Delegate;
    typedef FcFontSet*      Storage;
    typedef FcFontSet*      cobjptr;

    /// Create a font set
    /**
     *  Creates an empty font set.
     */
    static RefPtr<FontSet> create (void);
};

} // namespace fontconfig 

#endif // FONTSET_H_
