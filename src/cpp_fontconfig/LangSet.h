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
 *  @file   include/cppfontconfig/LangSet.h
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_LANGSET_H_
#define CPPFONTCONFIG_LANGSET_H_

#include <fontconfig/fontconfig.h>
#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/RefPtr.h>
#include <cpp_fontconfig/StrSet.h>

namespace fontconfig
{

class LangSet;

/// An LangSet is an abstract type that holds the set of languages supported
/// by a font.
/**
 *  Operations to build and compare these sets are provided. These are computed
 *  for a font based on orthographic information built into the fontconfig
 *  library. Fontconfig has orthographies for all of the ISO 639-1 languages
 *  except for MS, NA, PA, PS, QU, RN, RW, SD, SG, SN, SU and ZA. If you have
 *  orthographic information for any of these languages, please submit them.
 */
class LangSetDelegate
{
    private:
        FcLangSet* m_ptr;

        /// wrap constructor
        /**
         *  wraps the pointer with this interface, does nothing else, only
         *  called by RefPtr<Atomic>
         */
        explicit LangSetDelegate(FcLangSet* ptr):
            m_ptr(ptr)
        {}

        /// not copy-constructable
        LangSetDelegate( const LangSetDelegate& other );

        /// not assignable
        LangSetDelegate& operator=( const LangSetDelegate& other );

    public:
        friend class RefPtr<LangSet>;

        LangSetDelegate* operator->(){ return this; }
        const LangSetDelegate* operator->() const { return this; }

        /// destroy a langset object
        /**
         *  FcLangSetDestroy destroys a FcLangSet object, freeing all memory
         *  associated with it.
         */
        void  destroy ();

        /// copy a langset object
        /**
         *  FcLangSetCopy creates a new FcLangSet object and populates it with
         *  the contents of ls.
         */
        RefPtr<LangSet>  copy ();

        /// add a language to a langset
        /**
         *  lang is added to ls. lang should be of the form Ll-Tt where Ll is
         *  a two or three letter language from ISO 639 and Tt is a territory
         *  from ISO 3166.
         */
        bool  add (const Char8_t *lang);

        /// delete a language from a langset
        /**
         *  lang is removed from ls. lang should be of the form Ll-Tt where Ll
         *  is a two or three letter language from ISO 639 and Tt is a
         *  territory from ISO 3166.
         */
        bool  del (const Char8_t *lang);

        /// test langset for language support
        /**
         *  FcLangSetHasLang checks whether ls supports lang. If ls has a
         *  matching language and territory pair, this function returns
         *  FcLangEqual. If ls has a matching language but differs in which
         *  territory that language is for, this function returns
         *  FcLangDifferentTerritory. If ls has no matching language, this
         *  function returns FcLangDifferentLang.
         */
        LangResult_t  hasLang (const Char8_t *lang);

        /// compare language sets
        /**
         *  FcLangSetCompare compares language coverage for ls_a and ls_b. If
         *  they share any language and territory pair, this function returns
         *  FcLangEqual. If they share a language but differ in which
         *  territory that language is for, this function returns
         *  FcLangDifferentTerritory. If they share no languages in common,
         *  this function returns FcLangDifferentLang.
         */
        LangResult_t  compare (const RefPtr<LangSet> lsb);

        /// check langset subset relation
        /**
         *  FcLangSetContains returns FcTrue if ls_a contains every language
         *  in ls_b. ls_a will 'contain' a language from ls_b if ls_a has
         *  exactly the language, or either the language or ls_a has no
         *  territory.
         */
        bool  contains (const RefPtr<LangSet> lsb);

        /// test for matching langsets
        /**
         *  Returns FcTrue if and only if ls_a supports precisely the same
         *  language and territory combinations as ls_b.
         */
        bool  equal (const RefPtr<LangSet> lsb);

        /// return a hash value for a langset
        /**
         *  This function returns a value which depends solely on the
         *  languages supported by ls. Any language which equals ls will have
         *  the same result from FcLangSetHash. However, two langsets with the
         *  same hash value may not be equal.
         */
        Char32_t  hash ();

        /// Get list of languages
        /**
         *
         */
        RefPtr<StrSet> getLangs ();

        /// Add langsets
        /**
         *  Returns a set including only those languages found in either
         *  ls_a or ls_b.
         */
        RefPtr<LangSet>  creatUnion (const RefPtr<LangSet> b);

        /// Subtract langsets
        /**
         *  Returns a set including only those languages found in ls_a but
         *  not in ls_b.
         */
        RefPtr<LangSet>  subtract (const RefPtr<LangSet> b);
};


struct LangSet
{
    typedef LangSetDelegate Delegate;
    typedef FcLangSet*      Storage;
    typedef FcLangSet*      cobjptr;

    /// create a langset object
    /**
     *  FcLangSetCreate creates a new FcLangSet object.
     */
    static RefPtr<LangSet>  create (void);


};


} // namespace fontconfig 

#endif // LANGSET_H_
