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
 *  @file   include/cppfontconfig/String.h
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_STRING_H_
#define CPPFONTCONFIG_STRING_H_

#include <cpp_fontconfig/common.h>

namespace fontconfig
{

template <typename Ptr_t>
class StringOther;

template<>
struct StringOther<Char8_t*>
{
    typedef const Char8_t* Type;
};

template<>
struct StringOther<const Char8_t*>
{
    typedef Char8_t* Type;
};


template <typename Ptr_t>
class String
{
    public:
        typedef String<Ptr_t>           This_t;
        typedef String<Char8_t*>        String_t;
        typedef String<const Char8_t*>  ConstString_t;

        typedef typename StringOther<Ptr_t>::Type      Other_t;

    private:
        Ptr_t m_ptr;

    public:
        String(Ptr_t ptr);

        Char8_t* get_mutable_ptr();
        const Char8_t* get_const_ptr() const;

        String_t copy() const;
        String_t copyFilename() const;

        template <typename OtherPtr_t>
        String_t plus(const String<OtherPtr_t>& other) const;
        void   free();
        String_t downcase() const;

        template <typename OtherPtr_t>
        int    cmpIgnoreCase(const String<OtherPtr_t>& other);

        template <typename OtherPtr_t>
        int    cmp(const String<OtherPtr_t>& other);

        template <typename OtherPtr_t>
        ConstString_t findIgnoreCase(const String<OtherPtr_t>& needle);

        template <typename OtherPtr_t>
        ConstString_t find(const String<OtherPtr_t>& needle);

        String_t dirName() const;
        String_t baseName() const;
};


typedef String<Char8_t*>        String_t;
typedef String<const Char8_t*>  ConstString_t;






} // namespace fontconfig 

#endif // STRING_H_
