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
 *  @file   src/String.cpp
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/String.h>
#include <fontconfig/fontconfig.h>
#include <cassert>

namespace fontconfig
{

template <typename Ptr_t>
String<Ptr_t>::String(Ptr_t ptr):
    m_ptr(ptr)
{

}


template <>
Char8_t* String<Char8_t*>::get_mutable_ptr()
{
    return m_ptr;
}

template <>
Char8_t* String<const Char8_t*>::get_mutable_ptr()
{
    assert(0 == "ConstString_t cannot access a mutable pointer");
    return 0;
}

template <typename Ptr_t>
String_t String<Ptr_t>::copy() const
{
    return String_t( FcStrCopy(m_ptr) );
}

template <typename Ptr_t>
String_t String<Ptr_t>::copyFilename() const
{
    return String( FcStrCopyFilename(m_ptr) );
}

template <typename Ptr_t>
template <typename OtherPtr_t>
String_t String<Ptr_t>::plus(const String<OtherPtr_t>& other) const
{
    return String_t( FcStrPlus(m_ptr, other.m_ptr) );
}

template <>
void String<Char8_t*>::free()
{
    FcStrFree(m_ptr);
}

template <>
void String<const Char8_t*>::free()
{
    assert(0 == "ConstString_t cannot be free'd");
}

template <typename Ptr_t>
String_t String<Ptr_t>::downcase() const
{
    return String_t(FcStrDowncase(m_ptr) );
}

template <typename Ptr_t>
template <typename OtherPtr_t>
int String<Ptr_t>::cmpIgnoreCase(const String<OtherPtr_t>& other)
{
    return FcStrCmpIgnoreCase( m_ptr, other.m_ptr );
}

template <typename Ptr_t>
template <typename OtherPtr_t>
int String<Ptr_t>::cmp(const String<OtherPtr_t>& other)
{
    return FcStrCmp( m_ptr, other.m_ptr );
}

template <typename Ptr_t>
template <typename OtherPtr_t>
ConstString_t String<Ptr_t>::findIgnoreCase(const String<OtherPtr_t>& needle)
{
    return ConstString_t( FcStrStrIgnoreCase(needle.m_ptr, m_ptr) );
}

template <typename Ptr_t>
template <typename OtherPtr_t>
ConstString_t String<Ptr_t>::find(const String<OtherPtr_t>& needle)
{
    return ConstString_t( FcStrStr(needle.m_ptr, m_ptr) );
}

template <typename Ptr_t>
String_t String<Ptr_t>::dirName() const
{
    return String_t( FcStrDirname(m_ptr) );
}

template <typename Ptr_t>
String_t String<Ptr_t>::baseName() const
{
    return String_t( FcStrBasename(m_ptr) );
}


} // namespace fontconfig 
