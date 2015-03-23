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
 *  @file   src/LangSet.cpp
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/LangSet.h>
#include <fontconfig/fontconfig.h>

namespace fontconfig
{


RefPtr<LangSet> LangSet::create()
{
    return FcLangSetCreate( );
}

void LangSetDelegate::destroy()
{
    return FcLangSetDestroy( m_ptr );
}

RefPtr<LangSet> LangSetDelegate::copy()
{
    return FcLangSetCopy( m_ptr );
}

bool LangSetDelegate::add(const Char8_t* lang)
{
    return FcLangSetAdd( m_ptr, lang);
}

bool LangSetDelegate::del(const Char8_t* lang)
{
    return FcLangSetDel( m_ptr, lang);
}

LangResult_t LangSetDelegate::hasLang(const Char8_t* lang)
{
    return (LangResult_t) FcLangSetHasLang( m_ptr, lang);
}

LangResult_t LangSetDelegate::compare(const RefPtr<LangSet> lsb)
{
    return (LangResult_t) FcLangSetCompare( m_ptr, lsb.subvert());
}

bool LangSetDelegate::contains(const RefPtr<LangSet> lsb)
{
    return FcLangSetContains( m_ptr, lsb.subvert());
}

bool LangSetDelegate::equal(const RefPtr<LangSet> lsb)
{
    return FcLangSetEqual( m_ptr, lsb.subvert());
}

Char32_t LangSetDelegate::hash()
{
    return FcLangSetHash( m_ptr );
}

RefPtr<StrSet> LangSetDelegate::getLangs()
{
    return  FcLangSetGetLangs( m_ptr );
}

RefPtr<LangSet> LangSetDelegate::creatUnion(const RefPtr<LangSet> b)
{
    return FcLangSetUnion( m_ptr, b.subvert() );
}

RefPtr<LangSet> LangSetDelegate::subtract(const RefPtr<LangSet> b)
{
    return FcLangSetSubtract( m_ptr, b.subvert() );
}

} // namespace fontconfig 
