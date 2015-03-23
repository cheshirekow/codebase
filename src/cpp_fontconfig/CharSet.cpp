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
 *  @file   src/CharSet.cpp
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/CharSet.h>
#include <fontconfig/fontconfig.h>

namespace fontconfig
{


template<>
void RefPtr<CharSet>::reference()
{
    if(m_ptr)
        m_ptr = FcCharSetCopy( m_ptr );
}

template<>
void RefPtr<CharSet>::dereference()
{
    if(m_ptr)
        FcCharSetDestroy( m_ptr );
}





bool CharSetDelegate::addChar(Char32_t ucs4)
{
    return FcCharSetAddChar( m_ptr, ucs4 );
}

bool CharSetDelegate::delChar(Char32_t ucs4)
{
    return FcCharSetDelChar( m_ptr, ucs4 );
}

bool CharSetDelegate::equal(const RefPtr<CharSet>& other) const
{
    return FcCharSetEqual( m_ptr, other.subvert() );
}

RefPtr<CharSet> CharSetDelegate::intersect(const RefPtr<CharSet>& other)
{
    return RefPtr<CharSet>(
            FcCharSetIntersect( m_ptr, other.subvert() ) );
}

RefPtr<CharSet> CharSetDelegate::createUnion(const RefPtr<CharSet>& other)
{
    return RefPtr<CharSet>(
            FcCharSetUnion(m_ptr, other.subvert() ) );
}

RefPtr<CharSet> CharSetDelegate::subtract(const RefPtr<CharSet>& other)
{
    return RefPtr<CharSet>(
            FcCharSetSubtract( m_ptr, other.subvert() ) );
}

bool CharSetDelegate::merge(const RefPtr<CharSet>& other, bool& changed)
{
    FcBool changed2;
    bool result = FcCharSetMerge( m_ptr, other.subvert(), &changed2 );
    changed = changed2;
    return result;
}

bool CharSetDelegate::merge(const RefPtr<CharSet>& other)
{
    return FcCharSetMerge( m_ptr, other.subvert(), 0 );
}

bool CharSetDelegate::hasChar(Char32_t ucs4) const
{
    return FcCharSetHasChar( m_ptr, ucs4 );
}

Char32_t CharSetDelegate::count() const
{
    return FcCharSetCount( m_ptr );
}

Char32_t CharSetDelegate::intersectCount(const RefPtr<CharSet>& other)
{
    return FcCharSetIntersectCount( m_ptr, other.subvert() );
}

Char32_t CharSetDelegate::subtractCount(const RefPtr<CharSet>& other)
{
    return FcCharSetSubtractCount( m_ptr, other.subvert() );
}

bool CharSetDelegate::isSubset(const RefPtr<CharSet>& other) const
{
    return FcCharSetIsSubset( m_ptr, other.subvert() );
}

Char32_t CharSetDelegate::firstPage(Char32_t map[MAP_SIZE], Char32_t* next)
{
    return FcCharSetFirstPage( m_ptr, map, next );
}

Char32_t CharSetDelegate::nextPage(Char32_t map[MAP_SIZE], Char32_t* next)
{
    return FcCharSetNextPage( m_ptr, map, next );
}

RefPtr<CharSet> CharSet::create(void)
{
    return RefPtr<CharSet> ( FcCharSetCreate() );
}

} // namespace fontconfig 
