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
 *  @file   src/StrSet.cpp
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/StrSet.h>
#include <fontconfig/fontconfig.h>

namespace fontconfig
{

RefPtr<StrSet> StrSet::create(void)
{
    return FcStrSetCreate();
}


bool StrSetDelegate::setMember(const Char8_t* s)
{
    return FcStrSetMember( m_ptr, s );
}

bool StrSetDelegate::equal(RefPtr<StrSet> other)
{
    return FcStrSetEqual( m_ptr, other.subvert() );
}

bool StrSetDelegate::add(const Char8_t* s)
{
    return FcStrSetAdd( m_ptr, s );
}

bool StrSetDelegate::addFilename(const Char8_t* s)
{
    return FcStrSetAddFilename( m_ptr, s );
}

bool StrSetDelegate::del(const Char8_t* s)
{
    return FcStrSetDel( m_ptr, s );
}

void StrSetDelegate::destroy()
{
    return FcStrSetDestroy( m_ptr );
}

} // namespace fontconfig 
