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
 *  @file   src/StrList.cpp
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/StrList.h>
#include <fontconfig/fontconfig.h>

namespace fontconfig
{

RefPtr<StrList> StrList::create(RefPtr<StrSet> set)
{
    return FcStrListCreate( set.subvert() );
}

Char8_t* StrListDelegate::next()
{
    return FcStrListNext( m_ptr );
}

void StrListDelegate::done()
{
    FcStrListDone( m_ptr );
}

} // namespace fontconfig 
