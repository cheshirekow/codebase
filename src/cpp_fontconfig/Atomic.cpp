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
 *  @file   src/Atomic.cpp
 *
 *  \date   Jul 22, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/Atomic.h>
#include <fontconfig/fontconfig.h>

namespace fontconfig
{



bool AtomicDelegate::lock()
{
    return FcAtomicLock( m_ptr );
}

Char8_t* AtomicDelegate::newFile()
{
    return FcAtomicNewFile( m_ptr );
}

Char8_t* AtomicDelegate::origFile()
{
    return FcAtomicOrigFile( m_ptr );
}

bool AtomicDelegate::replaceOrig()
{
    return FcAtomicReplaceOrig( m_ptr );
}

void AtomicDelegate::deleteNew()
{
    return FcAtomicDeleteNew( m_ptr );
}

void AtomicDelegate::unlock()
{
    return FcAtomicUnlock( m_ptr );
}

void AtomicDelegate::destroy()
{
    return FcAtomicDestroy( m_ptr );
}

RefPtr<Atomic> Atomic::create(const Char8_t* file)
{
    return FcAtomicCreate(file);
}

} // namespace fontconfig 
