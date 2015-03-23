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
 *  @file   src/ObjectSet.cpp
 *
 *  \date   Jul 22, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/ObjectSet.h>
#include <fontconfig/fontconfig.h>

namespace fontconfig
{



bool ObjectSetDelegate::add(const char* obj)
{
    return FcObjectSetAdd( m_ptr, obj);
}

void ObjectSetDelegate::destroy()
{
    FcObjectSetDestroy(m_ptr);
}





ObjectSet::Builder::Builder( RefPtr<ObjectSet> objset ):
    m_objset(objset)
{

}

ObjectSet::Builder& ObjectSet::Builder::operator()( const char* object )
{
    m_objset->add(object);
    return *this;
}

RefPtr<ObjectSet> ObjectSet::Builder::done()
{
    return m_objset;
}


RefPtr<ObjectSet> ObjectSet::create()
{
    return FcObjectSetCreate();
}

RefPtr<ObjectSet> ObjectSet::build(const char* first, ...)
{

    va_list argp;
    va_start(argp, first);
    RefPtr<ObjectSet> result( FcObjectSetVaBuild(first, argp) );
    va_end(argp);

    return result;
}






}// namespace fontconfig
