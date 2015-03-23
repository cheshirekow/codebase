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
 *  @file   src/name.cpp
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/name.h>
#include <fontconfig/fontconfig.h>
#include <cassert>

namespace fontconfig
{


ObjectTypeList::BuildToken ObjectTypeList::sm_seed;

ObjectTypeList::ObjectTypeList():
    m_ptr(0),
    m_nItems(0)
{

}

ObjectTypeList::ObjectTypeList( ObjectTypeList::BuildToken& token):
    m_ptr(token.m_ptr),
    m_nItems(token.m_nItems)
{

}

ObjectTypeList::ObjectTypeList(const ObjectTypeList& other):
    m_ptr(other.m_ptr),
    m_nItems(other.m_nItems)
{

}

FcObjectType* ObjectTypeList::get_ptr()
{
    return m_ptr;
}

const FcObjectType* ObjectTypeList::get_ptr() const
{
    return m_ptr;
}

int ObjectTypeList::get_nItems() const
{
    return m_nItems;
}

void ObjectTypeList::destroy()
{
    if(m_ptr)
        delete [] m_ptr;
}


ObjectTypeList::Item ObjectTypeList::create()
{
    sm_seed.init();
    return Item(sm_seed);
}



void ObjectTypeList::BuildToken::init()
{
    m_ptr    = 0;
    m_iItem  = 0;
    m_nItems = 0;
}


void ObjectTypeList::BuildToken::incrementCount()
{
    assert( !m_ptr );
    m_nItems++;
}

void ObjectTypeList::BuildToken::allocate()
{
    // there's one extra item due to the Item created by () at the end of
    // the list
    --m_nItems;

    assert( !m_ptr );
    m_ptr = new FcObjectType[m_nItems];
}

void ObjectTypeList::BuildToken::write( const char* object, Type_t type )
{
    assert( m_iItem < m_nItems );

    FcObjectType* ptr = (FcObjectType*)m_ptr;
    ptr[m_iItem].object = object;
    ptr[m_iItem].type   = (FcType)type;
    m_iItem++;
}




ObjectTypeList::Item::Item( ObjectTypeList::BuildToken& token ):
    m_token(token),
    m_object(0),
    m_type(type::Void),
    m_isLast(false)
{
    m_token.incrementCount();
}

ObjectTypeList::Item::~Item()
{
    if(!m_isLast)
        m_token.write(m_object,m_type);
}

ObjectTypeList::Item ObjectTypeList::Item::operator()( const char* object, Type_t type )
{
    m_object = object;
    m_type   = type;
    return Item(m_token);
}

ObjectTypeList::BuildToken& ObjectTypeList::Item::operator()()
{
    m_isLast = true;
    m_token.allocate();
    return m_token;
}








ConstantList::BuildToken ConstantList::sm_seed;

ConstantList::ConstantList():
    m_ptr(0),
    m_nItems(0)
{

}

ConstantList::ConstantList( ConstantList::BuildToken& token):
    m_ptr(token.m_ptr),
    m_nItems(token.m_nItems)
{

}

ConstantList::ConstantList(const ConstantList& other):
    m_ptr(other.m_ptr),
    m_nItems(other.m_nItems)
{

}

FcConstant* ConstantList::get_ptr()
{
    return m_ptr;
}

const FcConstant* ConstantList::get_ptr() const
{
    return m_ptr;
}

int ConstantList::get_nItems() const
{
    return m_nItems;
}

void ConstantList::destroy()
{
    if(m_ptr)
        delete [] m_ptr;
}


ConstantList::Item ConstantList::create()
{
    sm_seed.init();
    return Item(sm_seed);
}



void ConstantList::BuildToken::init()
{
    m_ptr    = 0;
    m_iItem  = 0;
    m_nItems = 0;
}


void ConstantList::BuildToken::incrementCount()
{
    assert( !m_ptr );
    m_nItems++;
}

void ConstantList::BuildToken::allocate()
{
    // there's one extra item due to the Item created by () at the end of
    // the list
    --m_nItems;

    assert( !m_ptr );
    m_ptr = new FcConstant[m_nItems];
}

void ConstantList::BuildToken::write(
        const Char8_t* name, const char* object, int value)
{
    assert( m_iItem < m_nItems );

    m_ptr[m_iItem].name   = name;
    m_ptr[m_iItem].object = object;
    m_ptr[m_iItem].value  = value;
    m_iItem++;
}




ConstantList::Item::Item( ConstantList::BuildToken& token ):
    m_token(token),
    m_name(0),
    m_object(0),
    m_value(0),
    m_isLast(false)
{
    m_token.incrementCount();
}

ConstantList::Item::~Item()
{
    if(!m_isLast)
        m_token.write(m_name,m_object,m_value);
}

ConstantList::Item ConstantList::Item::operator()(
        const Char8_t* name, const char* object, int value )
{
    m_name   = name;
    m_object = object;
    m_value  = value;

    return Item(m_token);
}

ConstantList::BuildToken& ConstantList::Item::operator()()
{
    m_isLast = true;
    m_token.allocate();
    return m_token;
}





















namespace name
{


bool registerObjectTypes( const ObjectTypeList& list )
{
    return FcNameRegisterObjectTypes(
            list.get_ptr(),
            list.get_nItems() );
}

bool unregisterObjectTypes( const ObjectTypeList& list )
{
    return FcNameUnregisterObjectTypes(
            list.get_ptr(),
            list.get_nItems() );
}

RefPtr<ObjectType> getObjectType( const char* object )
{
    return RefPtr<ObjectType>(  FcNameGetObjectType(object) );
}

bool registerConstants(const ConstantList& list)
{
    return FcNameRegisterConstants(
                list.get_ptr(),
                list.get_nItems() );
}

bool unregisterConstants(const ConstantList& list)
{
    return FcNameUnregisterConstants(
                list.get_ptr(),
                list.get_nItems() );
}

RefPtr<Constant> getConstant(Char8_t* string)
{
    return FcNameGetConstant(string);
}

bool constant(Char8_t* string, int* result)
{
    return FcNameConstant(string, result);
}














}
}











