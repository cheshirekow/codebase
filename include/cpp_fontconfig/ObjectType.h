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
 *  @file   include/cppfontconfig/ObjectType.h
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_OBJECTTYPE_H_
#define CPPFONTCONFIG_OBJECTTYPE_H_

#include <fontconfig/fontconfig.h>
#include <cpp_fontconfig/RefPtr.h>
#include <cpp_fontconfig/common.h>

namespace fontconfig
{

class ObjectType;

/// wraps FcObjectType
class ObjectTypeDelegate
{
    private:
        const FcObjectType* m_ptr;

        /// wrap constructor
        /**
         *  wraps the pointer with this interface, does nothing else, only
         *  called by RefPtr<Atomic>
         */
        explicit ObjectTypeDelegate(FcObjectType* ptr):
            m_ptr(ptr)
        {}

        /// not copy-constructable
        ObjectTypeDelegate( const ObjectTypeDelegate& other );

        /// not assignable
        ObjectTypeDelegate& operator=( const ObjectTypeDelegate& other );

    public:
        friend class RefPtr<ObjectType>;

        ObjectTypeDelegate* operator->(){ return this; }
        const ObjectTypeDelegate* operator->() const { return this; }

        Type_t get_type();
};

struct ObjectType
{
    typedef ObjectTypeDelegate  Delegate;
    typedef const FcObjectType* Storage;
    typedef const FcObjectType* cobjptr;
};

} // namespace fontconfig 

#endif // OBJECTTYPE_H_
