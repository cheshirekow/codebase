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
 *  @file   include/cppfontconfig/name.h
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_NAME_H_
#define CPPFONTCONFIG_NAME_H_


#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/ObjectType.h>
#include <cpp_fontconfig/Constant.h>

namespace fontconfig {



/// utility class for building a static list of ObjectTypes, replaces
/// static allocation of a list of FcObjectType objects
/**
 *  Example usage:
 *
 *  \code{.cpp}
 *      ObjectTypeList* list = ObjectTypeList::create()
 *          ( "ObjectA", type::Integer )
 *          ( "ObjectB", type::String  )
 *          ( "ObjectC", type::Double  )
 *          ();
 *  \endcode
 *
 *  @note an ObjectTypeList is not copyable or publicly constructable. It can
 *        only be constructed in the manner of the above example. We take
 *        advantage of this by setting @p m_selfDestruct of the list
 *        returned by ObjectTypeList::create() to false, which means that
 *        the @p m_selfDestruct bit of the created list is true
 */
class ObjectTypeList
{
    public:
        class Item;
        class BuildToken;

    private:
        FcObjectType*   m_ptr;      ///< array of objects
        int             m_nItems;   ///< number of items

        static BuildToken sm_seed;

        /// private constructor, use ObjectTypeList::create() instead
        ObjectTypeList();

    public:
        /// construct an object from a token
        ObjectTypeList( BuildToken& token );

        /// copy a list
        ObjectTypeList(const ObjectTypeList& other );

        /// return the c-array of objects
        FcObjectType* get_ptr();

        /// return a const c-array of objects
        const FcObjectType* get_ptr() const;

        /// return the number of objects
        int get_nItems() const;

        /// destroy underlying data
        void destroy();

        /// creates a new Object type list, and returns an item which points
        /// to it
        static Item create();
};

class ObjectTypeList::BuildToken
{
    private:
        FcObjectType*   m_ptr;      ///< array of objects
        int             m_iItem;    ///< index of the current item to write
        int             m_nItems;   ///< number of items

        /// can only be created by ObjectTypeList
        explicit BuildToken(){}

        /// cannot be copied
        BuildToken( const BuildToken& other ){}

        /// cannot be copied
        BuildToken& operator=( const BuildToken& other ){ return *this; }

        /// initializes fields to zero
        void init();

        /// increments the item count
        /**
         *  Called by Item constructor
         */
        void incrementCount();

        /// allocates the object type buffer given the current value of
        /// m_nItems
        /**
         *  Called by Item::operator()()
         */
        void allocate();

        /// writes data and increments counter
        /**
         *  Called by Item d'tor
         */
        void write( const char* object, Type_t type );

    public:
        friend class ObjectTypeList;
        friend class ObjectTypeList::Item;



};

class ObjectTypeList::Item
{
    private:
        ObjectTypeList::BuildToken&     m_token;
        const char*                     m_object;
        Type_t                          m_type;
        bool                            m_isLast;

    public:
        /// initializes the item as an empty last item and increments the
        /// number of items in the list
        Item( ObjectTypeList::BuildToken& token );

        /// writes the item data to the list
        ~Item();

        /// fills this item, increments the item count,  and returns a new item;
        Item operator()( const char* object, Type_t type );

        /// finalizes the initialization by signaling the list to allocate
        /// data, the data is filled by ~Item()
        ObjectTypeList::BuildToken& operator()();
};




/// utility class for building a static list of ObjectTypes, replaces
/// static allocation of a list of FcObjectType objects
/**
 *  Example usage:
 *
 *  \code{.cpp}
 *      ConstantList list = ConstantList::create()
 *          ( "NameA", "ObjectA", 1001 )
 *          ( "NameB", "ObjectB", 1002  )
 *          ( "NameC", "ObjectC", 1003  )
 *          ();
 *  \endcode
 */
class ConstantList
{
    public:
        class Item;
        class BuildToken;

    private:
        FcConstant*   m_ptr;
        int           m_nItems;

        static BuildToken sm_seed;

        /// private constructor, use ObjectTypeList::create() instead
        ConstantList();

    public:
        /// construct an object from a token
        ConstantList(ConstantList::BuildToken& token );

        /// copy a list
        ConstantList(const ConstantList& other );

        /// return a c-array of constants
        FcConstant* get_ptr();

        /// return a  const c-array of constants
        const FcConstant* get_ptr() const;

        /// return the number of constants
        int get_nItems() const;

        /// destroy underlying data
        void destroy();

        /// creates a new Object type list, and returns an item which points
        /// to it
        static Item create();

};

class ConstantList::BuildToken
{
    private:
        FcConstant* m_ptr;      ///< array of objects
        int         m_iItem;    ///< index of the current item to write
        int         m_nItems;   ///< number of items

        /// can only be created by ObjectTypeList
        explicit BuildToken(){}

        /// cannot be copied
        BuildToken( const BuildToken& other ){}

        /// cannot be copied
        BuildToken& operator=( const BuildToken& other ){ return *this; }

        /// initializes fields to zero
        void init();

        /// increments the item count
        /**
         *  Called by Item constructor
         */
        void incrementCount();

        /// allocates the object type buffer given the current value of
        /// m_nItems
        /**
         *  Called by Item::operator()()
         */
        void allocate();

        /// writes data and increments counter, returns true if this is the
        /// last item
        /**
         *  Called by Item d'tor
         */
        void write( const Char8_t* name, const char* object, int value );

    public:
        friend class ConstantList;
        friend class ConstantList::Item;
};

class ConstantList::Item
{
    private:
        ConstantList::BuildToken&   m_token;
        const Char8_t*              m_name;
        const char*                 m_object;
        int                         m_value;
        bool                        m_isLast;

    public:
        /// initializes the item as an empty last item and increments the
        /// number of items in the list
        Item( ConstantList::BuildToken& token );

        /// writes the item data to the list
        ~Item();

        /// fills this item, increments the item count,  and returns a new item;
        Item operator()( const Char8_t* name, const char* object, int value );

        /// finalizes the initialization by signaling the list to allocate
        /// data, the data is filled by ~Item()
        ConstantList::BuildToken& operator()();
};



namespace       name {

/// Register object types
/**
 *  Register ntype new object types. Returns FcFalse if some of the names
 *  cannot be registered (due to allocation failure). Otherwise returns FcTrue.
 */
bool registerObjectTypes( const ObjectTypeList& list );

/// Unregister object types
/**
 *  Unregister ntype object types. Returns FcTrue.
 */
bool unregisterObjectTypes( const ObjectTypeList& list );

/// Lookup an object type
/**
 *  Return the object type for the pattern element named object.
 */
RefPtr<ObjectType> getObjectType( const char* object );

/// Register symbolic constants
/**
 *  Register nconsts new symbolic constants. Returns FcFalse if the constants
 *  cannot be registered (due to allocation failure). Otherwise returns FcTrue.
 */
bool registerConstants( const ConstantList& list );

/// Unregister symbolic constants
/**
 *  Unregister nconsts symbolic constants. Returns FcFalse if the specified
 *  constants were not registered. Otherwise returns FcTrue.
 */
bool unregisterConstants( const ConstantList& list );

/// Lookup symbolic constant
/**
 *  Return the FcConstant structure related to symbolic constant string.
 */
RefPtr<Constant> getConstant(Char8_t* string);

/// Get the value for a symbolic constant
/**
 *  Returns whether a symbolic constant with name string is registered,
 *  placing the value of the constant in result if present.
 */
bool constant( Char8_t* string, int* result );



}
}














#endif // NAME_H_
