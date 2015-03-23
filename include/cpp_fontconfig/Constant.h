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
 *  @file   include/cppfontconfig/Constant.h
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_CONSTANT_H_
#define CPPFONTCONFIG_CONSTANT_H_

#include <fontconfig/fontconfig.h>
#include <cpp_fontconfig/RefPtr.h>
#include <cpp_fontconfig/common.h>

namespace fontconfig
{

class Constant;


class ConstantDelegate
{
    private:
        const FcConstant* m_ptr;

        /// wrap constructor
        /**
         *  wraps the pointer with this interface, does nothing else, only
         *  called by RefPtr<Atomic>
         */
        explicit ConstantDelegate(const FcConstant* ptr):
            m_ptr(ptr)
        {}

        /// not copy-constructable
        ConstantDelegate( const ConstantDelegate& other );

        /// not assignable
        ConstantDelegate& operator=( const ConstantDelegate& other );

    public:
        friend class RefPtr<Constant>;

        ConstantDelegate* operator->(){ return this; }
        const ConstantDelegate* operator->() const { return this; }


        const Char8_t*  get_name()   const;
        const char*     get_object() const;
        int             get_value()  const;
};


struct Constant
{
    typedef ConstantDelegate    Delegate;
    typedef const FcConstant*   Storage;
    typedef const FcConstant*   cobjptr;


};

} // namespace fontconfig 

#endif // CONSTANT_H_
