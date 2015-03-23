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
 *  @file   src/Pattern.cpp
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/Pattern.h>
#include <cpp_fontconfig/Config.h>
#include <fontconfig/fontconfig.h>

namespace fontconfig
{



template<>
void RefPtr<Pattern>::reference()
{
    if(m_ptr)
        FcPatternReference( m_ptr );
}

template<>
void RefPtr<Pattern>::dereference()
{
    if(m_ptr)
        FcPatternDestroy( m_ptr );
}




RefPtr<Pattern> Pattern::create(void)
{
    return FcPatternCreate();
}

Pattern::Builder Pattern::buildNew( )
{
    RefPtr<Pattern> result = Pattern::create();
    return result->build();
}

RefPtr<Pattern> Pattern::parse(const Char8_t* name)
{
    return FcNameParse( name );
}


Pattern::Builder::Builder(RefPtr<Pattern> pattern):
    m_pattern(pattern)
{
}

Pattern::Builder& Pattern::Builder::operator ()(const char* obj, int i)
{
    m_pattern->add(obj,i);
    return *this;
}

Pattern::Builder& Pattern::Builder::operator ()(const char* obj, double d)
{
    m_pattern->add(obj,d);
    return *this;
}

Pattern::Builder& Pattern::Builder::operator ()(const char* obj, Char8_t* s)
{
    m_pattern->add(obj,s);
    return *this;
}

Pattern::Builder& Pattern::Builder::operator ()(const char* obj, const Matrix& m)
{
    m_pattern->add(obj,m);
    return *this;
}

Pattern::Builder& Pattern::Builder::operator ()(const char* obj, const RefPtr<CharSet>& cs)
{
    m_pattern->add(obj,cs);
    return *this;
}

Pattern::Builder& Pattern::Builder::operator ()(const char* obj, bool b)
{
    m_pattern->add(obj,b);
    return *this;
}

Pattern::Builder& Pattern::Builder::operator ()(const char* obj, const RefPtr<LangSet>& ls)
{
    m_pattern->add(obj,ls);
    return *this;
}

RefPtr<Pattern> Pattern::Builder::done()
{
    return m_pattern;
}




RefPtr<Pattern> PatternDelegate::duplicate()
{
    return FcPatternDuplicate( m_ptr );
}

RefPtr<Pattern> PatternDelegate::filter(const RefPtr<ObjectSet> os)
{
    return RefPtr<Pattern>(
            FcPatternFilter( m_ptr,
                             os.subvert() ) );
}

bool PatternDelegate::equal(const RefPtr<Pattern> pb)
{
    return FcPatternEqual( m_ptr, pb.subvert() );
}

bool PatternDelegate::equalSubset(
        const RefPtr<Pattern> pb, const RefPtr<ObjectSet> os)
{
    return FcPatternEqualSubset(
                m_ptr,
                pb.subvert(),
                os.subvert() );

}

Char32_t PatternDelegate::hash()
{
    return FcPatternHash( m_ptr );
}

bool PatternDelegate::del(const char* object)
{
    return FcPatternDel( m_ptr, object );
}

bool PatternDelegate::remove(const char* object, int id)
{
    return FcPatternRemove( m_ptr, object, id );
}

bool PatternDelegate::add(const char* obj, int i)
{
    return FcPatternAddInteger( m_ptr, obj, i );
}

bool PatternDelegate::add(const char* obj, double d)
{
    return FcPatternAddDouble( m_ptr, obj, d );
}

bool PatternDelegate::add(const char* obj, const Char8_t* s)
{
    return FcPatternAddString( m_ptr, obj, s );
}

bool PatternDelegate::add(const char* obj, const Matrix& m)
{
    return FcPatternAddMatrix( m_ptr,
                                obj,
                                &m );
}

bool PatternDelegate::add(const char* obj, const RefPtr<CharSet> c)
{
    return FcPatternAddCharSet( m_ptr,
                                obj,
                                c.subvert() );
}

bool PatternDelegate::add(const char* obj, bool b)
{
    return FcPatternAddBool( m_ptr,
                                obj,
                                b ? FcTrue : FcFalse );
}

bool PatternDelegate::add(const char* obj, const RefPtr<LangSet> ls)
{
    return FcPatternAddLangSet( m_ptr,
                                obj,
                                ls.subvert() );
}

Result_t PatternDelegate::get(const char* obj, int n, int& i)
{
    return (Result_t)FcPatternGetInteger( m_ptr, obj, n, &i );
}

Result_t PatternDelegate::get(const char* obj, int n, double& d)
{
    return (Result_t)FcPatternGetDouble( m_ptr, obj, n, &d );
}

Result_t PatternDelegate::get(const char* obj, int n, Char8_t*& s)
{
    return (Result_t)FcPatternGetString( m_ptr, obj, n, &s );
}


// FIXME: Figure out what to do about matrices... if we do what we're doing now,
// the got matrix is not modifiable, or rather, if modified the matrix stored
// in the pattern wont get those changes
Result_t PatternDelegate::get(const char* obj, int n, Matrix*& m)
{
    FcMatrix* mm;
    Result_t result =
            (Result_t)FcPatternGetMatrix( m_ptr, obj, n, &mm );
    m = static_cast<Matrix*>(mm);

    return result;
}

Result_t PatternDelegate::get(const char* obj, int n, RefPtr<CharSet>& c)
{
    FcCharSet* cc;
    Result_t result =
            (Result_t)FcPatternGetCharSet( m_ptr, obj, n, &cc );

    // char sets are reference counted so we need to increment the reference
    // since we're taking a pointer to it
    c = cc;

    return result;
}

Result_t PatternDelegate::get(const char* obj, int n, bool& b)
{
    FcBool bb;
    Result_t result =
            (Result_t)FcPatternGetBool( m_ptr, obj, n, &bb );
    b = bb != 0;
    return result;
}

Result_t PatternDelegate::get(const char* obj, int n, RefPtr<LangSet>& ls)
{
    FcLangSet* lss;
    Result_t result =
            (Result_t)FcPatternGetLangSet( m_ptr, obj, n, &lss );

    ls = lss;
    return result;
}

Pattern::Builder PatternDelegate::build( )
{
    return Pattern::Builder(RefPtr<Pattern>(m_ptr));
}


Char8_t* PatternDelegate::unparse()
{
    return FcNameUnparse( m_ptr );
}

Char8_t* PatternDelegate::format(const Char8_t* format)
{
    return FcPatternFormat( m_ptr, format );
}

void PatternDelegate::print()
{
    return FcPatternPrint( m_ptr );
}

void PatternDelegate::defaultSubstitute()
{
    return FcDefaultSubstitute( m_ptr );
}

bool PatternDelegate::substitute(RefPtr<Config> c, MatchKind_t kind)
{
    return c->substitute(RefPtr<Pattern>(m_ptr),kind);
}

bool PatternDelegate::substitute(MatchKind_t kind)
{
    return FcConfigSubstitute(
                (FcConfig*)0,
                m_ptr,
                (FcMatchKind)kind );
}





}


 // namespace fontconfig 
