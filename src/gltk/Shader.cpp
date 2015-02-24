/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of openbook.
 *
 *  openbook is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  openbook is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with openbook.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   /home/josh/Codes/cpp/openglpp/src/glplus/Shader.cpp
 *
 *  @date   Dec 25, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */


#include <gltk/Shader.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <iostream>

namespace gltk
{


Shader::Shader( GLenum type )
{
    m_id = glCreateShader( type );
}


Shader::~Shader()
{
    glDeleteShader( m_id );
}




void Shader::setSource(GLsizei count, const GLchar** src, GLint* length )
{
    glShaderSource(m_id, count, src, length);
}

void Shader::setSource( const std::string& src )
{
    const char* srcStr = src.c_str();
    glShaderSource(m_id,1,&srcStr,0);
}

void Shader::getSource(GLsizei bufSize, GLint& length, GLchar* source)
{
    glGetShaderSource(m_id,bufSize,&length,source);
}

void Shader::getSource( std::string& src )
{
    GLsizei size    = getSourceLength();
    GLsizei length  = 0;
    src.reserve(size+1);

    char* writable = new char[size+1];
    glGetShaderSource(m_id,size,&length,writable);
    src = writable;
    delete [] writable;
}

GLint Shader::getType()
{
    GLint val=0;
    glGetShaderiv( m_id, GL_SHADER_TYPE, &val );
    return val;
}

bool Shader::getDeleteStatus()
{
    GLint val=0;
    glGetShaderiv( m_id, GL_DELETE_STATUS, &val);
    return ( val == GL_TRUE );
}

bool Shader::getCompileStatus()
{
    GLint val=0;
    glGetShaderiv( m_id, GL_COMPILE_STATUS, &val);
    return ( val == GL_TRUE );
}

GLint Shader::getInfoLogLength()
{
    GLint val=0;
    glGetShaderiv( m_id, GL_INFO_LOG_LENGTH, &val);
    return val;
}

GLint Shader::getSourceLength()
{
    GLint val=0;
    glGetShaderiv( m_id, GL_SHADER_SOURCE_LENGTH, &val);
    return val;
}

void Shader::compile()
{
    glCompileShader(m_id);
}

void Shader::getInfoLog(GLsizei  maxLength,  GLsizei * length,  GLchar * infoLog)
{
    glGetShaderInfoLog( m_id, maxLength, length, infoLog );
}

void Shader::getInfoLog( std::string& log )
{
    log.clear();
    GLsizei size = getInfoLogLength();

    if( size < 1 )
        return;

    log.reserve(size);
    GLchar* writable = new GLchar[size+1];
    glGetShaderInfoLog(m_id,size,0,writable);
    log = writable;
    delete [] writable;
}



RefPtr<Shader> Shader::create( GLenum type )
{
    return RefPtr<Shader>( new Shader(type) );
}

RefPtr<Shader> Shader::create_from_file(GLenum type, const char* filename)
{
    // open the file
    int fd = open(filename,O_RDONLY);
    if (fd == -1)
        return RefPtr<Shader>();

    struct stat st;
    if( stat(filename, &st) != 0 )
    {
        close(fd);
        return RefPtr<Shader>();
    }
    size_t fileLen = st.st_size;

    void* map = mmap(0,fileLen,PROT_READ,MAP_SHARED,fd,0);
    if( map == MAP_FAILED )
    {
        close(fd);
        return RefPtr<Shader>();
    }

    const GLchar*  cmap = (const GLchar*)map;
    GLint          clen = fileLen-1;

    RefPtr<Shader> shader = create(type);
    shader->setSource(1,&cmap,&clen);

    munmap(map,fileLen);
    return shader;
}

RefPtr<Shader> Shader::create_from_string(GLenum type, const char* src)
{
    RefPtr<Shader>  shader = create(type);
    shader->setSource(1,&src,0);
    shader->compile();
    return shader;
}




}




