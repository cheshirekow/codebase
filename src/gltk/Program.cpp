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
 *  @file   /home/josh/Codes/cpp/openglpp/src/gl/Program.cpp
 *
 *  @date   Dec 25, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */




#include <gltk/Program.h>
#include <GL/glew.h>

namespace gltk
{

Program::Program()
{
    m_id = glCreateProgram();
}

Program::~Program()
{
    if(m_id)
        glDeleteProgram(m_id);
}

void Program::attachShader( const RefPtr<Shader>& shader )
{
    glAttachShader(m_id,*shader);
}

bool Program::getDeleteStatus()
{
    GLint val;
    glGetProgramiv( m_id, GL_DELETE_STATUS, &val );
    return (val > 0 );
}

bool Program::getLinkStatus()
{
    GLint val;
    glGetProgramiv( m_id, GL_LINK_STATUS, &val );
    return (val > 0 );
}


GLint Program::getInfoLogLength()
{
    GLint val;
    glGetProgramiv( m_id, GL_INFO_LOG_LENGTH, &val );
    return val;
}

void Program::link()
{
    glLinkProgram(m_id);
}

void Program::use()
{
    glUseProgram(m_id);
}

void Program::getInfoLog( GLuint  program,  GLsizei  maxLength,
                            GLsizei * length,  GLchar * infoLog )
{
    glGetProgramInfoLog( m_id, maxLength, length, infoLog );
}

void Program::getInfoLog( std::string& log )
{
    GLsizei size    = getInfoLogLength();
    GLsizei length  = 0;
    log.reserve(size);

    char* writable = new char[size+1];
    glGetProgramInfoLog(m_id,size,&length,writable);
    log = writable;
    delete [] writable;
}


gltk::RefPtr<Program> Program::create()
{
    return gltk::RefPtr<Program>( new Program() );
}






}









