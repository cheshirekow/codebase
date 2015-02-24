/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of gltk.
 *
 *  gltk is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gltk is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with gltk.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   /home/josh/Codes/cpp/openglpp/include/gltk/gl/Program.h
 *
 *  @date   Dec 25, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef GLPLUS_PROGRAM_H_
#define GLPLUS_PROGRAM_H_


#include <GL/glew.h>
#include <GL/glfw.h>
#include <gltk/GLuintRef.h>
#include <gltk/RefPtr.h>
#include <gltk/Shader.h>
#include <string>

namespace gltk
{

class Program:
    public GLuintRef
{
    protected:
        /// constructs a new program object calling glCreateProgram, storing
        /// it's handle
        explicit Program();

    public:
        /// destroy the prgram using glDeleteProgram
        ~Program();

        /// attach a shader to the program
        void attachShader( const RefPtr<Shader>& shader );

        /// returns true if marked for deletion
        bool getDeleteStatus();

        /// returns true if last link operation was successful
        bool getLinkStatus();

        /// return the length of the info log
        GLint getInfoLogLength();

        /// link the shader program
        void link();

        /// activates the program (i.e. calls glUseProgram)
        void use();

        /// get the information log for a shader object
        void getInfoLog( GLuint  program,  GLsizei  maxLength,
                          GLsizei * length,  GLchar * infoLog );

        /// get the information log for a shader object
        void getInfoLog( std::string& log );


        /// create a new shader
        static RefPtr<Program> create();
};

}
















#endif // PROGRAM_H_
