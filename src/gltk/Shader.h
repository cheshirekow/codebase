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
 *  @file   include/gltk/gl/Shader.h
 *
 *  @date   Dec 23, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef GLTK_SHADER_H_
#define GLTK_SHADER_H_

#include <GL/glew.h>
#include <GL/glfw.h>
#include <gltk/GLuintRef.h>
#include <gltk/RefPtr.h>
#include <string>


namespace gltk
{

class Shader:
    public GLuintRef
{
    protected:
        /// constructs a new shader object calling glCreateShader, storing it's
        /// handle
        explicit Shader( GLenum type );

    public:
        /// destroy the shader using glDeleteShader
        ~Shader();

        /// Replaces the source code in a shader object
        /**
         *  @param count    the number of elements in the @p src and @p length
         *                  arrays
         *  @param src      specifies an array of pointers to strings containing
         *                  the source code to be loaded into the shader
         *  @param length   specifies an array of string lengths
         *
         *  glShaderSource sets the source code in shader to the source code in
         *  the array of strings specified by string. Any source code
         *  previously stored in the shader object is completely replaced. The
         *  number of strings in the array is specified by count. If length is
         *  NULL, each string is assumed to be null terminated. If length is a
         *  value other than NULL, it points to an array containing a string
         *  length for each of the corresponding elements of string. Each
         *  element in the length array may contain the length of the
         *  corresponding string (the null character is not counted as part of
         *  the string length) or a value less than 0 to indicate that the
         *  string is null terminated. The source code strings are not scanned
         *  or parsed at this time; they are simply copied into the specified
         *  shader object.
         *
         *  @note   OpenGL copies the shader source code strings when
         *  glShaderSource is called, so an application may free its copy of
         *  the source code strings immediately after the function returns.
         */
        void setSource(GLsizei count, const GLchar** src, GLint* length=0);

        /// set the source using a string object
        void setSource( const std::string& src );

        /// Returns the source code string from a shader object
        /**
         *  @param bufSize
         *  @param length
         *  @param source
         *
         *  glGetShaderSource returns the concatenation of the source code
         *  strings from the shader object specified by shader. The source code
         *  strings for a shader object are the result of a previous call to
         *  glShaderSource. The string returned by the function will be null
         *  terminated.glGetShaderSource returns in source as much of the
         *  source code string as it can, up to a maximum of bufSize
         *  characters. The number of characters actually returned, excluding
         *  the null termination character, is specified by length.
         *  If the length of the returned string is not required, a value of
         *  NULL can be passed in the length argument. The size of the buffer
         *  required to store the returned source code string can be obtained
         *  by calling glGetShader with the value GL_SHADER_SOURCE_LENGTH.
         */
        void getSource(GLsizei bufSize, GLint& length, GLchar* source);

        /// return the source code for this shader
        void getSource( std::string& src );

        /// returns the type (Vertex shader, fragment shader)
        GLint getType();

        /// returns true if marked for deletion
        bool getDeleteStatus();

        /// returns true if last compile operation was successful
        bool getCompileStatus();

        /// return the length of the info log
        GLint getInfoLogLength();

        /// return the shader source code lenght
        GLint getSourceLength();

        /// compile the shader
        void compile();

        /// get the information log for a shader object
        void getInfoLog( GLsizei  maxLength,  GLsizei * length,  GLchar * infoLog );

        /// get the information log for a shader object
        void getInfoLog( std::string& log );

        /// create a new shader
        static RefPtr<Shader> create(GLenum);

        /// create a new shader from a source code file
        static RefPtr<Shader> create_from_file( GLenum, const char* filename );

        /// create a new shader from a source code string
        static RefPtr<Shader> create_from_string( GLenum, const char* src );
};

}

#endif // SHADER_H_
