/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
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
 *  @file
 *  @date   Dec 25, 2012
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */

#ifndef GLTK_PROGRAM_H_
#define GLTK_PROGRAM_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <gltk/gluintref.h>
#include <gltk/refptr.h>
#include <gltk/shader.h>
#include <string>

namespace gltk {

class Program : public GLuintRef {
 protected:
  /// constructs a new program object calling glCreateProgram, storing
  /// it's handle
  explicit Program();

 public:
  /// destroy the prgram using glDeleteProgram
  ~Program();

  /// attach a shader to the program
  void AttachShader(const RefPtr<Shader>& shader);

  /// returns true if marked for deletion
  bool GetDeleteStatus();

  /// returns true if last link operation was successful
  bool GetLinkStatus();

  /// return the length of the info log
  GLint GetInfoLogLength();

  /// link the shader program
  void Link();

  /// activates the program (i.e. calls glUseProgram)
  void Use();

  /// get the information log for a shader object
  void GetInfoLog(GLuint program, GLsizei maxLength, GLsizei * length,
                  GLchar * infoLog);

  /// get the information log for a shader object
  void GetInfoLog(std::string& log);

  /// create a new shader
  static RefPtr<Program> Create();
};

}  // namespace gltk

#endif // GLTK_PROGRAM_H_
