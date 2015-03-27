/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
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
 *  @file
 *  @date   Dec 25, 2012
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */

#include <gltk/program.h>
#include <GL/glew.h>

namespace gltk {

Program::Program() {
  m_id = glCreateProgram();
}

Program::~Program() {
  if (m_id)
    glDeleteProgram(m_id);
}

void Program::AttachShader(const RefPtr<Shader>& shader) {
  glAttachShader(m_id, *shader);
}

bool Program::GetDeleteStatus() {
  GLint val;
  glGetProgramiv(m_id, GL_DELETE_STATUS, &val);
  return (val > 0);
}

bool Program::GetLinkStatus() {
  GLint val;
  glGetProgramiv(m_id, GL_LINK_STATUS, &val);
  return (val > 0);
}

GLint Program::GetInfoLogLength() {
  GLint val;
  glGetProgramiv(m_id, GL_INFO_LOG_LENGTH, &val);
  return val;
}

void Program::Link() {
  glLinkProgram(m_id);
}

void Program::Use() {
  glUseProgram(m_id);
}

void Program::GetInfoLog(GLuint program, GLsizei maxLength, GLsizei * length,
                         GLchar * infoLog) {
  glGetProgramInfoLog(m_id, maxLength, length, infoLog);
}

void Program::GetInfoLog(std::string& log) {
  GLsizei size = GetInfoLogLength();
  GLsizei length = 0;
  log.reserve(size);

  char* writable = new char[size + 1];
  glGetProgramInfoLog(m_id, size, &length, writable);
  log = writable;
  delete[] writable;
}

gltk::RefPtr<Program> Program::Create() {
  return gltk::RefPtr<Program>(new Program());
}

}  // namespace gltk
