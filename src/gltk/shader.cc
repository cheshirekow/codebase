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

#include <gltk/shader.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <iostream>

namespace gltk {

Shader::Shader(GLenum type) {
  id_ = glCreateShader(type);
}

Shader::~Shader() {
  glDeleteShader(id_);
}

void Shader::SetSource(GLsizei count, const GLchar** src, GLint* length) {
  glShaderSource(id_, count, src, length);
}

void Shader::SetSource(const std::string& src) {
  const char* srcStr = src.c_str();
  glShaderSource(id_, 1, &srcStr, 0);
}

void Shader::GetSource(GLsizei bufSize, GLint& length, GLchar* source) {
  glGetShaderSource(id_, bufSize, &length, source);
}

void Shader::GetSource(std::string& src) {
  GLsizei size = GetSourceLength();
  GLsizei length = 0;
  src.reserve(size + 1);

  char* writable = new char[size + 1];
  glGetShaderSource(id_, size, &length, writable);
  src = writable;
  delete[] writable;
}

GLint Shader::GetType() {
  GLint val = 0;
  glGetShaderiv(id_, GL_SHADER_TYPE, &val);
  return val;
}

bool Shader::GetDeleteStatus() {
  GLint val = 0;
  glGetShaderiv(id_, GL_DELETE_STATUS, &val);
  return (val == GL_TRUE);
}

bool Shader::GetCompileStatus() {
  GLint val = 0;
  glGetShaderiv(id_, GL_COMPILE_STATUS, &val);
  return (val == GL_TRUE);
}

GLint Shader::GetInfoLogLength() {
  GLint val = 0;
  glGetShaderiv(id_, GL_INFO_LOG_LENGTH, &val);
  return val;
}

GLint Shader::GetSourceLength() {
  GLint val = 0;
  glGetShaderiv(id_, GL_SHADER_SOURCE_LENGTH, &val);
  return val;
}

void Shader::compile() {
  glCompileShader(id_);
}

void Shader::GetInfoLog(GLsizei maxLength, GLsizei * length, GLchar * infoLog) {
  glGetShaderInfoLog(id_, maxLength, length, infoLog);
}

void Shader::GetInfoLog(std::string& log) {
  log.clear();
  GLsizei size = GetInfoLogLength();

  if (size < 1)
    return;

  log.reserve(size);
  GLchar* writable = new GLchar[size + 1];
  glGetShaderInfoLog(id_, size, 0, writable);
  log = writable;
  delete[] writable;
}

RefPtr<Shader> Shader::Create(GLenum type) {
  return RefPtr<Shader>(new Shader(type));
}

RefPtr<Shader> Shader::CreateFromFile(GLenum type, const char* filename) {
  // open the file
  int fd = open(filename, O_RDONLY);
  if (fd == -1)
    return RefPtr<Shader>();

  struct stat st;
  if (stat(filename, &st) != 0) {
    close(fd);
    return RefPtr<Shader>();
  }
  size_t fileLen = st.st_size;

  void* map = mmap(0, fileLen, PROT_READ, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    close(fd);
    return RefPtr<Shader>();
  }

  const GLchar* cmap = (const GLchar*) map;
  GLint clen = fileLen - 1;

  RefPtr<Shader> shader = Create(type);
  shader->SetSource(1, &cmap, &clen);

  munmap(map, fileLen);
  return shader;
}

RefPtr<Shader> Shader::CreateFromString(GLenum type, const char* src) {
  RefPtr<Shader> shader = Create(type);
  shader->SetSource(1, &src, 0);
  shader->compile();
  return shader;
}

}  // namespace gltk

