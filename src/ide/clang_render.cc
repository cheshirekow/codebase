/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of clang-render.
 *
 *  clang-render is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  clang-render is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with clang-render.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Jan 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <set>
#include <string>

#include <clang-c/Index.h>
#include <cppformat/format.h>
#include <glog/logging.h>
#include <json_spirit/json_spirit.h>
#include <tclap/CmdLine.h>

#include "ide/string_to_argv.h"

bool NameMatches(const std::string &compile_file,
                  const std::string &query_file) {
  return compile_file.size() >= query_file.size() &&
         compile_file.substr(compile_file.size() - query_file.size(),
                             query_file.size()) == query_file;
}

std::string GetClangString(CXString str) {
  const char* tmp = clang_getCString(str);
  if (tmp == NULL) {
    return "";
  } else {
    std::string translated = std::string(tmp);
    clang_disposeString(str);
    return translated;
  }
}



#define MAP_ENUM(X) {X, #X}

std::set<int> kHighlightSet = {
  CXCursor_Namespace,
  CXCursor_CXXMethod,
  CXCursor_TypeRef,
  CXCursor_NamespaceRef
};

std::map<int,std::string> kCursorKindStr = {
  MAP_ENUM(CXCursor_Namespace),
  MAP_ENUM(CXCursor_CXXMethod),
  MAP_ENUM(CXCursor_TypeRef),
  MAP_ENUM(CXCursor_NamespaceRef),
};

void HtmlWrite(std::ostream& out, const char data) {
  switch (data) {
    case '&':
      out << "&amp;";
      break;
    case '\"':
      out << "&quot;";
      break;
    case '\'':
      out << "&apos;";
      break;
    case '<':
      out << "&lt;";
      break;
    case '>':
      out << "&gt;";
      break;
    default:
      out << data;
      break;
  }
}

void WriteHeader(std::ostream& out) {
  out << "  <html>\n"
         "    <head>\n"
         "      <style type=\"text/css\">\n"
         "        .CXCursor_TypeRef {\n"
         "          font-weight: bold;\n"
         "          color: red;\n"
         "        }\n"
         "\n"
         "        .cxx {\n"
         "          font-family: monospace;\n"
         "          white-space: pre;\n"
         "        }\n"
         "      </style>\n"
         "    </head>\n"
         "    <body>\n"
         "      <div class=\"cxx\">\n";
}

void WriteFooter(std::ostream& out) {
  out << "    </div>\n"
         "  </body>\n"
         "</html>\n";
}

void DoRender(const std::string& compile_command,
               const std::string& query_file,
              const std::string& output_file) {

  std::ofstream output_stream(output_file);
  if (!output_stream.good()) {
    LOG(FATAL) << "Failed to create output stream for " << output_file;
    return;
  }
  WriteHeader(output_stream);

  // open the file for read
  int fd = open(query_file.c_str(), O_RDONLY);

  // get the length of the file
  struct stat stat_buf;
  fstat(fd, &stat_buf);
  int file_size = stat_buf.st_size;

  // map the file into memory
  char* file_content = static_cast<char*>(
      mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0 /* offset */));

  std::list<std::string> argl = StringToArgv(compile_command);
  // pop off call to compiler
  argl.pop_front();

  // filter out -o flag
  for (auto iter = argl.begin(); iter != argl.end(); ++iter) {
    while (*iter == "-o" || *iter == "-c") {
      auto begin = iter;
      auto end = begin;
      ++end;
      ++end;
      argl.erase(begin, end);
      iter  = end;
    }
  }

  fmt::print("Using compiler options:\n");
  for (const std::string& arg : argl) {
    fmt::print("   {}\n", arg);
  }

  // create argv argc
  std::vector<char*> argv;
  argv.reserve(argl.size());
  for (std::string& arg : argl) {
    argv.push_back(&arg[0]);
  }

  CXIndex index = clang_createIndex(0, 0);
  CXTranslationUnit translation_unit =
      clang_parseTranslationUnit(index, query_file.c_str(), argv.data(),
                                 argv.size(), 0, 0, CXTranslationUnit_None);
  CXFile file = clang_getFile(translation_unit, query_file.c_str());
//  CXCursor startCursor = clang_getTranslationUnitCursor(translation_unit);

  unsigned line = 0;
  unsigned column = 0;
  for (char* current = file_content; current < file_content + file_size;
       ++current) {
    CXSourceLocation location =
        clang_getLocation(translation_unit, file, line, column);
    CXCursor cursor = clang_getCursor(translation_unit, location);
    CXCursorKind cursor_kind = clang_getCursorKind(cursor);
    // if it's something we want to highlight
    if(kHighlightSet.count(cursor_kind) > 0) {
      fmt::print(output_stream, "<span class=\"{}\">",
                 kCursorKindStr[cursor_kind]);
      CXSourceRange extent = clang_getCursorExtent(cursor);
      CXSourceLocation end_location = clang_getRangeEnd(extent);

      unsigned line_to;
      unsigned column_to;
      clang_getFileLocation(end_location, nullptr, &line_to, &column_to,
                            nullptr);
      for (; current <= file_content + file_size &&
                 (line < line_to || column < column_to);
           ++current) {
        HtmlWrite(output_stream, *current);
        if (*current == '\n') {
          ++line;
          column = 0;
        } else {
          ++column;
        }
      }

      fmt::print(output_stream, "</span>");
    }


    HtmlWrite(output_stream, *current);
    if (*current == '\n') {
      ++line;
      column = 0;
    } else {
      ++column;
    }
  }

  WriteFooter(output_stream);
  munmap(static_cast<void*>(file_content), file_size);
  close(fd);
}

int main(int argc, char **argv) {
  std::string db_path;
  std::string query_file;
  std::string output_file;

  try {
    TCLAP::ValueArg<std::string> db_path_arg(
        "d", "db_path", "Path to the compilation database", true,
        "./compilation_commands.json", "string");
    TCLAP::ValueArg<std::string> query_file_arg(
        "f", "file", "The file to render", true, "foo.cc", "string");
    TCLAP::ValueArg<std::string> output_file_arg(
        "o", "output", "The file to write to", true, "foo.html", "string");

    TCLAP::CmdLine cmd("Editor Proof of Concept", ' ', "0.1");
    cmd.add(db_path_arg);
    cmd.add(query_file_arg);
    cmd.add(output_file_arg);
    cmd.parse(argc, argv);
    db_path = db_path_arg.getValue();
    query_file = query_file_arg.getValue();
    output_file = output_file_arg.getValue();
  } catch (TCLAP::ArgException &e) {
    LOG(FATAL) << "error: " << e.error() << " for arg " << e.argId()
               << std::endl;
    return 1;
  }

  std::ifstream db_stream(db_path);
  if (!db_stream.good()) {
    LOG(FATAL) << "Failed to open compilation database" << db_path;
    return 1;
  }

  json_spirit::mValue db_root;
  json_spirit::read(db_stream, db_root);
  if (!db_root.type() == json_spirit::array_type) {
    LOG(FATAL) << "Compilation database root is not an array";
    return 1;
  }

  for (auto& value : db_root.get_array()) {
    if (value.type() == json_spirit::obj_type) {
      json_spirit::mObject obj = value.get_obj();
      if (NameMatches(obj["file"].get_str(), query_file)) {
        fmt::print("Found file {} in database\n", query_file);
        DoRender(obj["command"].get_str(), obj["file"].get_str(), output_file);
        return 0;
      }
    } else {
      LOG(WARNING) << "Skipping non object db entry";
    }
  }

  return 0;
}