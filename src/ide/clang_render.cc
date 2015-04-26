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
#include <string>

#include <clang-c/Index.h>
#include <cppformat/format.h>
#include <glog/logging.h>
#include <json_spirit/json_spirit.h>
#include <tclap/CmdLine.h>

#include "ide/string_to_argv.h"

bool name_matches(const std::string &compile_file,
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

//highlightMap_[CXToken_Punctuation] = QBrush(QColor("black"));
//highlightMap_[CXToken_Keyword] = QBrush(QColor("green"));
//highlightMap_[CXToken_Identifier] = QBrush(QColor("black"));
//highlightMap_[CXToken_Literal] = QBrush(QColor("red"));
//highlightMap_[CXToken_Comment] = QBrush(QColor("blue"));

struct VisitorContext {
  std::string query_file;
};

CXChildVisitResult Visitor(CXCursor cursor, CXCursor parent,
                           CXClientData client_data) {
  VisitorContext* context = static_cast<VisitorContext*>(client_data);

  CXFile file;
  unsigned int line;
  unsigned int column;
  unsigned int offset;

  CXSourceLocation loc = clang_getCursorLocation(cursor);
  clang_getFileLocation(loc, &file, &line, &column, &offset);

  if (GetClangString(clang_getFileName(file)) != context->query_file) {
    // Skip files other than the one we're highlighting
    return CXChildVisit_Continue;
  }

  CXTranslationUnit tu = clang_Cursor_getTranslationUnit(cursor);
  CXSourceRange range = clang_getCursorExtent(cursor);

  CXToken* tokens;
  unsigned int numTokens;
  clang_tokenize(tu, range, &tokens, &numTokens);

  if (numTokens > 0) {
    for (unsigned int i = 0; i < numTokens - 1; i++) {
      std::string token = GetClangString(clang_getTokenSpelling(tu, tokens[i]));
      CXSourceLocation tl = clang_getTokenLocation(tu, tokens[i]);

      clang_getFileLocation(tl, &file, &line, &column, &offset);
      fmt::print("{} : {}[{}]\n", GetClangString(clang_getFileName(file)), line,
                 column);
//            mw->highlightText(line, column, token.size(),
//                clang_getTokenKind(tokens[i]));
    }
  }

  return CXChildVisit_Continue;
}

void DoRender(const std::string& compile_command,
               const std::string& query_file) {
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
  CXCursor startCursor = clang_getTranslationUnitCursor(translation_unit);

  VisitorContext context;
  context.query_file = query_file;
  clang_visitChildren(startCursor, Visitor, &context);

  munmap(static_cast<void*>(file_content), file_size);
}

int main(int argc, char **argv) {
  std::string db_path;
  std::string query_file;

  try {
    TCLAP::ValueArg<std::string> db_path_arg(
        "d", "db_path", "Path to the compilation database", true,
        "./compilation_commands.json", "string");
    TCLAP::ValueArg<std::string> query_file_arg(
        "f", "file", "The file to render", true, "foo.cc", "string");

    TCLAP::CmdLine cmd("Editor Proof of Concept", ' ', "0.1");
    cmd.add(db_path_arg);
    cmd.add(query_file_arg);
    cmd.parse(argc, argv);
    db_path = db_path_arg.getValue();
    query_file = query_file_arg.getValue();
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
      if (name_matches(obj["file"].get_str(), query_file)) {
        fmt::print("Found file {} in database\n", query_file);
        DoRender(obj["command"].get_str(), obj["file"].get_str());
        return 0;
      }
    } else {
      LOG(WARNING) << "Skipping non object db entry";
    }
  }

  return 0;
}
