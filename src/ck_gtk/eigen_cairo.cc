/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of ck_gtk.
 *
 *  ck_gtk is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  ck_gtk is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with ck_gtk.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "ck_gtk/eigen_cairo.h"

#include <iostream>
#include <iomanip>

namespace ck_gtk {

EigenCairo::EigenCairo(const Cairo::RefPtr<Cairo::Context>& ctx) { ctx_ = ctx; }

void EigenCairo::set_source(const Gdk::RGBA& rgba) {
  ctx_->set_source_rgba(rgba.get_red(), rgba.get_green(), rgba.get_blue(),
                        rgba.get_alpha());
}

void EigenCairo::set_source_rgb(const char* str) {
  unsigned int spec[3];
  bool valid = true;

  if (strlen(str) == 7 && str[0] == '#')
    str++;
  else if (strlen(str) != 6)
    valid = false;

  for (int i = 0; i < 3 && valid; i++) {
    spec[i] = 0;
    for (int j = 0; j < 2; j++) {
      char c = str[2 * i + j];
      if ('0' <= c && c <= '9')
        spec[i] |= (unsigned char)(c - '0') << 4 * (1 - j);
      else if ('a' <= c && c <= 'f')
        spec[i] |= ((unsigned char)(c - 'a') + 0x0A) << 4 * (1 - j);
      else if ('A' <= c && c <= 'F')
        spec[i] |= ((unsigned char)(c - 'A') + 0x0A) << 4 * (1 - j);
      else
        valid = false;
    }
  }

  if (!valid) {
    std::cerr << "invalid color spec: " << str << std::endl;
    return;
  }

  ctx_->set_source_rgb(spec[0] / 255.0, spec[1] / 255.0, spec[2] / 255.0);
}

void EigenCairo::set_source_rgba(const char* str) {
  unsigned int spec[4];
  bool valid = true;

  if (strlen(str) == 9 && str[0] == '#')
    str++;
  else if (strlen(str) != 8)
    valid = false;

  for (int i = 0; i < 4 && valid; i++) {
    spec[i] = 0;
    for (int j = 0; j < 2; j++) {
      char c = str[2 * i + j];
      if ('0' <= c && c <= '9')
        spec[i] |= (unsigned char)(c - '0') << 4 * (1 - j);
      else if ('a' <= c && c <= 'f')
        spec[i] |= ((unsigned char)(c - 'a') + 0x0A) << 4 * (1 - j);
      else if ('A' <= c && c <= 'F')
        spec[i] |= ((unsigned char)(c - 'A') + 0x0A) << 4 * (1 - j);
      else
        valid = false;
    }
  }

  if (!valid) {
    std::cerr << "invalid color spec: " << str << std::endl;
    return;
  }

  ctx_->set_source_rgba(spec[0] / 255.0, spec[1] / 255.0, spec[2] / 255.0,
                        spec[3] / 255.0);
}

void EigenCairo::user_to_device(Eigen::Matrix<double, 2, 1>& v) const {
  ctx_->user_to_device(v[0], v[1]);
}

Cairo::RefPtr<Cairo::SolidPattern> EigenCairo::SolidPattern::create_rgb(
    const char* str) {
  unsigned int spec[3];
  bool valid = true;

  if (strlen(str) == 7 && str[0] == '#')
    str++;
  else if (strlen(str) != 6)
    valid = false;

  for (int i = 0; i < 3 && valid; i++) {
    spec[i] = 0;
    for (int j = 0; j < 2; j++) {
      char c = str[2 * i + j];
      if ('0' <= c && c <= '9')
        spec[i] |= (unsigned char)(c - '0') << 4 * (1 - j);
      else if ('a' <= c && c <= 'f')
        spec[i] |= ((unsigned char)(c - 'a') + 0x0A) << 4 * (1 - j);
      else if ('A' <= c && c <= 'F')
        spec[i] |= ((unsigned char)(c - 'A') + 0x0A) << 4 * (1 - j);
      else
        valid = false;
    }
  }

  if (!valid) {
    std::cerr << "invalid color spec: " << str << std::endl;
    return Cairo::RefPtr<Cairo::SolidPattern>();
  }

  return Cairo::SolidPattern::create_rgb(spec[0] / 255.0, spec[1] / 255.0,
                                         spec[2] / 255.0);
}

Cairo::RefPtr<Cairo::SolidPattern> EigenCairo::SolidPattern::create_rgba(
    const char* str) {
  unsigned int spec[4];
  bool valid = true;

  if (strlen(str) == 9 && str[0] == '#')
    str++;
  else if (strlen(str) != 8)
    valid = false;

  for (int i = 0; i < 4 && valid; i++) {
    spec[i] = 0;
    for (int j = 0; j < 2; j++) {
      char c = str[2 * i + j];
      if ('0' <= c && c <= '9')
        spec[i] |= (unsigned char)(c - '0') << 4 * (1 - j);
      else if ('a' <= c && c <= 'f')
        spec[i] |= ((unsigned char)(c - 'a') + 0x0A) << 4 * (1 - j);
      else if ('A' <= c && c <= 'F')
        spec[i] |= ((unsigned char)(c - 'A') + 0x0A) << 4 * (1 - j);
      else
        valid = false;
    }
  }

  if (!valid) {
    std::cerr << "invalid color spec: " << str << std::endl;
    return Cairo::RefPtr<Cairo::SolidPattern>();
  }

  return Cairo::SolidPattern::create_rgba(spec[0] / 255.0, spec[1] / 255.0,
                                          spec[2] / 255.0, spec[3] / 255.0);
}

Cairo::RefPtr<Cairo::SolidPattern> EigenCairo::SolidPattern::create_rgb(
    const Gdk::Color& rgb) {
  return Cairo::SolidPattern::create_rgb(
      rgb.get_red() / 255.0, rgb.get_green() / 255.0, rgb.get_blue() / 255.0);
}

Cairo::RefPtr<Cairo::SolidPattern> EigenCairo::SolidPattern::create_rgba(
    const Gdk::RGBA& rgba) {
  return Cairo::SolidPattern::create_rgba(rgba.get_red(), rgba.get_green(),
                                          rgba.get_blue(), rgba.get_alpha());
}

}  // namespace ck_gtk
