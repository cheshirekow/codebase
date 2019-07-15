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
#include "ck_gtk/layout_map.h"

#include <iostream>
#include <fstream>
#include <string>
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

namespace ck_gtk {

template <class Widget_t>
void WidgetMapBase<Widget_t>::Init(const std::string& key) {
  map_[key] = 0;
  std::cout << "added key: " << key << std::endl;
}

template <class Widget_t>
void WidgetMapBase<Widget_t>::Load(const Glib::RefPtr<Gtk::Builder>& builder) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    builder->get_widget(iPair->first, iPair->second);
    std::cout << "loaded widget: " << iPair->first << std::endl;
  }
}

template <class Widget_t>
void WidgetMapBase<Widget_t>::Read(YAML::Node& doc) {}

template <class Widget_t>
void WidgetMapBase<Widget_t>::Write(YAML::Emitter& doc) {}

template <>
void WidgetMapBase<Gtk::CheckButton>::Read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_active(doc[key].as<bool>());
    }
  }
}

template <>
void WidgetMapBase<Gtk::CheckButton>::Write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_active();
  }
}

template <>
void WidgetMapBase<Gtk::RadioButton>::Read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_active(doc[key].as<bool>());
    }
  }
}

template <>
void WidgetMapBase<Gtk::RadioButton>::Write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_active();
  }
}

template <>
void WidgetMapBase<Gtk::ColorButton>::Read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      Gdk::RGBA color(doc[key].as<std::string>());
      iPair->second->set_rgba(color);
    }
  }
}

template <>
void WidgetMapBase<Gtk::ColorButton>::Write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_rgba().to_string();
  }
}

template <>
void WidgetMapBase<Gtk::Expander>::Read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_expanded(doc[key].as<bool>());
    }
  }
}

template <>
void WidgetMapBase<Gtk::Expander>::Write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_expanded();
  }
}

template <class Object_t>
void ObjectMapBase<Object_t>::Init(const std::string& key) {
  map_[key] = Glib::RefPtr<Object_t>();
  std::cout << "added key: " << key << std::endl;
}

template <class Object_t>
void ObjectMapBase<Object_t>::Load(const Glib::RefPtr<Gtk::Builder>& builder) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    iPair->second =
        Glib::RefPtr<Object_t>::cast_static(builder->get_object(iPair->first));
    std::cout << "loaded object: " << iPair->first << std::endl;
  }
}

template <class Object_t>
void ObjectMapBase<Object_t>::Read(YAML::Node& doc) {}

template <class Object_t>
void ObjectMapBase<Object_t>::Write(YAML::Emitter& doc) {}

template <>
void ObjectMapBase<Gtk::Adjustment>::Read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_value(doc[key].as<double>());
    }
  }
}

template <>
void ObjectMapBase<Gtk::Adjustment>::Write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_value();
  }
}

template <>
void ObjectMapBase<Gtk::EntryBuffer>::Read(YAML::Node& doc) {
  std::string value;

  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_text(doc[key].as<std::string>());
    }
  }
}

template <>
void ObjectMapBase<Gtk::EntryBuffer>::Write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = map_.begin(); iPair != map_.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_text();
  }
}

void LayoutMap::LoadRecurse(tinyxml2::XMLElement* elmnt) {
  namespace tiny = tinyxml2;

  tiny::XMLElement* obj;
  tiny::XMLElement* child;

  // iterate over all objects
  obj = elmnt->FirstChildElement("object");
  while (obj) {
    const char* classStr = obj->Attribute("class");
    const char* idStr = obj->Attribute("id");

    if (strcmp(classStr, "GtkAdjustment") == 0)
      ObjectMapBase<Gtk::Adjustment>::Init(idStr);
    else if (strcmp(classStr, "GtkEntryBuffer") == 0)
      ObjectMapBase<Gtk::EntryBuffer>::Init(idStr);
    else if (strcmp(classStr, "GtkTextBuffer") == 0)
      ObjectMapBase<Gtk::TextBuffer>::Init(idStr);
    else if (strcmp(classStr, "GtkWindow") == 0)
      WidgetMapBase<Gtk::Window>::Init(idStr);
    else if (strcmp(classStr, "GtkAspectFrame") == 0)
      WidgetMapBase<Gtk::AspectFrame>::Init(idStr);
    else if (strcmp(classStr, "GtkFrame") == 0)
      WidgetMapBase<Gtk::Frame>::Init(idStr);
    else if (strcmp(classStr, "GtkBox") == 0)
      WidgetMapBase<Gtk::Box>::Init(idStr);
    else if (strcmp(classStr, "GtkAlignment") == 0)
      WidgetMapBase<Gtk::Alignment>::Init(idStr);
    else if (strcmp(classStr, "GtkButton") == 0)
      WidgetMapBase<Gtk::Button>::Init(idStr);
    else if (strcmp(classStr, "GtkCheckButton") == 0)
      WidgetMapBase<Gtk::CheckButton>::Init(idStr);
    else if (strcmp(classStr, "GtkToggleButton") == 0)
      WidgetMapBase<Gtk::ToggleButton>::Init(idStr);
    else if (strcmp(classStr, "GtkRadioButton") == 0)
      WidgetMapBase<Gtk::RadioButton>::Init(idStr);
    else if (strcmp(classStr, "GtkColorButton") == 0)
      WidgetMapBase<Gtk::ColorButton>::Init(idStr);
    else if (strcmp(classStr, "GtkScale") == 0)
      WidgetMapBase<Gtk::Scale>::Init(idStr);
    else if (strcmp(classStr, "GtkSpinButton") == 0)
      WidgetMapBase<Gtk::SpinButton>::Init(idStr);
    else if (strcmp(classStr, "GtkProgressBar") == 0)
      WidgetMapBase<Gtk::ProgressBar>::Init(idStr);
    else if (strcmp(classStr, "GtkComboBox") == 0)
      WidgetMapBase<Gtk::ComboBox>::Init(idStr);
    else if (strcmp(classStr, "GtkComboBoxText") == 0)
      WidgetMapBase<Gtk::ComboBoxText>::Init(idStr);
    else if (strcmp(classStr, "GtkTextView") == 0)
      WidgetMapBase<Gtk::TextView>::Init(idStr);
    else if (strcmp(classStr, "GtkGrid") == 0)
      WidgetMapBase<Gtk::Grid>::Init(idStr);
    else if (strcmp(classStr, "GtkDrawingArea") == 0)
      WidgetMapBase<Gtk::DrawingArea>::Init(idStr);
    else if (strcmp(classStr, "GtkTreeView") == 0)
      WidgetMapBase<Gtk::TreeView>::Init(idStr);
    else if (strcmp(classStr, "GtkExpander") == 0)
      WidgetMapBase<Gtk::Expander>::Init(idStr);

    LoadRecurse(obj);
    obj = obj->NextSiblingElement("object");
  }

  // iterate over all children and recurse into
  child = elmnt->FirstChildElement("child");
  while (child) {
    LoadRecurse(child);
    child = child->NextSiblingElement("child");
  }
}

bool LayoutMap::loadLayout(std::string layoutFile) {
  namespace tiny = tinyxml2;

  tiny::XMLDocument doc;
  if (doc.LoadFile(layoutFile.c_str()) != tiny::XML_SUCCESS) {
    std::cerr << "Failed to read " << layoutFile << std::endl;
    return false;
  }

  tiny::XMLElement* root = doc.RootElement();
  if (!root) {
    std::cerr << "No root element in " << layoutFile << std::endl;
    return false;
  }

  LoadRecurse(root);

  Glib::RefPtr<Gtk::Builder> builder;
  try {
    builder = Gtk::Builder::create_from_file(layoutFile);
  } catch (const Gtk::BuilderError& ex) {
    std::cerr << "Failed to load gladefile: " << layoutFile << "\n";
    std::cerr << ex.what();
    return false;
  }

  ObjectMapBase<Gtk::Adjustment>::Load(builder);
  ObjectMapBase<Gtk::EntryBuffer>::Load(builder);
  ObjectMapBase<Gtk::TextBuffer>::Load(builder);
  WidgetMapBase<Gtk::Button>::Load(builder);
  WidgetMapBase<Gtk::ToggleButton>::Load(builder);
  WidgetMapBase<Gtk::CheckButton>::Load(builder);
  WidgetMapBase<Gtk::RadioButton>::Load(builder);
  WidgetMapBase<Gtk::ColorButton>::Load(builder);
  WidgetMapBase<Gtk::SpinButton>::Load(builder);
  WidgetMapBase<Gtk::Window>::Load(builder);
  WidgetMapBase<Gtk::AspectFrame>::Load(builder);
  WidgetMapBase<Gtk::Frame>::Load(builder);
  WidgetMapBase<Gtk::Box>::Load(builder);
  WidgetMapBase<Gtk::Alignment>::Load(builder);
  WidgetMapBase<Gtk::Scale>::Load(builder);
  WidgetMapBase<Gtk::ProgressBar>::Load(builder);
  WidgetMapBase<Gtk::ComboBox>::Load(builder);
  WidgetMapBase<Gtk::ComboBoxText>::Load(builder);
  WidgetMapBase<Gtk::TextView>::Load(builder);
  WidgetMapBase<Gtk::Grid>::Load(builder);
  WidgetMapBase<Gtk::DrawingArea>::Load(builder);
  WidgetMapBase<Gtk::TreeView>::Load(builder);
  WidgetMapBase<Gtk::Expander>::Load(builder);

  return true;
}

void LayoutMap::LoadValues(std::string yamlFile) {
  try {
    YAML::Node doc = YAML::LoadFile(yamlFile);
    LoadValues(doc);
  } catch (const YAML::BadFile& ex) {
    std::cerr << "Warning: " << ex.what() << std::endl;
  }
}

void LayoutMap::LoadValues(YAML::Node& doc) {
  ObjectMapBase<Gtk::Adjustment>::Read(doc);
  ObjectMapBase<Gtk::EntryBuffer>::Read(doc);
  WidgetMapBase<Gtk::CheckButton>::Read(doc);
  WidgetMapBase<Gtk::RadioButton>::Read(doc);
  WidgetMapBase<Gtk::ColorButton>::Read(doc);
  WidgetMapBase<Gtk::Expander>::Read(doc);
}

void LayoutMap::SaveValues(std::string yamlFile) {
  YAML::Emitter out;
  SaveValues(out);

  std::ofstream fout(yamlFile.c_str());
  if (fout.good()) {
    fout << out.c_str();
  }

  std::cout << "Saving state: \n" << out.c_str() << std::endl;
}

void LayoutMap::SaveValues(YAML::Emitter& out) {
  out << YAML::BeginMap;
  ObjectMapBase<Gtk::Adjustment>::Write(out);
  ObjectMapBase<Gtk::EntryBuffer>::Write(out);
  WidgetMapBase<Gtk::CheckButton>::Write(out);
  WidgetMapBase<Gtk::RadioButton>::Write(out);
  WidgetMapBase<Gtk::ColorButton>::Write(out);
  WidgetMapBase<Gtk::Expander>::Write(out);
  out << YAML::EndMap;
}

}  // namespace ck_gtk
