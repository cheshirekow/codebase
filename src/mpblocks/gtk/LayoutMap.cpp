/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of mpblocks.
 *
 *  mpblocks is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  mpblocks is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with mpblocks.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Oct 24, 2012
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#include <mpblocks/gtk/LayoutMap.h>
#include <tinyxml2.h>
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>

namespace mpblocks {
namespace      gtk {

template <class Widget_t>
void WidgetMapBase<Widget_t>::init(std::string key) {
  m_map[key] = 0;
  std::cout << "added key: " << key << std::endl;
}

template <class Widget_t>
void WidgetMapBase<Widget_t>::load(const Glib::RefPtr<Gtk::Builder>& builder) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    builder->get_widget(iPair->first, iPair->second);
    std::cout << "loaded widget: " << iPair->first << std::endl;
  }
}

template <class Widget_t>
void WidgetMapBase<Widget_t>::read(YAML::Node& doc) {}

template <class Widget_t>
void WidgetMapBase<Widget_t>::write(YAML::Emitter& doc) {}

template <>
void WidgetMapBase<Gtk::CheckButton>::read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_active(doc[key].as<bool>());
    }
  }
}

template <>
void WidgetMapBase<Gtk::CheckButton>::write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_active();
  }
}

template <>
void WidgetMapBase<Gtk::RadioButton>::read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_active(doc[key].as<bool>());
    }
  }
}

template <>
void WidgetMapBase<Gtk::RadioButton>::write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_active();
  }
}

template <>
void WidgetMapBase<Gtk::ColorButton>::read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      Gdk::RGBA color(doc[key].as<std::string>());
      iPair->second->set_rgba(color);
    }
  }
}

template <>
void WidgetMapBase<Gtk::ColorButton>::write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_rgba().to_string();
  }
}

template <>
void WidgetMapBase<Gtk::Expander>::read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_expanded(doc[key].as<bool>());
    }
  }
}

template <>
void WidgetMapBase<Gtk::Expander>::write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_expanded();
  }
}

template <class Object_t>
void ObjectMapBase<Object_t>::init(std::string key) {
  m_map[key] = Glib::RefPtr<Object_t>();
  std::cout << "added key: " << key << std::endl;
}

template <class Object_t>
void ObjectMapBase<Object_t>::load(const Glib::RefPtr<Gtk::Builder>& builder) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    iPair->second =
        Glib::RefPtr<Object_t>::cast_static(builder->get_object(iPair->first));
    std::cout << "loaded object: " << iPair->first << std::endl;
  }
}

template <class Object_t>
void ObjectMapBase<Object_t>::read(YAML::Node& doc) {}

template <class Object_t>
void ObjectMapBase<Object_t>::write(YAML::Emitter& doc) {}

template <>
void ObjectMapBase<Gtk::Adjustment>::read(YAML::Node& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_value(doc[key].as<double>());
    }
  }
}

template <>
void ObjectMapBase<Gtk::Adjustment>::write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_value();
  }
}

template <>
void ObjectMapBase<Gtk::EntryBuffer>::read(YAML::Node& doc) {
  std::string value;

  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    std::string key = iPair->first;
    if (doc[key]) {
      iPair->second->set_text(doc[key].as<std::string>());
    }
  }
}

template <>
void ObjectMapBase<Gtk::EntryBuffer>::write(YAML::Emitter& doc) {
  typename Map_t::iterator iPair;
  for (iPair = m_map.begin(); iPair != m_map.end(); ++iPair) {
    doc << YAML::Key << iPair->first;
    doc << YAML::Value << iPair->second->get_text();
  }
}

void LayoutMap::loadRecurse(tinyxml2::XMLElement* elmnt) {
  namespace tiny = tinyxml2;

  tiny::XMLElement* obj;
  tiny::XMLElement* child;

  // iterate over all objects
  obj = elmnt->FirstChildElement("object");
  while (obj) {
    const char* classStr = obj->Attribute("class");
    const char* idStr = obj->Attribute("id");

    if (strcmp(classStr, "GtkAdjustment") == 0)
      ObjectMapBase<Gtk::Adjustment>::init(idStr);
    else if (strcmp(classStr, "GtkEntryBuffer") == 0)
      ObjectMapBase<Gtk::EntryBuffer>::init(idStr);
    else if (strcmp(classStr, "GtkTextBuffer") == 0)
      ObjectMapBase<Gtk::TextBuffer>::init(idStr);
    else if (strcmp(classStr, "GtkWindow") == 0)
      WidgetMapBase<Gtk::Window>::init(idStr);
    else if (strcmp(classStr, "GtkAspectFrame") == 0)
      WidgetMapBase<Gtk::AspectFrame>::init(idStr);
    else if (strcmp(classStr, "GtkFrame") == 0)
      WidgetMapBase<Gtk::Frame>::init(idStr);
    else if (strcmp(classStr, "GtkBox") == 0)
      WidgetMapBase<Gtk::Box>::init(idStr);
    else if (strcmp(classStr, "GtkAlignment") == 0)
      WidgetMapBase<Gtk::Alignment>::init(idStr);
    else if (strcmp(classStr, "GtkButton") == 0)
      WidgetMapBase<Gtk::Button>::init(idStr);
    else if (strcmp(classStr, "GtkCheckButton") == 0)
      WidgetMapBase<Gtk::CheckButton>::init(idStr);
    else if (strcmp(classStr, "GtkToggleButton") == 0)
      WidgetMapBase<Gtk::ToggleButton>::init(idStr);
    else if (strcmp(classStr, "GtkRadioButton") == 0)
      WidgetMapBase<Gtk::RadioButton>::init(idStr);
    else if (strcmp(classStr, "GtkColorButton") == 0)
      WidgetMapBase<Gtk::ColorButton>::init(idStr);
    else if (strcmp(classStr, "GtkScale") == 0)
      WidgetMapBase<Gtk::Scale>::init(idStr);
    else if (strcmp(classStr, "GtkSpinButton") == 0)
      WidgetMapBase<Gtk::SpinButton>::init(idStr);
    else if (strcmp(classStr, "GtkProgressBar") == 0)
      WidgetMapBase<Gtk::ProgressBar>::init(idStr);
    else if (strcmp(classStr, "GtkComboBox") == 0)
      WidgetMapBase<Gtk::ComboBox>::init(idStr);
    else if (strcmp(classStr, "GtkComboBoxText") == 0)
      WidgetMapBase<Gtk::ComboBoxText>::init(idStr);
    else if (strcmp(classStr, "GtkTextView") == 0)
      WidgetMapBase<Gtk::TextView>::init(idStr);
    else if (strcmp(classStr, "GtkGrid") == 0)
      WidgetMapBase<Gtk::Grid>::init(idStr);
    else if (strcmp(classStr, "GtkDrawingArea") == 0)
      WidgetMapBase<Gtk::DrawingArea>::init(idStr);
    else if (strcmp(classStr, "GtkTreeView") == 0)
      WidgetMapBase<Gtk::TreeView>::init(idStr);
    else if (strcmp(classStr, "GtkExpander") == 0)
      WidgetMapBase<Gtk::Expander>::init(idStr);

    loadRecurse(obj);
    obj = obj->NextSiblingElement("object");
  }

  // iterate over all children and recurse into
  child = elmnt->FirstChildElement("child");
  while (child) {
    loadRecurse(child);
    child = child->NextSiblingElement("child");
  }
}

bool LayoutMap::loadLayout(std::string layoutFile) {
  namespace tiny = tinyxml2;

  tiny::XMLDocument doc;
  if(doc.LoadFile(layoutFile.c_str()) != tiny::XML_SUCCESS) {
    std::cerr << "Failed to read " << layoutFile << std::endl;
    return false;
  }

  tiny::XMLElement* root = doc.RootElement();
  if (!root) {
    std::cerr << "No root element in " << layoutFile << std::endl;
    return false;
  }

  loadRecurse(root);

  Glib::RefPtr<Gtk::Builder> builder;
  try {
    builder = Gtk::Builder::create_from_file(layoutFile);
  } catch (const Gtk::BuilderError& ex) {
    std::cerr << "Failed to load gladefile: " << layoutFile << "\n";
    std::cerr << ex.what();
    return false;
  }

  ObjectMapBase<Gtk::Adjustment>::load(builder);
  ObjectMapBase<Gtk::EntryBuffer>::load(builder);
  ObjectMapBase<Gtk::TextBuffer>::load(builder);
  WidgetMapBase<Gtk::Button>::load(builder);
  WidgetMapBase<Gtk::ToggleButton>::load(builder);
  WidgetMapBase<Gtk::CheckButton>::load(builder);
  WidgetMapBase<Gtk::RadioButton>::load(builder);
  WidgetMapBase<Gtk::ColorButton>::load(builder);
  WidgetMapBase<Gtk::SpinButton>::load(builder);
  WidgetMapBase<Gtk::Window>::load(builder);
  WidgetMapBase<Gtk::AspectFrame>::load(builder);
  WidgetMapBase<Gtk::Frame>::load(builder);
  WidgetMapBase<Gtk::Box>::load(builder);
  WidgetMapBase<Gtk::Alignment>::load(builder);
  WidgetMapBase<Gtk::Scale>::load(builder);
  WidgetMapBase<Gtk::ProgressBar>::load(builder);
  WidgetMapBase<Gtk::ComboBox>::load(builder);
  WidgetMapBase<Gtk::ComboBoxText>::load(builder);
  WidgetMapBase<Gtk::TextView>::load(builder);
  WidgetMapBase<Gtk::Grid>::load(builder);
  WidgetMapBase<Gtk::DrawingArea>::load(builder);
  WidgetMapBase<Gtk::TreeView>::load(builder);
  WidgetMapBase<Gtk::Expander>::load(builder);

  return true;
}

void LayoutMap::loadValues(std::string yamlFile) {
  try {
    YAML::Node doc = YAML::LoadFile(yamlFile);
    loadValues(doc);
  } catch (const YAML::BadFile& ex) {
    std::cerr << "Warning: " << ex.what() << std::endl;
  }
}

void LayoutMap::loadValues(YAML::Node& doc) {
  ObjectMapBase<Gtk::Adjustment>::read(doc);
  ObjectMapBase<Gtk::EntryBuffer>::read(doc);
  WidgetMapBase<Gtk::CheckButton>::read(doc);
  WidgetMapBase<Gtk::RadioButton>::read(doc);
  WidgetMapBase<Gtk::ColorButton>::read(doc);
  WidgetMapBase<Gtk::Expander>::read(doc);
}

void LayoutMap::saveValues(std::string yamlFile) {
  YAML::Emitter out;
  saveValues(out);

  std::ofstream fout(yamlFile.c_str());
  if (fout.good()) {
    fout << out.c_str();
  }

  std::cout << "Saving state: \n" << out.c_str() << std::endl;
}

void LayoutMap::saveValues(YAML::Emitter& out) {
  out << YAML::BeginMap;
  ObjectMapBase<Gtk::Adjustment>::write(out);
  ObjectMapBase<Gtk::EntryBuffer>::write(out);
  WidgetMapBase<Gtk::CheckButton>::write(out);
  WidgetMapBase<Gtk::RadioButton>::write(out);
  WidgetMapBase<Gtk::ColorButton>::write(out);
  WidgetMapBase<Gtk::Expander>::write(out);
  out << YAML::EndMap;
}

} // gtk
} // mpblocks

