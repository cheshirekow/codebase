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

#ifndef MPBLOCKS_GTK_LAYOUTMAP_H_
#define MPBLOCKS_GTK_LAYOUTMAP_H_

#include <gtkmm.h>
#include <map>
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

namespace mpblocks {
namespace      gtk {

/// base class for a map of string names -> widget pointers
template <class Widget_t>
class WidgetMapBase {
 public:
  /// the map type stored by this base class
  typedef std::map<std::string, Widget_t*> Map_t;

 protected:
  /// the actual map
  Map_t m_map;

 public:
  /// adds the key to the map with an empty pointer
  void init(std::string key);

  /// get widget pointers for all widgets whose names have been read in
  void load(const Glib::RefPtr<Gtk::Builder>& builder);

  /// read in any values for our widgets which are stored in the
  /// node
  void read(YAML::Node&);

  /// write out the values of our widgets into the yaml file
  void write(YAML::Emitter&);
};

/// base class for a map of string names -> object RefPtrs
template <class Object_t>
class ObjectMapBase {
 public:
  /// the map type stored by this base class
  typedef std::map<std::string, Glib::RefPtr<Object_t> > Map_t;

 protected:
  /// the actual map
  Map_t m_map;

 public:
  /// adds a key to the map with an empty reference
  void init(std::string key);

  /// retrieves RefPtrs for all the objects whose keys have been
  /// entered
  void load(const Glib::RefPtr<Gtk::Builder>& builder);

  /// read in values for any of the objects in our map
  void read(YAML::Node&);

  /// write out the values for all of the objects in our map
  void write(YAML::Emitter&);
};

struct WidgetType{};
struct ObjectType{};

/// maps a Gtk:: class to either a WidgetType or an ObjectType, default is
/// WidgetType
template <class T> struct TypeOf{ typedef WidgetType type; };

template <> struct TypeOf< Gtk::Adjustment >{ typedef ObjectType type; };
template <> struct TypeOf< Gtk::EntryBuffer>{ typedef ObjectType type; };
template <> struct TypeOf< Gtk::TextBuffer>{  typedef ObjectType type; };

/// maps a class to the pointer type and map type corresponding to it, default
/// is a widget map whose pointer types are regular points
template <class T, class Type> struct GetTypes;

/// sepecialization for widget types
template <class T>
struct GetTypes<T, WidgetType> {
  typedef T* Ptr_t;
  typedef WidgetMapBase<T> Map_t;
};

/// specialization for object types
template <class T>
struct GetTypes<T, ObjectType> {
  typedef Glib::RefPtr<T> Ptr_t;
  typedef ObjectMapBase<T> Map_t;
};

/// maps a widget or object class to their pointer and map types
template <class T>
struct DerivedTypes {
  typedef GetTypes<T, typename TypeOf<T>::type> GetTypes_t;
  typedef typename GetTypes_t::Ptr_t Ptr_t;
  typedef typename GetTypes_t::Map_t Map_t;
};

/// stores typed maps from string names -> pointers or RefPtrs depending on
/// if the pointed-to object is a widget or an object
class LayoutMap : public WidgetMapBase<Gtk::Window>,
                  public WidgetMapBase<Gtk::Button>,
                  public WidgetMapBase<Gtk::CheckButton>,
                  public WidgetMapBase<Gtk::ToggleButton>,
                  public WidgetMapBase<Gtk::RadioButton>,
                  public WidgetMapBase<Gtk::ColorButton>,
                  public WidgetMapBase<Gtk::SpinButton>,
                  public WidgetMapBase<Gtk::AspectFrame>,
                  public WidgetMapBase<Gtk::Frame>,
                  public WidgetMapBase<Gtk::Box>,
                  public WidgetMapBase<Gtk::Alignment>,
                  public WidgetMapBase<Gtk::Scale>,
                  public WidgetMapBase<Gtk::ProgressBar>,
                  public WidgetMapBase<Gtk::ComboBox>,
                  public WidgetMapBase<Gtk::ComboBoxText>,
                  public WidgetMapBase<Gtk::TextView>,
                  public WidgetMapBase<Gtk::Grid>,
                  public WidgetMapBase<Gtk::DrawingArea>,
                  public WidgetMapBase<Gtk::TreeView>,
                  public WidgetMapBase<Gtk::Expander>,
                  public ObjectMapBase<Gtk::Adjustment>,
                  public ObjectMapBase<Gtk::EntryBuffer>,
                  public ObjectMapBase<Gtk::TextBuffer> {
 private:
  /// recursively read in a glade file and record all keys for widgets
  /// that we  know about
  void loadRecurse(tinyxml2::XMLElement*);

 public:
  /// load a GUI from a glade file, mapping each widget/object name
  /// to a reference/pointer to that object
  bool loadLayout(std::string layoutFile);

  /// read in values from a yaml file
  void loadValues(std::string yamlFile);
  void loadValues(YAML::Node& node);

  /// record values from all the widgets to the yaml file
  void saveValues(std::string yamlFile);
  void saveValues(YAML::Emitter& emitter);

  /// return a pointer to a widget
  template <class Widget_t>
  Widget_t* widget(std::string key) {
    return WidgetMapBase<Widget_t>::m_map[key];
  }

  /// return a RefPtr to an object
  template <class Object_t>
  Glib::RefPtr<Object_t> object(std::string key) {
    return ObjectMapBase<Object_t>::m_map[key];
  }

  /// unified getter
  template <class T>
  typename DerivedTypes<T>::Ptr_t get(const std::string& key) {
    typedef typename DerivedTypes<T>::Map_t Base_t;
    typename Base_t::Map_t::iterator iter = Base_t::m_map.find(key);
    if (iter != Base_t::m_map.end()) {
      return iter->second;
    } else {
      return typename DerivedTypes<T>::Ptr_t(nullptr);
    }
  }

  /// unified setter
  template <class T>
  void set(const std::string& key, typename DerivedTypes<T>::Ptr_t ptr) {
    typedef typename DerivedTypes<T>::Map_t Map_t;
    Map_t::m_map[key] = ptr;
  }
};

}  // curves
}  // gtk

#endif  // MPBLOCKS_GTK_LAYOUTMAP_H_
