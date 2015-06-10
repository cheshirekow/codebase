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

#ifndef CK_GTK_LAYOUT_MAP_H_
#define CK_GTK_LAYOUT_MAP_H_

#include <map>
#include <string>
#include <gtkmm.h>
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

namespace ck_gtk {

/// base class for a map of string names -> widget pointers
template <class Widget_t>
class WidgetMapBase {
 public:
  /// the map type stored by this base class
  typedef std::map<std::string, Widget_t*> Map_t;

 protected:
  /// the actual map
  Map_t map_;

 public:
  /// adds the key to the map with an empty pointer
  void Init(const std::string& key);

  /// get widget pointers for all widgets whose names have been read in
  void Load(const Glib::RefPtr<Gtk::Builder>& builder);

  /// read in any values for our widgets which are stored in the
  /// node
  void Read(YAML::Node&);

  /// write out the values of our widgets into the yaml file
  void Write(YAML::Emitter&);
};

/// base class for a map of string names -> object RefPtrs
template <class Object_t>
class ObjectMapBase {
 public:
  /// the map type stored by this base class
  typedef std::map<std::string, Glib::RefPtr<Object_t> > Map_t;

 protected:
  /// the actual map
  Map_t map_;

 public:
  /// adds a key to the map with an empty reference
  void Init(const std::string& key);

  /// retrieves RefPtrs for all the objects whose keys have been
  /// entered
  void Load(const Glib::RefPtr<Gtk::Builder>& builder);

  /// read in values for any of the objects in our map
  void Read(YAML::Node&);

  /// write out the values for all of the objects in our map
  void Write(YAML::Emitter&);
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
  void LoadRecurse(tinyxml2::XMLElement*);

 public:
  /// load a GUI from a glade file, mapping each widget/object name
  /// to a reference/pointer to that object
  bool loadLayout(std::string layoutFile);

  /// read in values from a yaml file
  void LoadValues(std::string yamlFile);
  void LoadValues(YAML::Node& node);

  /// record values from all the widgets to the yaml file
  void SaveValues(std::string yamlFile);
  void SaveValues(YAML::Emitter& emitter);

  /// return a pointer to a widget
  template <class Widget_t>
  Widget_t* GetWidget(std::string key) {
    return WidgetMapBase<Widget_t>::map_[key];
  }

  /// return a RefPtr to an object
  template <class Object_t>
  Glib::RefPtr<Object_t> GetObject(std::string key) {
    return ObjectMapBase<Object_t>::map_[key];
  }

  /// unified getter
  template <class T>
  typename DerivedTypes<T>::Ptr_t Get(const std::string& key) {
    typedef typename DerivedTypes<T>::Map_t Base_t;
    typename Base_t::Map_t::iterator iter = Base_t::map_.find(key);
    if (iter != Base_t::map_.end()) {
      return iter->second;
    } else {
      return typename DerivedTypes<T>::Ptr_t(nullptr);
    }
  }

  /// unified setter
  template <class T>
  void Set(const std::string& key, typename DerivedTypes<T>::Ptr_t ptr) {
    typedef typename DerivedTypes<T>::Map_t Map_t;
    Map_t::map_[key] = ptr;
  }
};

}  // namespace ck_gtk

#endif  // CK_GTK_LAYOUT_MAP_H_
