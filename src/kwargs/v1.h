/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of kwargs.
 *
 *  kwargs is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  kwargs is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with kwargs.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @author Josh Bialkowski <josh.bialkowski@gmail.com>
 *  @brief tooling for python-like kwargs in c++
 */
#ifndef KWARGS_H_
#define KWARGS_H_

#include <cstdint>

/// classes and templates used by kwargs library
namespace kw {

/// parameter storage for type T within a parameter pack
template <uint64_t Tag, typename T>
struct Arg {
  T v;
  Arg(T v) : v(v) {}
  constexpr bool IsTagged(uint64_t query){ return query == Tag; }
};

/// returns a 64bit hash of @p string
template<uint32_t N>
inline constexpr uint64_t Hash(const char (&string)[N]);

/// signifies that the parameter should be passed by reference
template <typename T>
struct RefWrap {
  T& v;
  RefWrap(T& v) : v(v) {}
};

/// signifies the parameter should be passed by const reference
template <typename T>
struct ConstRefWrap {
  const T& v;
  ConstRefWrap(const T& v) : v(v) {}
};

/// forces an argument to be passed by reference
template <typename T>
RefWrap<T> Ref(T& v){ return RefWrap<T>(v); }

/// forces an argument to be passed by const reference
template <typename T>
const ConstRefWrap<T> ConstRef(const T& v) { return ConstRefWrap<T>(v); }

/// assignment operator sentinal used as a key for key-values pairs in the
/// kwargs parameter pack
template <uint64_t Tag>
struct Key {
	template <typename T>
	Arg<Tag,T> operator=(T v) { return v; }

  template <typename T>
	Arg<Tag,T&> operator=(RefWrap<T> vw) { return vw.v; }

  template <typename T>
	Arg<Tag,const T&> operator=(ConstRefWrap<T> vw) { return vw.v; }
};

/// template meta-function contains a static boolean variable 'result' which
/// is true if Tag is in the list of kwargs and false if not
template <uint64_t Tag, typename... Args>
struct ContainsTag {};

/// template meta function provides a member typedef Result which evaluates
/// to T if Arg<Tag,T> is in Args... or Default if it is not
template <uint64_t Tag, typename Default, typename... Args>
struct TypeOfTagDefault {};

template <typename... Args>
struct ParamPack;

/// provides static member Get() with implementations depending on whether 
/// or not Tag is in Args...
template <uint64_t Tag, bool exists, typename Default, typename... Args>
struct GetImpl{};

/// given a parameter pack, retrieves and return sthe parameter tagged with tag,
/// or else returns a default value if tag is not in Args...
template <uint64_t Tag, typename Default, typename... Args>
inline typename TypeOfTagDefault<Tag,Default,Args...>::Result
    Get(ParamPack<Args...>& pack, Default d) {
  return 
    GetImpl<Tag,ContainsTag<Tag,Args...>::result,Default,Args...>::Get(pack,d);
}

/// given a parameter pack, retrieves and return sthe parameter tagged with tag,
/// or else returns a default value if tag is not in Args...
template <uint64_t Tag, typename Default, typename... Args>
inline typename TypeOfTagDefault<Tag,Default,Args...>::Result
    Get(ParamPack<Args...>& pack, const Key<Tag>& key, Default d) {
  return 
    GetImpl<Tag,ContainsTag<Tag,Args...>::result,Default,Args...>::Get(pack,d);
}

/// Return an unsigned 64bit hash of @p string
inline uint64_t HashString(const std::string& str, int i,
                                     uint64_t hash) {
  return i == str.size() ? hash :
          HashString(str, i + 1, ((hash << 5) ^ (hash >> 27)) ^ str[i]);
}

inline uint64_t HashString(const std::string& str) {
  return HashString(str, 0, str.size());
}

/// storage for kwargs parameter pack
template <typename... Args>
struct ParamPack{

  /// returns true if the numeric tag is a member of the kwargs in Args...
  template<uint64_t Tag>
  constexpr bool Contains(const Key<Tag>& key) {
    return ContainsTag<Tag, Args...>::result;
  }

  /// return the value associated with Tag, if it exists in the kwargs,
  /// otherwise return the default
  template<uint64_t Tag, typename T>
  typename TypeOfTagDefault<Tag, T, Args...>::Result Get(const Key<Tag>& key,
                                                         T default_value) {
    return kw::Get(*this, key, default_value);
  }

  template<typename ReturnType>
  ReturnType GetAs(uint64_t tag, ReturnType default_value) {
    return default_value;
  }

  template<typename ReturnType>
  ReturnType GetAs(const std::string& tag, ReturnType default_value) {
    return GetAs(HashString(tag),default_value);
  }

  bool Contains(uint64_t tag){ return false; }
  bool Contains(const std::string& tag){ return Contains(HashString(tag)); }
};



/// constexpr string, used to implement tagging with strings
/**
 *  via Scott Schurr's C++ Now 2012 presentation
 */
class Tag {
 public:
  template<uint32_t N>
  /// Construct from a string literal
  constexpr Tag(const char (&string)[N])
      : m_ptr(string),
        m_size(N - 1) {
  }

  constexpr Tag(const char* string, size_t len)
      : m_ptr(string),
        m_size(len) {
  }

  /// return the character at index i
  constexpr char operator[](uint32_t i) {
    // if we care about overflow
    // return n < m_size ? m_ptr[i] : throw std::out_of_range("");
    return m_ptr[i];
  }

  /// return the number of characters in the string, including the terminal
  /// null
  constexpr uint32_t size() {
    return m_size;
  }

  /// Donald Knuth's hash function
  /// TODO: validate or replace this hash function
  constexpr uint64_t Hash(int i, uint64_t hash) {
    return i == size() ? hash :
        Hash(i + 1, ((hash << 5) ^ (hash >> 27)) ^ m_ptr[i]);
  }

  /// return a hash of the string
  constexpr uint64_t Hash(){
    return Hash(0,size());
  }

 private:
  const char* const m_ptr;
  const uint32_t m_size;
};

/// Return an unsigned 64bit hash of @p string
template<uint32_t N>
inline constexpr uint64_t Hash(const char (&string)[N]){
  return Tag(string).Hash();
}

}  //< namespace kw


// ----------------------------------------------------------------------------
//   Danger!!! Horrible implementation details below!!!
//   Continue at your own risk!
// ----------------------------------------------------------------------------

#ifndef DOXYGEN_IGNORE

namespace kw {

/// SFINAE to safely iterate over params in a parampack since we cannot use
/// the ?: operator on any old pairs of types
template <typename T, typename ReturnType>
ReturnType SafeReturn(T query, ReturnType default_value){
  return query;
}

/// specialization for recursion, provides storage for the current type in the
/// type list, then recursively derives from the remaining types in the type
/// list
template <typename Head, typename... Tail>
struct ParamPack<Head,Tail...>
  : Head,
    ParamPack<Tail...> {
  ParamPack( Head head, Tail... tail )
    : Head(head),
      ParamPack<Tail...>(tail...) {}

  template <uint64_t Tag>
  constexpr bool Contains(const Key<Tag>& key) const{
    return ContainsTag<Tag,Head,Tail...>::result;
  }

  template<uint64_t Tag, typename T>
    typename TypeOfTagDefault<Tag,T,Head,Tail...>::Result
    Get(const Key<Tag>& key, T default_value) {
      return kw::Get<Tag>(*this, key, default_value);
  }

  template<typename ReturnType>
  ReturnType GetAs(uint64_t tag, ReturnType default_value) {
    return
        Head::IsTagged(tag) ?
            this->Head::v : ParamPack<Tail...>::GetAs(tag, default_value);
  }

  template<typename ReturnType>
  ReturnType GetAs(const std::string& string, ReturnType default_value) {
    return GetAs(HashString(string),default_value);
  }

  bool Contains(uint64_t tag) {
    return Head::IsTagged(tag) ? true : ParamPack<Tail...>::Contains(tag);
  }

  bool Contains(const std::string& tag) {
    return Contains(HashString(tag));
  }
};

/// specialization for base case, provides storage for the last type in the
/// type list
template <typename Tail>
struct ParamPack<Tail> : Tail {
  ParamPack( Tail tail ) : Tail(tail) {}

  template <uint64_t Tag>
  constexpr bool Contains(const Key<Tag>& key) const{
    return ContainsTag<Tag,Tail>::result;
  }

  template<uint64_t Tag, typename T>
    typename TypeOfTagDefault<Tag,T,Tail>::Result
    Get(const Key<Tag>& key, T default_value) {
      return kw::Get<Tag>(*this, key, default_value);
  }

  template<typename ReturnType>
  ReturnType GetAs(uint64_t tag, ReturnType default_value) {
    return Tail::IsTagged(tag) ? this->Tail::v : default_value;
  }

  template<typename ReturnType>
  ReturnType GetAs(const std::string& string, ReturnType default_value) {
    return GetAs(HashString(string), default_value);
  }

  bool Contains(uint64_t tag) {
      return Tail::IsTagged(tag) ? true : false;
  }

  bool Contains(const std::string& tag) {
    return Contains(HashString(tag));
  }
};

/// Function version of ContainsTag
inline constexpr bool ContainsTagFn(uint64_t tag){
  return false;
}

template <uint64_t Tag, typename T, typename... Rest>
inline constexpr bool ContainsTagFn(uint64_t tag, Arg<Tag,T> first, Rest... rest){
  return (tag==Tag) ? true : ContainsTagFn(tag,rest...);
}

/// specialization for recursion
template <uint64_t Tag, typename First, typename... Rest>
struct ContainsTag<Tag,First,Rest...> {
  static const bool result = ContainsTag<Tag, Rest...>::result;
};

/// specialization for when the Arg<Tag,T> type is found
template <uint64_t Tag, typename T, typename... Rest>
struct ContainsTag<Tag, Arg<Tag,T>, Rest...> {
  static const bool result = true;
};

/// specialization for when Tag is not in the type list
template <uint64_t Tag>
struct ContainsTag<Tag> {
  static const bool result = false;
};

/// specialization for recursion
template <uint64_t Tag, typename Default, typename Head, typename... Tail>
struct TypeOfTagDefault<Tag, Default, Head, Tail...> {
  typedef typename TypeOfTagDefault<Tag, Default, Tail...>::Result Result;
};

/// specialization for when Arg<Tag,T> is found
template <uint64_t Tag, typename Default, typename T, typename... Tail>
struct TypeOfTagDefault<Tag, Default, Arg<Tag, T>, Tail...> {
  typedef T Result;
};

/// specialization for when Arg<Tag,T> is not in the type list
template <uint64_t Tag, typename Default>
struct TypeOfTagDefault<Tag,Default> {
  typedef Default Result;
};

/// specialization for when Tag is not in Args..., returns the default value
template <uint64_t Tag, typename Default, typename... Args>
struct GetImpl<Tag,false,Default,Args...>{
  static inline Default Get(ParamPack<Args...>& pack, Default d) {
    return d;
  }
};

/// specialization for when Tag is in Args... returns the value corresponding
/// to Tag
template <uint64_t Tag, typename Default, typename... Args>
struct GetImpl<Tag,true,Default,Args...>{
  static inline typename TypeOfTagDefault<Tag,Default,Args...>::Result Get(
      ParamPack<Args...>& pack, Default d) {
    typedef typename TypeOfTagDefault<Tag,Default,Args...>::Result StorageType;
    return static_cast<Arg<Tag,StorageType>&>(pack).v;
  }
};

}  //< namespace kw

#define TAG(key) kw::Hash(#key)
#define KW(key) kw::Key<kw::Hash(#key)>()

#endif  // DOXYGEN_IGNORE
#endif  // KWARGS_H_
