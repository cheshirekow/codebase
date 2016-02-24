#pragma once

#include <string>
#include <glog/logging.h>

#include "container_id.h"

namespace tap {

// hides away the type of a destination object and provides a common interface
// which statically depends only on the value type.
template <typename ValueType>
struct ValueConsumer {
 public:
  virtual ~ValueConsumer() {}

  // perform actual storage action, writing the value converted from a string
  // into the  destinaton object
  virtual void ConsumeValue(const ValueType& value) = 0;
};

// Stores a pointer to a scalar value and assigns to that value when it consumes
// arguments
template <typename ValueType, typename StorageType>
struct ScalarValueConsumer : public ValueConsumer<ValueType> {
 public:
  ScalarValueConsumer(StorageType* storage) : storage_(storage) {}

  void ConsumeValue(const ValueType& value) override {
    *storage_ = value;
  }

 private:
  StorageType* storage_;
};

// Stores a pointer to any fixed-sized iterable (i.e. c-style array, or
// std::array)
template <typename ValueType, typename Iterator>
struct ArrayValueConsumer : public ValueConsumer<ValueType> {
 public:
  ArrayValueConsumer(Iterator begin, Iterator end) : next_(begin), end_(end) {}

  void ConsumeValue(const ValueType& value) override {
    if (next_ != end_) {
      *(next_++) = value;
    } else {
      LOG(WARNING) << "Attempt to append a new value " << value
                   << " to a fixed sized container which is already full";
    }
  }

 private:
  Iterator next_;
  Iterator end_;
};

// Stores a pointer to a std container that supports push_back (i.e. a list, or
// vector)
template <typename ValueType, typename ContainerType>
struct BackValueConsumer : public ValueConsumer<ValueType> {
 public:
  BackValueConsumer(ContainerType* container) : container_(container) {}

  void ConsumeValue(const ValueType& value) override {
    container_->push_back(value);
  }

 private:
  ContainerType* container_;
};

// Stores a pointer to a std container that supports insert (i.e. a set)
template <typename ValueType, typename ContainerType>
struct InsertValueConsumer : public ValueConsumer<ValueType> {
 public:
  InsertValueConsumer(ContainerType* container) : container_(container) {}

  void ConsumeValue(const ValueType& value) override {
    container_->insert(value);
  }

 private:
  ContainerType* container_;
};

template <typename ValueType, class T, bool can_push_back_, bool can_insert_,
          bool has_iterators_>
struct ValueConsumerFactory {};

// If it's a container type that can push back we prefer to use that interface
template <typename ValueType, class T, bool can_insert_, bool has_iterators_>
struct ValueConsumerFactory<ValueType, T, true, can_insert_, has_iterators_> {
  static ValueConsumer<ValueType>* Create(T* container) {
    return BackValueConsumer<ValueType, T>(container);
  }
};

// If it's a container type without push_back but does hve insert, then use
// that interface
template <typename ValueType, class T, bool has_iterators_>
struct ValueConsumerFactory<ValueType, T, false, true, has_iterators_> {
  static ValueConsumer<ValueType>* Create(T* container) {
    return InsertValueConsumer<ValueType, T>(container);
  }
};

// If it's a container type with neither push_back nor insert but has a
// begin(), end() pair, then use those
template <typename ValueType, class T>
struct ValueConsumerFactory<ValueType, T, false, false, true> {
  static ValueConsumer<ValueType>* Create(T* container) {
    return ArrayValueConsumer<ValueType, T>(container->begin(),
                                            container->end());
  }
};

// If it's not any recognizable container, then assume it's a scalar and
// use the scalar consumer
template <typename ValueType, class T>
struct ValueConsumerFactory<ValueType, T, false, false, false> {
  static ValueConsumer<ValueType>* Create(T* storage) {
    return ScalarValueConsumer<ValueType, T>(storage);
  }
};

template <typename ValueType, class T>
ValueConsumer<ValueType>* CreateValueConsumer(T* dest) {
  return ValueConsumerFactory<ValueType, T,
                              can_push_back<T, void(ValueType)>::value,
                              can_insert<T, void(ValueType)>::value,
                              has_iterators<T>::value>::Create(dest);
}

}  // namespace tap
