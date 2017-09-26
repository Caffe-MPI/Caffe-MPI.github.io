/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

class LayerBase;

template<template <typename Ftype, typename Btype> class LayerType>
inline shared_ptr<LayerBase> CreateLayerBase(const LayerParameter& param,
    Type ftype, Type btype) {
  bool failed = false;
  shared_ptr<LayerBase> ptr;
  if (ftype == FLOAT) {
    if (btype == FLOAT) {
      ptr.reset(new LayerType<float, float>(param));
    }
#ifndef CPU_ONLY
    else if (btype == FLOAT16) {
      ptr.reset(new LayerType<float, float16>(param));
    }
#endif
    else if (btype == DOUBLE) {
      ptr.reset(new LayerType<float, double>(param));
    } else {
      failed = true;
    }
  }
#ifndef CPU_ONLY
  else if (ftype == FLOAT16) {
    if (btype == FLOAT) {
      ptr.reset(new LayerType<float16, float>(param));
    } else if (btype == FLOAT16) {
      ptr.reset(new LayerType<float16, float16>(param));
    } else if (btype == DOUBLE) {
      ptr.reset(new LayerType<float16, double>(param));
    } else {
      failed = true;
    }
  }
#endif
  else if (ftype == DOUBLE) {
    if (btype == FLOAT) {
      ptr.reset(new LayerType<double, float>(param));
    }
#ifndef CPU_ONLY
    else if (btype == FLOAT16) {
      ptr.reset(new LayerType<double, float16>(param));
    }
#endif
    else if (btype == DOUBLE) {
      ptr.reset(new LayerType<double, double>(param));
    } else {
      failed = true;
    }
  } else {
    failed = true;
  }

  if (failed) {
    LOG(FATAL) << "Combination of layer types " << Type_Name(ftype) << " and "
        << Type_Name(btype) << " is not currently supported "
        << "(discovered in layer '" << param.name() << "').";
  }
  CHECK_NOTNULL(ptr.get());
  return ptr;
}

class LayerRegistry {
 public:
  typedef shared_ptr<LayerBase> (*Creator)(const LayerParameter&, Type, Type);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry g_registry_;
    return g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  static shared_ptr<LayerBase> CreateLayer(const LayerParameter& param) {
    const string& layer_type = param.type();
    const string& layer_name = param.name();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating layer '" << layer_name << "' of type '" << layer_type << "'";
    }
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(layer_type), 1) << "Unknown layer type: '" << layer_type
        << "' (known types: " << LayerTypeListString() << ")";

    //  We compose these types in Net::Init
    Type ftype = param.forward_type();
    Type btype = param.backward_type();
    Type fmath = param.forward_math();
    Type bmath = param.backward_math();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Layer's types are Ftype:" << Type_Name(ftype)
          << " Btype:" << Type_Name(btype)
          << " Fmath:" << Type_Name(fmath)
          << " Bmath:" << Type_Name(bmath);
    }
    return registry[layer_type](param, ftype, btype);
  }

  static vector<string> LayerTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry() {}

  static string LayerTypeListString() {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};

class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
      shared_ptr<LayerBase> (*creator)(const LayerParameter&, Type, Type)) {
    LayerRegistry::AddCreator(type, creator);
  }
};

#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer g_creator_##type(#type, creator);

#define REGISTER_LAYER_CLASS(type)                                             \
  shared_ptr<LayerBase> Creator_##type##Layer(const LayerParameter& param,     \
                                              Type ftype, Type btype)          \
  {                                                                            \
    return CreateLayerBase<type##Layer>(param, ftype, btype);                  \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

void check_precision_support(Type& ftype, Type& btype, LayerParameter& param);

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
