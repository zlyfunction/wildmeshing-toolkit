#pragma once
#include <nlohmann/json.hpp>
namespace {

nlohmann::json delaunay_spec = R"(
[
  {
    "pointer": "/",
    "type": "object",
    "required": [
      "input",
      "pos_attr_name",
      "vertex_label",
      "edge_label",
      "face_label",
      "input_values",
      "output_value",
      "output"
    ],
    "optional": ["pass_through", "report", "input_path"]
  },
  {
    "pointer": "/input",
    "type": "string"
  },
  {
    "pointer": "/pos_attr_name",
    "type": "string",
    "doc": "The vertex double attribute which stores the vertex positions."
  },
  {
    "pointer": "/vertex_label",
    "type": "string",
    "doc": "The vertex int64_t attribute which is used for deciding if an edge should be split."
  },
  {
    "pointer": "/edge_label",
    "type": "string",
    "doc": "edge int64_t attribute."
  },
  {
    "pointer": "/face_label",
    "type": "string",
    "doc": "face int64_t attribute."
  },
  {
    "pointer": "/input_values",
    "type": "list",
    "doc": "List of vertex labels that are considered in the split",
    "min": 1,
    "max": 2
  },
  {
    "pointer": "/input_values/*",
    "type": "int",
    "doc": "input value"
  },
  {
    "pointer": "/output_value",
    "type": "int",
    "doc": "The label that is assigned to new vertices emerging from a split, i.e. vertices that are on the isosurface."
  },
  {
    "pointer": "/output",
    "type": "string"
  },
  {
    "pointer": "/pass_through",
    "type": "list",
    "default": [],
    "doc": "all attributes that are not deleted by the component but also not required"
  },
  {
    "pointer": "/pass_through/*",
    "type": "string"
  },
  {
    "pointer": "/report",
    "type": "string",
    "default": ""
  },
  {
    "pointer": "/input_path",
    "type": "string",
    "default": ""
  }
]
)"_json;

}
