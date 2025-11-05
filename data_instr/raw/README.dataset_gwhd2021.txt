# gwhd2021-fsod-atsv > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/gwhd2021-fsod-atsv-6sfuy

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Wheat Head](#Wheat-Head)

# Introduction
The dataset focuses on detecting wheat heads in agricultural images to assist with monitoring and analysis. It contains annotations for the following class:

- **Wheat Heads**: Wheat heads are the parts of the wheat plant where the grains are formed.

# Object Classes

## Wheat Head

### Description
The wheat head is part of the wheat plant, distinguished by its elongated shape and spikelets containing grains. They may appear amidst leaves and stems, often seen in dense clusters or isolated. Wheat heads may be green or yellow in color depending on their moisture content.

### Instructions
- Annotate the entire wheat head, capturing the full elongated shape from the base near the stem to the tip, even if partially obscured by other parts of the plant.
- Ensure the bounding box includes the entire visible head while excluding leaves and stems unless they are integral to the shape.
- Do not annotate if the head is less than 20% visible, or if it is unclear whether the object is a wheat head.
- Avoid annotating overly blurred objects that cannot be confidently identified as wheat heads.
- Cross-referencing other instances can help disambiguate partially visible wheat heads.