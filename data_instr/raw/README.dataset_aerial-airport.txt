# aerial-airport-7ap9o-fsod-ddgc > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/aerial-airport-7ap9o-fsod-ddgc-4qt0q

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Airplane](#airplane)

# Introduction
This dataset is designed to support the identification and localization of airplanes in aerial images. The primary task is object detection, focusing on identifying airplanes based on their distinct shapes and structures as seen from above. The dataset consists of a single class:

- **Airplane**: Represents aircrafts parked or moving on an airport's tarmac.

# Object Classes

## Airplane

### Description
Airplanes in this dataset are typically positioned on airport runways and near terminal buildings. Distinctive features include a central fuselage, two wings extending from the sides, and a vertical tail fin. These elements should be clearly identifiable from an aerial perspective.

### Instructions
- Draw a box around the entire airplane, ensuring that all main components such as the fuselage, wings, and tail are within the box. If part of an airplane is obscured by other objects or cut off by image boundaries, estimate the full extent of the relevant parts based on visible clues. 
- Only annotate objects that can be confidently identified as airplanes. Blurry or indistinct shapes should not be annotated. 
- Ensure that multiple airplanes in close proximity are individually boxed, without overlapping annotations. 
- Avoid annotating any other objects, shadows, or reflections that might appear similar but do not fit the described distinctive airplane features.