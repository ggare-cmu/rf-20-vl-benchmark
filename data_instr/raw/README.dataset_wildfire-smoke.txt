# wildfire-smoke-fsod-myxt > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/wildfire-smoke-fsod-myxt-tided

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Smoke](#smoke)

# Introduction
This dataset is focused on identifying wildfire smoke in outdoor images. The primary task is to detect and annotate smoke plumes, providing accurate data for monitoring and analyzing wildfire smoke spread and impact. The dataset consists of images with the following class:

- **Smoke**: Visual representation of smoke plumes rising from potential wildfire events.

# Object Classes

## Smoke

### Description
Smoke in this dataset appears as cloud-like formations, typically originating from the ground or behind a ridge, and extending upwards or horizontally. It can vary in density, appearing either wispy or thick and obscuring background details such as mountains or trees.

### Instructions
- Annotate the entire visible extent of the smoke plume. The annotation should encompass both dense, opaque sections and lighter, more diffuse areas.
- Include smoke that may be partially obscured by foreground objects like trees or structures.
- Do not annotate clouds; focus only on formations directly connected to a potential ground source.
- If the smoke is cut off at the edge of an image, end the annotation at the image boundary.
- Ensure that annotations follow the shape of the smoke, capturing its undulating and irregular patterns.
- Avoid labeling faint atmospheric haze not clearly distinguishable as a smoke source. Only annotate smoke that is clearly visible and identifiable.
- In cases where smoke appears to blend with clouds, use context clues like a clear origin point or directional flow to differentiate and annotate accurately.