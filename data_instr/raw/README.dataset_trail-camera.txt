# trail-camera-fsod-egos > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/trail-camera-fsod-egos-ztdfw

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Deer](#deer)
  - [Hog](#hog)

# Introduction
This dataset is designed for object detection using trail camera images. It includes two classes of animals commonly found in the wild: Deer and Hog. The objective is to accurately identify and annotate these animals within the images.

# Object Classes

## Deer
### Description
Deer are large herbivorous mammals characterized by their slender bodies and long legs. They often have antlers, particularly the males, which make them distinguishable. Their faces are elongated, and they typically have intermediate-sized ears.

### Instructions
- **Bounding Box**: Annotate the entire visible body of the deer, ensuring the bounding box includes the head and limbs. Include antlers, even if partially occluded.
- **Visibility**: If only part of the deer is visible, annotate based on the visible portion, extending the box as needed for occluded antlers.
- **Exclusions**: Do not annotate if only a vague outline is discernible, and no distinct features such as legs or head are visible.

## Hog
### Description
Hogs are stocky, muscular animals with short legs and broad bodies. They have elongated snouts and small, pointed ears. Their bodies are generally more rounded, contrasting with the slender profile of deer.

### Instructions
- **Bounding Box**: Capture the full visible body within the box, including the distinct rounded back and snout of the hog.
- **Visibility**: Annotate based on visible parts. If the hog is partially obscured by foliage, still capture the recognizable features like the snout.
- **Exclusions**: Do not annotate if the hog is too obscured to identify specific features or if it is evident only by vague movement or noise.