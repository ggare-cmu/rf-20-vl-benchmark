# dentalai-i4clz-fsod-fsuo > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/dentalai-i4clz-fsod-fsuo-zuruj

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Cavity](#cavity)
  - [Fillings](#fillings)
  - [Impacted Tooth](#impacted-tooth)
  - [Implant](#implant)

# Introduction
This dataset focuses on detecting dental conditions using x-ray images. The task is to identify and annotate specific dental features. The classes include:
- **Cavity**: Areas of decay on a tooth.
- **Fillings**: Restorations placed to treat cavities.
- **Impacted Tooth**: A tooth prevented from erupting properly.
- **Implant**: Artificial tooth roots placed into the jawbone.

# Object Classes

## Cavity
### Description
Cavities appear as dark or shadowy areas on the tooth, often with irregular edges.

### Instructions
- Annotate areas with a distinct dark appearance on the tooth surface.
- Ensure the bounding box includes the entire dark region but does not extend into healthy tooth tissue.
- Do not annotate stains or discoloration that do not penetrate into enamel or dentine layers.

## Fillings
### Description
Fillings are bright, opaque areas within the structure of a tooth, indicating dental restorative material. Fillings are typically on the crown of the tooth.

### Instructions
- Draw bounding boxes around bright, distinct areas within a tooth that indicate filling material.
- Ensure differentiation from natural tooth structure by identifying a noticeable brightness.
- Avoid annotating areas where brightness is due to x-ray exposure artifacts.

## Impacted Tooth
### Description
An impacted tooth is positioned awkwardly, often tilted or trapped beneath the gum or bone.

### Instructions
- Identify teeth that are not aligned with the dental arch and annotate them fully within their misplaced domain.
- Make sure to include the entire visible structure of the impacted tooth in the annotation.
- Do not label teeth that might appear slightly misaligned but are not obstructed.

## Implant
### Description
Implants appear as thin metallic, smooth, and densely bright artifacts in the jaw with cylindrical structures.

### Instructions
- Encapsulate the entire metallic, dense structure representing the implant.
- Confirm the presence of uniform brightness that suggests metallic material.
- Exclude any natural tooth roots or other bright artifacts that are not implants.