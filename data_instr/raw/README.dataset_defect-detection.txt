# defect-detection-yjplx-fxobh-fsod-amdi > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/defect-detection-yjplx-fxobh-fsod-amdi-hmafe

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Defective Fishplate](#defective-fishplate)
  - [Fastener](#fastener)
  - [Missing Fastener](#missing-fastener)
  - [Non Defective Fishplate](#non-defective-fishplate)

# Introduction
The dataset is focused on detecting defects in railway infrastructure. It includes images for identifying and classifying rail components into four categories: Defective Fishplate, Fastener, Missing Fastener, and Non Defective Fishplate. Each class is detailed below, highlighting the distinctive elements to consider for accurate annotation.

- **Defective Fishplate**: A flat piece of metal that joins two railway tracks that has visible damage or missing bolts.
- **Fastener**: A clamp that secures the rail to the railroad ties.
- **Missing Fastener**: A missing clamp that secures the rail to the railroad tie.
- **Non-Defective Fishplate**: A flat piece of metal that joins two railway tracks that does not have visible damage.

# Object Classes

## Defective Fishplate
### Description
A fishplate is a flat piece of metal used to join two lengths of railway track. A defective fishplate often has visible damage, misalignment, or missing connectors.

### Instructions
- Annotate the fishplate that shows signs of damage or misalignment.
- Include any visible deformations or cracks in your annotation.
- Do not annotate if the fishplate appears intact and undamaged.

## Fastener
### Description
Fasteners secure the rail to the railroad ties. They typically appear as small, noticeable clamps or clips attached directly to the rail.

### Instructions
- Annotate where the fasteners are present and properly securing the rail.
- Ensure that the annotated area includes the entire fastener and its attachment to the rail.
- Do not annotate areas where fasteners are missing or not visible.

## Missing Fastener
### Description
This class describes the absence of fasteners where they should be present along the rail.

### Instructions
- Identify and annotate areas on the rail where fasteners are supposed to be but are missing.
- Look for gaps between the rail and the ties where fasteners should be present.
- Do not annotate if there is an existing fastener correctly placed.

## Non Defective Fishplate
### Description
A non defective fishplate is a properly aligned and undamaged connector joining two rails.

### Instructions
- Annotate fishplates that are intact, with no signs of damage or displacement.
- Ensure the annotation covers the entire fishplate, including bolts and connections.
- Do not annotate if there are any visible defects or misalignments.