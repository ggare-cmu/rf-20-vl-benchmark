# x-ray-id-zfisb-fsod-dyjv > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/x-ray-id-zfisb-fsod-dyjv-olpha

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [DIP](#dip)
  - [MCP](#mcp)
  - [PIP](#pip)
  - [Radius](#radius)
  - [Ulna](#ulna)
  - [Wrist](#wrist)

# Introduction
This dataset provides X-ray images of human hands for medical studies. It aims to segment and identify key anatomical features in hand radiographs. The classes are:
- **DIP**: Distal Interphalangeal joint, the farthest joint in the fingers.
- **MCP**: Metacarpophalangeal joint, located at the base of each finger.
- **PIP**: Proximal Interphalangeal joint, the middle joint in the fingers.
- **Radius**: One of the two main bones in the forearm, on the thumb side.
- **Ulna**: The second forearm bone, located on the pinky side.
- **Wrist**: Carpal bones at the base of the hand connecting to the forearm.

# Object Classes

## DIP
### Description
The DIP (or Distal Interphalangeal) joint is located near the tip of the finger, between the last two phalanges. It appears as a small gap representing the joint space on the X-ray.

### Instructions
- Focus on the end joints of each digit.
- Ensure the bounding box encapsulates the visible joint space of the DIP.
- Avoid labeling any part of the bone shafts; the focus is on the joint itself.

## MCP
### Description
The MCP (or Metacarpophalangeal) joint is found at the knuckle, connecting fingers to the palm. It appears as a prominent gap near the base of each finger.

### Instructions
- Identify the knuckle joints visible where the fingers meet the palm.
- Draw the bounding box around the joint area, including any visible joint space.
- Exclude the phalange or metacarpal bone sections; only annotate the joint gap.

## PIP
### Description
The PIP (or Proximal Interphalangeal) joint is located between the first and second phalanges, appearing as a clear gap between the finger bones. It is below the DIP and above the MCP.

### Instructions
- Locate the middle joint in each finger.
- Capture the visible gap between the bones, indicative of the PIP joint.
- Do not extend the annotation to cover the phalanges themselves.

## Radius
### Description
The Radius is the thicker and shorter of the two forearm bones, located on the thumb side. It appears as a robust and continuous bone on the X-ray.

### Instructions
- Identify the forearm bone on the thumb side, distinguishable by its thicker dimensions.
- Cover the entire visible section of the Radius, from wrist to near the elbow.
- Ensure surrounding soft tissues or overlapping bones are not included.

## Ulna
### Description
The Ulna runs parallel to the Radius and is typically longer and thinner, extending to the elbow.

### Instructions
- Locate the forearm bone on the side opposite the thumb.
- Draw the bounding box to include the full visible length of the Ulna.
- Distinguish it from the Radius to avoid overlap in the annotation.

## Wrist
### Description
The Wrist comprises multiple small carpal bones at the base of the hand, forming the connection to the forearm.

### Instructions
- Focus on the cluster of bones at the base of the palm.
- Surround all visible carpal bones in the bounding box.
- Avoid marking the Radius or Ulna sections extending into the hand.

[Top](#overview)