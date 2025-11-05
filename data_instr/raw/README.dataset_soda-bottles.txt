# soda-bottles-fsod-haga > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/soda-bottles-fsod-haga-1d5c6

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Coca-Cola](#coca-cola)
  - [Fanta](#fanta)
  - [Sprite](#sprite)

# Introduction
This dataset is designed for object detection with the goal of identifying different brands of soda bottles within refrigerator environments. The classes included are:
- **Coca-Cola**: Bottles labeled as Coca-Cola.
- **Fanta**: Orange soda bottles labeled as Fanta.
- **Sprite**: Green bottles labeled as Sprite.

# Object Classes

## Coca-Cola
### Description
Coca-Cola bottles are typically dark in color with a distinctive label. The label features easily identifiable red branding. The shape of the bottle is usually symmetrical and cylindrical with a rounded red cap.

### Instructions
- Annotate the entire visible Coca-Cola bottle, including the label and cap.
- Ensure the bounding box includes both the top of the bottle cap and the bottom of the bottle, following the silhouette.
- Exclude any parts of the bottle that are occluded unless the shape is clearly discernible.
- Do not annotate reflections or images on other objects that feature the Coca-Cola logo.

## Fanta
### Description
Fanta bottles are characterized by their vibrant, orange color and rounded shape. The bottles have a label indicating the Fanta logo. The cap is typically orange.

### Instructions
- Draw a bounding box around the entire visible part of the Fanta bottle, capturing its cylindrical shape and cap.
- Ensure the bounding box is tight to the bottle's shape, including the entirety of the cap.
- Exclude occluded parts unless you can clearly infer the shape of the bottle.
- Do not annotate reflections or labels that aren’t directly on an identifiable Fanta bottle.

## Sprite
### Description
Sprite bottles are normally green, with a distinct label displaying the Sprite logo. The bottles have a slender, uniform shape with a green cap.

### Instructions
- Annotate the full visible Sprite bottle, ensuring the label and bottle shape are within the bounding box.
- The bounding box should include the full cap and base of the bottle.
- If only partially visible, annotate only when the bottle's shape and branding are clearly identifiable.
- Avoid labeling reflections or any indirect depictions of the Sprite bottle logo.