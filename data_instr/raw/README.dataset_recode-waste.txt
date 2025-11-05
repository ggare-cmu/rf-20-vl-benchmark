# recode-waste-czvmg-fsod-yxsw > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/recode-waste-czvmg-fsod-yxsw-tpakw

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Aggregate](#aggregate)
  - [Cardboard](#cardboard)
  - [Hard Plastic](#hard-plastic)
  - [Metal](#metal)
  - [Soft Plastic](#soft-plastic)
  - [Timber](#timber)

# Introduction
This dataset is designed to facilitate the task of object detection for various materials commonly found in construction and waste management. The dataset includes the following classes: aggregate, cardboard, hard plastic, metal, soft plastic, and timber. Each class exhibits distinct visual characteristics allowing for precise categorization and annotation.

- **Aggregate**: Small stones or rocks
- **Cardboard**: Brown paper sheets or boxes
- **Hard Plastic**: Smooth, rigid structures that maintain their geometric shape
- **Metal**: Shiny or rusted rods, copper pipes or bolts
- **Soft Plastic**: Transparent or semi-transparent sheets or bags.
- **Timber**: Pieces of wood with cut or natural edges.

# Object Classes

## Aggregate
### Description
Aggregate refers to small, irregularly shaped stones or crushed rock pieces used in construction. These often have rough, jagged edges and a consistent color and texture variation.

### Instructions
- Annotate each piece of aggregate by drawing a bounding box around the visible stone. Focus on enclosing the entire stone even if partially occluded. 
- Do not label areas where the aggregate appears as part of another material or is indistinct.

## Cardboard
### Description
Cardboard is generally flat, with a fibrous texture and often displays folds, tears, or creases. It commonly appears in the shape of boxes or sheets and is made of brown paper.

### Instructions
- Draw bounding boxes around visible cardboard sections. Include entire sheets or box components if they are partially visible. 
- Avoid labeling heavily damaged pieces that blend into other materials or surfaces.

## Hard Plastic
### Description
Hard plastic items are typically smooth, rigid, and maintain geometric shapes like pipes or containers. They often have defined edges and may include extrusion features. Hard pastic objects have diverse visual appearances.

### Instructions
- Encapsulate each distinct hard plastic item with a bounding box, ensuring all visible edges are covered. 
- Do not label if only a fragment is visible without context or shape.

## Metal
### Description
Metal objects can be shiny or matte with smooth or textured surfaces, often exhibiting signs of rust or wear. They frequently have curved or angular shapes such as rods, copper pipes or bolts. Metal objects have diverse visual appearances.

### Instructions
- Draw bounding boxes around visible metal parts, ensuring to cover the entire object. 
- Do not label fragments lacking distinctive metallic features or visual context.

## Soft Plastic
### Description
Soft plastic is often transparent or semi-transparent, featuring a flexible, wrinkled appearance. It may be found in the form of bags or sheets. Soft plastic objects have diverse visual appearances.

### Instructions
- Outline visible sections of soft plastic with bounding boxes, including areas that show any defining texture or folds. 
- Do not annotate sections without clear boundaries.

## Timber
### Description
Timber pieces appear as elongated, rectangular wood sections with grain patterns visible on the surface. They may have cut or natural edges.

### Instructions
- Annotate full visible sections of timber with bounding boxes, including any visible grain or wood texture. 
- Avoid labeling small fragments that do not demonstrate clear timber characteristics.