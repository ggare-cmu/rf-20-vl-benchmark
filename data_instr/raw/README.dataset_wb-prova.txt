# wb-prova-stqnm-fsod-rbvg > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/wb-prova-stqnm-fsod-rbvg-5oscw

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Adult](#adult)
  - [Juvenile](#juvenile)
  - [Piglet](#piglet)

# Introduction

This dataset aims to classify different stages of growth within boars. The classes include:

- **Adult**: Fully grown boars.
- **Juvenile**: Young boars that are smaller than adults but still sizable.
- **Piglet**: Very young and small boars.

# Object Classes

## Adult

### Description
Adults are fully grown and large in size, typically taking up a significant portion of the image. They have well-defined features, such as distinct body and facial structures.

### Instructions
- Annotate the entire body, ensuring all visible parts are included.
- Do not include shadows or blurry sections that do not distinctly outline the animal.
- Focus only on clear, distinguishable features of adults.

## Juvenile

### Description
Juveniles are smaller than adults but significantly larger than piglets. They retain the body shape of adults but are not fully grown.

### Instructions
- Annotate visible juveniles, focusing on their smaller size relative to adults.
- Ensure the entire juvenile is included, without capturing unrelated nearby objects.
- Distinguish from adults by their notably smaller size and slightly less mature features.

## Piglet

### Description
Piglets are the smallest, typically found very close to the ground. They have a more compact and less developed body shape compared to adults and juveniles.

### Instructions
- Carefully annotate the small, rounded structure of piglets.
- Ensure to capture only the piglet, avoiding any overlap with other classes.
- Distinguish from juveniles by their more compact and infant-like size.