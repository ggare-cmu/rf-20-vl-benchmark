# the-dreidel-project-anzyr-fsod-zejm > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/the-dreidel-project-anzyr-fsod-zejm-fwv5t

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Dreidel](#dreidel)
  - [Gimel](#gimel)
  - [Hay](#hay)
  - [Nun](#nun)
  - [Shin](#shin)
  - [Spinning Dreidel](#spinning-dreidel)

# Introduction
This dataset focuses on identifying dreidels and their associated Hebrew letters. The task is to annotate each occurrence of a dreidel and its visible letter. The classes are:
- **Dreidel**: The traditional spinning top.
- **Gimel**: A Hebrew letter found on one side of a dreidel that looks like the greek letter lambda.
- **Hay**: A Hebrew letter found on another side of a dreidel that looks like a horizontal line and two vertical lines with a gap between the horizontal line and one of the vertical lines.
- **Nun**: A Hebrew letter found on another side of a dreidel that looks like a horizontal line and two vertical lines.
- **Shin**: A Hebrew letter found on another side of a dreidel that looks like the letter W.
- **Spinning Dreidel**: A dreidel in motion, appearing blurred.

# Object Classes

## Dreidel
### Description
A wooden or plastic spinning top with four flat sides. It has a handle on top and a pointed bottom.

### Instructions
- Annotate the whole dreidel, including the handle and bottom point.
- Do not include the shadow or reflections.

## Gimel
### Description
A Hebrew letter resembling a figure with two branches. It looks similar to the greek letter lambda.

### Instructions
- Annotate the visible Gimel letter on the dreidel side.
- Ensure the annotation includes the entire character, not parts of other letters.

## Hay
### Description
A Hebrew letter with a horizonal line and two vertical strokes. Note that there is a gap between the top of the horizontal line and the second vertical stroke

### Instructions
- Focus the annotation on the full visible Hay letter.
- Avoid overlapping with other letters or design elements.

## Nun
### Description
A horizontal line with two vertical strokes. Unlike Hay, there is no gap between the vertical and horizontal lines.

### Instructions
- Capture the complete visible Nun letter.
- Exclude any background or other distracting elements.

## Shin
### Description
A letter with three vertical stems connected at the base. It looks like the letter W. 

### Instructions
- Annotate the visible Shin letter fully.
- Ensure clarity by avoiding other letter parts or ornaments.

## Spinning Dreidel
### Description
A dreidel in motion, appearing blurred due to its spin.

### Instructions
- Annotate the entire blurred shape of the spinning dreidel.
- Do not label if the blur obscures recognition of the dreidel form.