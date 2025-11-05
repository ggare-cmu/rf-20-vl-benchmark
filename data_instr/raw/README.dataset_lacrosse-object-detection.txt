# lacrosse-object-detection-fsod-uxkt > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/lacrosse-object-detection-fsod-uxkt-keltt

Provided by a Roboflow user
License: MIT

# Overview

- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Goalie](#goalie)
  - [Longpole](#longpole)
  - [Referee](#referee)
  - [Shortstick](#shortstick)

# Introduction

This dataset focuses on object detection in the sport of lacrosse. The goal is to accurately identify and annotate specific roles and equipment within the game. The classes include:

- **Goalie**: The player responsible for preventing the ball from entering the goal.
- **Longpole**: A player or item with a longer stick used primarily by defenders.
- **Referee**: The official who oversees the game and enforces the rules.
- **Shortstick**: A player or item with a shorter stick used primarily by offensive players.

# Object Classes

## Goalie

### Description

The goalie is typically positioned near the goal and wears protective gear, including a helmet and pads. This class is characterized by the proximity to the goal and the specific protective stance typical of a goalie. The goalie's stick typically has a wide round net.

### Instructions

- Annotate the area encompassing the whole individual, including the head, body, and any distinguishable goalie gear.
- Ensure the bounding box includes the player when they are in a typical guarding stance or engaged in a save attempt.
- Exclude any part of the goal structure or other players if they intersect with the goalie's space.

## Longpole

### Description

The longpole is identified by the player holding a visibly longer stick, which is used mainly in defensive roles. The stick length distinguishes this class from others.

### Instructions

- Annotate the whole player, centering on capturing the distinctively long stick.
- Include the full length of the pole to differentiate from a shortstick.
- Avoid labeling any players with short sticks or those without a visible long stick.

## Referee

### Description

Referees wear distinct uniforms typically featuring a black and white striped or yellow shirt and are easily recognizable by their positioning and activity on the field.

### Instructions

- Annotate the entire person, ensuring the bounding box includes the typical uniform and observable refereeing gestures.
- Capture the referee when they are in clear view, making calls, or moving to oversee the play.
- Do not include players or any goalies, only the referee in uniform.

## Shortstick

### Description

The shortstick pertains to players or equipment with a shorter stick used mainly by attackers and midfielders. These sticks are notably shorter than longpoles.

### Instructions

- Annotate the full player, ensuring the stick's length falls into the shorter category.
- Disambiguate from longpoles by focusing on the visibly shorter stick even when held at different angles.
- Exclude any players that appear with a long stick or those obstructed such that stick length is undetermined.