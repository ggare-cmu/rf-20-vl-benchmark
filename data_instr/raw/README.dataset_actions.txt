# actions-zzid2-zb1hq-fsod-amih > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/actions-zzid2-zb1hq-fsod-amih-ecd3n

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Attack](#attack)
  - [Block](#block)
  - [Defense](#defense)
  - [Serve](#serve)
  - [Set](#set)
  - [Ball](#ball)


# Introduction
This dataset is used for recognizing volleyball actions and objects. It includes six classes: Attack, Block, Defense, Serve, Set, and Ball. The aim is to annotate instances of these actions and objects in volleyball games to assist in automated sports analysis.

- **Attack**: Players hit the ball over the net.
- **Block**: Players deflect an opponents attempt to hit the ball over the net.
- **Defense**: Players adopt a low stance.
- **Serve**: Players hit the ball from behind the end line.
- **Set**: Players push the ball upwards with their fingertips.
- **Ball**: The volleyball.
 
# Object Classes

## Attack
### Description
The "Attack" action involves a player attempting to hit the ball forcefully over the net to score a point, usually characterized by an airborne leap and raised arm with the palm facing forward.

### Instructions
- Annotate the player as soon as their body is in mid-air with an arm extended above the head with an intention to strike the ball.
- Ensure the bounding box includes the full extent of the player’s body, anticipating motion, but exclude the ball unless it directly touches the player’s hand at the marking moment.

## Block
### Description
"Block" involves players reaching up near the net with both hands raised above the head to stop or deflect an opponent's attack.

### Instructions
- Capture players with both hands above their head, with palms facing the incoming ball.
- Ensure the bounding box contains the player’s arms and hands, particularly positioned near the net area, ready to block.

## Defense
### Description
"Defense" is characterized by players adopting a low stance, often with forearms parallel to the floor, ready to receive the ball from an opponent’s attack.

### Instructions
- Annotate players crouched with arms arranged to form a platform, usually positioned behind the front line.
- Don’t include players in a standing position unrelated to an immediate ball-defense action.

## Serve
### Description
"Serve" involves a player initiating play by striking the ball from behind the end line to send it over the net.

### Instructions
- Identify a player behind the end line, typically poised to toss and hit or having just made contact with the ball.
- Include the full player body preparing to serve, with a focus on hand positioning relative to the ball pre and post-contact.

## Set
### Description
A "Set" involves a player using their fingertips to push the ball upwards to set up a spike, often positioned right in front of the net.

### Instructions
- Box the player with arms elevated and fingers spread, set under the ball, usually just after a teammate's pass.
- Ensure proper focus on hand positions distinctly forming a cup shape under the ball.

## Ball
### Description
The volleyball is a spherical object used in play, identifiable by distinct panels often rotating mid-flight.

### Instructions
- Annotate the ball in every frame where it is visually distinct and not immediately obscured by another player.
- Focus on capturing the ball entirely within the bounding box regardless of its position relative to players or the court.