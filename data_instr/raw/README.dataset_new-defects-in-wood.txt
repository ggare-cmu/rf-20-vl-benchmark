# new-defects-in-wood-uewd1-fsod-tffp > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/new-defects-in-wood-uewd1-fsod-tffp-x9ygx

Provided by a Roboflow user
License: MIT

# Introduction
The purpose of this dataset is to help localize and classify different defects that are often found on wood planks used in manufacturing, or other wood products. Wood will often have knots, cracks, or holes in it, all of which inform the quality and condition of the wood. 

- **Crack**: A crack in the wood.
- **Holes**: A hole in the wood.
- **Knot with Crack**: A knot with a crack in it.
- **Dead Knot**: A darker colored knot.
- **Live Knot**: A lighter colored knot.

# Object Classes
## Crack
### Description
This class contains cracks in the wood. It specifically does not contain cracks that occur in knots.

### Instructions
- Draw a bounding box around the entire crack. If a crack has several clearly distinct sections to it, draw boxes around those individually. Try to make the box as tight around the crack as possible. 
- Cracks will often appear as dark lines in the wood that do not necessarily follow the wood grain. Do not tag cracks that appear in knots, as those belong to the knot with crack category. 
- If a crack extends a long way out of a knot, more than the diameter of the knot, draw a box around the part of the crack not in the knot.

## Holes
### Description
This class contains holes in wood. The two major kinds of holes are: the hole a termite or nail may leave behind, small and concentrated; or a hole that has the same dark color as a crack but just doesn't go anywhere, instead having a circular shape. These holes can be very small, so look carefully!

### Instructions
- Draw a tight bounding box around the hole. If there are clusters of holes, mark each individually. 
- Make sure not to tag holes that appear in knots.

## Knot with Crack
### Description
This is a knot that has a crack in it. It doesn't matter if the knot is dead or alive; if it has a crack in it, it's a knot with crack.

### Instructions
- Draw a tight bounding box around the knot that has the crack in it. Most of the time, the crack will be contained entirely within the knot. 
- However, if the crack extends outside of the knot, do not enlarge the bounding box to contain the rest of the crack. The important thing is that the box is tight around the knot.

## Dead Knot
### Description
This knot belonged to a branch that died before the tree was cut down. Dead knots typically have strong discoloration around at least some part of their boundary, indicating that its growth with the tree was poor. Often this discoloration is in the form of a very dark ring around the knot. Other times, there is just a sudden and sharp contrast with the color of the rest of the wood, that isn't necessarily shaped as a ring. The discoloration is sharp and sudden. It can look like the boundary of rot.

### Instructions
- Draw a tight bounding box around the knot. If there is clear discoloration, ensure that it is contained within the bounding box. 
- For example, if the knot has a dark ring, ensure that the ring is entirely contained within the box.

## Live Knot
### Description
This a knot that belonged to a living branch. These knots may be a different color from the rest of the wood, but the boundary between them and the wood is typically softer than it is for a dead knot. The knot looks like it is part of the wood, solidly attached all the way around. If the wood has grain around the knot, the grain moves smoothly around the knot. These knots can have rings, but their boundaries are soft.

### Instructions
- Draw a bounding box around the knot. Some knots are clear as to where they end and the rest of the wood begins. For those knots, draw a tight bounding box. 
- Other knots blend into the general grain of the wood. They will, however, always have a circular region where the branch grew. For those kinds of knots, do your best to draw the bounding box around the circular region and not around the rest of the warp of the grain.