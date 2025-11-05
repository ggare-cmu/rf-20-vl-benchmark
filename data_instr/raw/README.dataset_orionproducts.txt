# orionproducts-vtl2z-fsod-puhv > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/orionproducts-vtl2z-fsod-puhv-ce2it

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Candy Boom](#candy-boom)
  - [Chocopie Dark](#chocopie-dark)
  - [Chocopie Nor](#chocopie-nor)
  - [Marine Boy](#marine-boy)
  - [O'Star Red](#ostar-red)
  - [O'Star Yellow](#ostar-yellow)
  - [Swing Maxx](#swing-maxx)
  - [Swing Nor](#swing-nor)

# Introduction
The dataset is designed to recognize and annotate different snack packaging on supermarket shelves. It includes classes such as candy, chocopie, and various chips. Each class is described by distinctive packaging features including text, imagery, and package shape.

- **Candy Boom**: Snack with Candy Boom logo
- **Chocopie Dark**: Snack with Chocopie Dark logo
- **Chocopie Nor**: Snack with Chocopie Nor logo
- **Marine Boy**: Snack with Marine Boy logo
- **O'Star Red**: Snack with O'Star Red logo
- **O'Star Yellow**: Snack with O'Star Yellow logo
- **Swing Maxx**: Snack with Swing Maxx logo
- **Swing Nor**: Snack with Swing Nor logo

# Object Classes

## Candy Boom
### Description
Candy Boom packages have bold comic-style text “BOOM” against a vibrant background. The design often features a dynamic graphical explosion theme.

### Instructions
- Annotate the entire visible package surface that contains the “BOOM” text and dynamic graphical elements. Ensure the annotation encompasses all visible portions of the branding. 
- Do not label if the text or graphic is unrecognizable.

## Chocopie Dark
### Description
Chocopie Dark packaging features a rich brown color with prominent “Chocopie Dark” text. Often includes an image of a chocolate pie with dark filling.

### Instructions
- Draw bounding boxes around packages with visible “Chocopie Dark” branding. Ensure the label covers the entire visible package facing the observer. 
- Exclude any packaging where the text and image are not clear.

## Chocopie Nor
### Description
Chocopie Nor has a distinctive red packaging, prominently displaying “Chocopie” and typically featuring a snack image.

### Instructions
- Place annotations over any visible packaging with the red “Chocopie” label. Ensure full coverage of the package face displaying the brand. 
- Omit if the brand name is obscured or not immediately identifiable.

## Marine Boy
### Description
Marine Boy packaging typically has a playful ocean theme, featuring nautical characters and bold marine graphics on the package.

### Instructions
- Capture the full area of the packaging that is covered in marine-themed imagery and branding. Include all visible text and graphics tribal to Marine Boy.
-  Do not annotate if the theme is unclear or obscured.

## O'Star Red
### Description
O’Star Red packages are identified by their red packaging with bold “O’Star” branding. The pack often features imagery of chips.

### Instructions
- Annotate the entire surface showing the “O’Star” brand in red. Ensure the bounding box includes the logo and chip imagery. 
- Avoid labeling obscured or unclear visuals of the package.

## O'Star Yellow
### Description
O’Star Yellow can be recognized by its yellow package with similar branding style to O’Star Red, but in a distinct yellow color scheme.

### Instructions
- Encapsulate all areas of packaging with visible yellow “O’Star” branding. Include the primary face with visible text and graphics. 
- Exclude sections where brand elements are obstructed or imperceptible.

## Swing Maxx
### Description
Swing Maxx packaging is distinct with bold graphical elements and the “Swing Maxx” text, often in a striking layout.

### Instructions
- Highlight the areas of the packaging showing the “Swing Maxx” label and graphics. 
- The box should cover only well-defined and visible parts of the brand name and related elements.

## Swing Nor
### Description
Swing Nor packaging is similar in branding style to Swing Maxx but features variations in the graphic design and text placement.

### Instructions
- Draw bounding areas around any clear and visible representation of "Swing Nor" branding. Focus on the parts featuring text and related graphics. 
- Skip labeling if the logo or graphics are blocked or indistinct.