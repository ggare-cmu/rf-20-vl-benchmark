# all-elements-fsod-mebv > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/all-elements-fsod-mebv-6ioka

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Button](#button)
  - [Check Box](#check-box)
  - [Checked Radio Button](#checked-radio-button)
  - [Checked Box](#checked-box)
  - [Dropdown Box](#dropdown-box)
  - [Dropdown Expand](#dropdown-expand)
  - [Icon](#icon)
  - [Radio Button](#radio-button)
  - [Scroll Bar](#scroll-bar)
  - [Text Box](#text-box)

# Introduction

This dataset is focused on annotating common UI elements found in graphical interfaces. Each element is crucial for understanding and interacting with digital content. The task is to accurately identify and annotate these elements to facilitate their automated detection.

- **Button**: A clickable rectangular area often containing text or an icon.
- **Check Box**: A small square used to select or deselect options.
- **Checked Radio Button**: A circular button indicating a selected option.
- **Checked Box**: A check box that has been marked.
- **Dropdown Box**: A horizontal rectangle with an arrow indicating a menu.
- **Dropdown Expand**: The expanded view of a dropdown box showing options.
- **Icon**: A small, symbolic graphic.
- **Radio Button**: A circular button used for exclusive selections.
- **Scroll Bar**: A thin rectangle for scrolling through content.
- **Text Box**: A rectangular field for inputting text.

# Object Classes

## Button
### Description
A button is typically a rectangular area on the screen with text or an icon that can be clicked to perform an action.

### Instructions
- Annotate the entire clickable region of the button, including any border or shadow.
- Do not include adjacent decorative elements such as logos unless they are part of the button.

## Check Box
### Description
A square element used to toggle between two states: checked or unchecked.

### Instructions
- Outline the square region of the check box.
- Do not include the label text next to the check box.

## Checked Radio Button
### Description
A circular button that indicates a selected choice, often filled or marked.

### Instructions
- Encircle the entire radio button, including the outer ring and the inner filled area.
- Exclude any text associated with the button.

## Checked Box
### Description
A marked square indicating selection.

### Instructions
- Annotate only the square containing the check mark.
- Do not include adjacent text or icons.

## Dropdown Box
### Description
An interface element typically shown as a horizontal rectangle with a downward arrow, indicating a menu.

### Instructions
- Outline the rectangle of the dropdown box including the arrow.
- Exclude the text or items visible inside when expanded.

## Dropdown Expand
### Description
The expanded view of a dropdown box, showing all selectable options.

### Instructions
- Annotate the full area occupied by the menu, including visible options.
- Do not include items outside this expanded area.

## Icon
### Description
A small graphic symbol representing an action or object.

### Instructions
- Draw a tight bounding box around the entire icon.
- Do not clip parts of the icon or include surrounding text.

## Radio Button
### Description
A circular UI element used for making a single choice from multiple options.

### Instructions
- Circle the radio button, ensuring the boundary includes the outer circle.
- Do not include any additional elements or text.

## Scroll Bar
### Description
A vertical or horizontal bar used to scroll content, typically located on the edge of a screen or window.

### Instructions
- Annotate the full bar, including the draggable element and background track.
- Do not include adjacent UI components.

## Text Box
### Description
A rectangular input field for entering text.

### Instructions
- Outline the entire text box area, including any border or shadow.
- Exclude placeholder text or icons within or adjacent to the text box.