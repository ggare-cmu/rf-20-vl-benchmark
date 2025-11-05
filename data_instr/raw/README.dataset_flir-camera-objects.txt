# flir-camera-objects-fsod-tdqp > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/flir-camera-objects-fsod-tdqp-xpjid

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Bicycle](#bicycle)
  - [Car](#car)
  - [Dog](#dog)
  - [Person](#person)

# Introduction
This dataset is designed for object detection using images captured by FLIR cameras. It includes four classes: bicycle, car, dog, and person.

- **Bicycle:** Two-wheeled, pedal-driven vehicles.
- **Car:** Motor vehicles with four wheels, used for transportation.
- **Dog:** Domesticated animals, often four-legged and furry.
- **Person:** Human beings, typically upright in posture.

# Object Classes

## Bicycle

### Description
Bicycles are two-wheeled vehicles that can be identified by their circular wheels and frame structure. They often feature handlebars and pedals.

### Instructions
- Annotate the bicycle frame and wheels completely. Ensure both wheels are included, even if partially visible.
- Do not label accessories like water bottles unless they are part of the main structure.
- If a person is riding the bicycle, the person's body should be annotated separately.

## Car

### Description
Cars are four-wheeled motor vehicles. They are identifiable by their larger structure compared to bicycles, with a distinct hood and trunk area.

### Instructions
- Include all visible parts of the car, like the body and wheels.
- Do not separate individual parts of the car unless they are distinct objects, e.g., a detached roof rack.
- Exclude reflections of the car if visible in nearby surfaces.

## Dog

### Description
Dogs are four-legged animals often seen on roadsides or sidewalks. They are typically identified by their fur and tail.

### Instructions
- Annotate the entire body of the dog, ensuring legs, head, and tail are included.
- Do not label dogs if only a small portion is visible, making identification uncertain.
- Avoid annotating any accessories unless they are very evident (e.g., a leash attached to the dog).

## Person

### Description
People are identified by their upright posture and features like arms and legs. Typically captured engaging in various activities.

### Instructions
- Include the whole person, capturing arms, legs, and head.
- Persons in motion, like walking or running, should still be annotated if they are easily distinguishable.
- If a person is partially hidden behind objects, only annotate visible parts that make identification possible.