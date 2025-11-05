# paper-parts-fsod-rmrg > 2025-04-03 8:32pm
https://universe.roboflow.com/rf100-vl-fsod/paper-parts-fsod-rmrg-3lbxv

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Author](#author)
  - [Chapter](#chapter)
  - [Equation](#equation)
  - [Equation Number](#equation-number)
  - [Figure](#figure)
  - [Figure Caption](#figure-caption)
  - [Footnote](#footnote)
  - [List of Content Heading](#list-of-content-heading)
  - [List of Content Text](#list-of-content-text)
  - [Page Number](#page-number)
  - [Paragraph](#paragraph)
  - [Reference Text](#reference-text)
  - [Section](#section)
  - [Subsection](#subsection)
  - [Subsubsection](#subsubsection)
  - [Table](#table)
  - [Table Caption](#table-caption)
  - [Table of Contents Text](#table-of-contents-text)
  - [Title](#title)

# Introduction
This dataset is designed to annotate the structural elements of academic papers. It aims to train models to recognize different parts of a paper. Each class corresponds to a text or graphical element commonly found in papers.

- **Author**: The name(s) of the person(s) who wrote the document.
- **Chapter**: The major divisions within the paper, usually denoted by a number and a title.
- **Equation**: Mathematical formulas or expressions.
- **Equation Number**: The numeral identifiers for equations.
- **Figure**: Visual representations like graphs or charts.
- **Figure Caption**: Text descriptions associated with figures.
- **Footnote**: Additional information at the bottom of the page.
- **List of Content Heading**: The titles of content sections in a list.
- **List of Content Text**: Descriptions or details within a list of content.
- **Page Number**: The numeral indicating the page's position.
- **Paragraph**: Blocks of text conveying an idea or point.
- **Reference Text**: Citations or bibliographic information.
- **Section**: Main headings within a chapter.
- **Subsection**: Subheadings under a section.
- **Subsubsection**: Further subdivisions under a subsection.
- **Table**: Data or information arranged in rows and columns.
- **Table Caption**: Text descriptions associated with tables.
- **Table of Contents Text**: Entries listing sections and page numbers.
- **Title**: The main heading or name of the paper.

# Object Classes

## Author
### Description
Text indicating the name(s) of the author(s), typically found near the beginning of a document.
### Instructions
Identify the text block containing the author names. It usually follows the title and may include affiliations. Do not include titles, affiliations or titles of sections adjacent to author names.

## Chapter
### Description
Indicates a major division of the document, often labeled with a number and title.
### Instructions
Locate text labeled with "Chapter" followed by a number and title. Capture the entire heading, ensuring no unrelated text is included.

## Equation
### Description
Symbols and numbers arranged to represent a mathematical concept.
### Instructions
Draw boxes around all mathematical expressions, excluding any accompanying text or numbers identifying the equations.

## Equation Number
### Description
Numerals used to uniquely identify equations.
### Instructions
Identify numbers in parentheses next to equations. Do not include equation text or variables.

## Figure
### Description
Visual content such as graphs, diagrams, code or images.
### Instructions
Outline the entire graphical representation. Do not include captions or any surrounding text.

## Figure Caption
### Description
Text providing a description or explanation above or below a figure.
### Instructions
Identify the text directly associated with a figure. Ensure no unrelated figures or text are included.

## Footnote
### Description
Clarifications or additional details located at the bottom of a page.
### Instructions
Locate text at the page's bottom that refers back to a mark or reference in the main text. Exclude any unrelated content.

## List of Content Heading
### Description
Headings at the list of context text, identifying its purpose or content. This may also be called a list of figures. 
### Instructions
Identify and label only the heading for lists in content sections. Do not include subsequent list items.

## List of Content Text
### Description
The detailed entries or points in a list. These often summarize all figures in the paper.
### Instructions
Identify each item in a content list. Exclude list headings and any non-list content.

## Page Number
### Description
Numerical indication of the current page.
### Instructions
Locate numbers typically positioned at the top or bottom margins. Do not include text or symbols beside the numbers.

## Paragraph
### Description
Blocks of text separated by spacing or indentation.
### Instructions
Enclose individual text blocks that form coherent sections. Ensure each paragraph is distinguished separately.

## Reference Text
### Description
Bibliographic information found typically in a reference section.
### Instructions
Identify the full reference entries. Do not draw boxes around individual citations.

## Section
### Description
Major subheadings documenting specific topics. This is typically represented using a single number (e.g. Section 3).
### Instructions
Outline the whole heading text. Ensure no text from following lines is included unless part of the heading.

## Subsection
### Description
Secondary subheadings under sections. This is typically represented using a number followed by a decimal point, followed by another number (e.g. Section 3.2).
### Instructions
Identify and box all subsection headings just beneath section headings. Ensure complete titles are included.

## Subsubsection
### Description
Specific subdivisions within subsections. This is typically represented using a number followed by a decimal point, followed by a number, followed by a decimal point, followed by another number (e.g. Section 3.2.4).
### Instructions
Identify the entire subsubsection heading text. Confirm no unrelated text is mixed with the heading.

## Table
### Description
Information organized in rows and columns.
### Instructions
Enclose the entire grid, including borders. Do not include titles or captions outside the table.

## Table Caption
### Description
Descriptive text above or below a table explaining its contents.
### Instructions
Identify and outline the text directly related to a table. Ensure it is not mixed with unrelated text or figures.

## Table of Contents Text
### Description
Entries in the table of contents listing sections and their page numbers.
### Instructions
Locate and box each entry without including any headings or unrelated sections from other pages.

## Title
### Description
The headline identifying the document's main subject.
### Instructions
Outline the primary title text of the paper, usually found at the beginning. Ensure no subsequent author or affiliation details are included.