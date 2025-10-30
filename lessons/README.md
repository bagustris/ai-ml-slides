# Lessons Overview

This directory contains slide presentations for the AI-For-Beginners course.

## Available Lessons

### 1. Introduction to AI
**Location:** `1-Intro/slides.md`

Topics covered:
- What is Artificial Intelligence?
- Weak AI vs. Strong AI
- Defining Intelligence
- The Turing Test
- Brief History of AI
- Modern Deep Learning Revolution

### 2. Symbolic AI
**Location:** `2-Symbolic/slides.md`

Topics covered:
- Knowledge Representation
- Expert Systems and Knowledge-Based Systems
- Semantic Networks
- Ontologies and Knowledge Graphs
- Rule-Based Systems

### 3. Neural Networks
**Location:** `3-NeuralNetworks/slides.md`

Topics covered:
- Introduction to Neural Networks
- Artificial vs. Biological Neurons
- Training Neural Networks
- Overfitting and Regularization
- The Perceptron
- Activation Functions

## Image Assets

- **images/** - Lesson-specific images and diagrams
- **sketchnotes/** - Hand-drawn visual summaries by Tomomi Imura

## Converting Slides to Presentations

These Markdown slides can be converted to various presentation formats:

### Using Marp
```bash
# Install Marp CLI
npm install -g @marp-team/marp-cli

# Convert to PDF
marp lessons/1-Intro/slides.md -o output.pdf

# Convert to HTML
marp lessons/1-Intro/slides.md -o output.html
```

### Using Pandoc with Beamer
```bash
pandoc lessons/1-Intro/slides.md -t beamer -o output.pdf
```

### Using reveal.js
```bash
pandoc lessons/1-Intro/slides.md -t revealjs -s -o output.html
```
