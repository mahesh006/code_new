# Radiology Report Generation

This repository implements a two-stage radiology report generation system using EfficientNet-B4 as the image encoder and GPT-2 for text generation.

## Directory Structure

- `models/encoder.py`: Defines the image encoder using EfficientNet-B4.
- `models/text_generator.py`: Implements the GPT-2-based text generation models.
- `main.py`: Main script to run the radiology report generation pipeline.
- `requirements.txt`: List of required dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/radiology-report-generation.git
   cd radiology-report-generation
   ```
