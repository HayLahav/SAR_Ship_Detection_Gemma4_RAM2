# SAR Ship Detection with Gemma 4 E2B + RAM 2

Agentic maritime surveillance using Sentinel-1 SAR satellite imagery, Blur Labs' RAM 2 for despeckling and super-resolution, and Google's Gemma 4 E2B for vision-based ship classification and tool-using analysis.

## Overview

Production-ready pipeline for detecting and classifying ships from Synthetic Aperture Radar (SAR) data. Combines signal processing with lightweight AI for efficient maritime monitoring on standard hardware.

## Key Features

- Direct integration with Google Earth Engine for Sentinel-1 SAR data
- SAR despeckling + 2× super-resolution with [RAM 2](https://www.blurlabs.ai/ram2/) (Blur Labs)
- Automated ship detection using threshold-based bright-spot detection
- Vision-based classification with Gemma 4 E2B: each ship's image chip + measurements → type, confidence, short description
- **Agent mode:** ask in plain English; Gemma 4 plans and calls the analysis tools itself (native function calling)
- Runs on a Google Colab T4 GPU (4-bit option for tighter memory)
- CSV export for GIS integration
- Detailed ship metrics: position (lat/lon), size, type, confidence, description

## System Architecture

```
Sentinel-1 SAR → RAM 2 (despeckle + 2× SR) → Ship Detection → Image chip + measurements → Gemma 4 E2B → Type + description

Agent:  user request → Gemma 4 E2B ──calls──▶ lookup_location / analyze_area / list_ships / describe_ship / export_results
                              ◀──results──┘  … repeats until it can answer
```

## Detected Ship Types

- Small Vessel (fishing boats, yachts, patrol boats)
- Fishing Vessel (trawlers, seiners)
- Medium Cargo (feeder ships, coastal cargo)
- Container Ship (large cargo vessels)
- Bulk Carrier (grain, coal carriers)
- Tanker (oil, chemical tankers)
- Large Ship (VLCC, large carriers)
- Offshore Platform (stationary structures)

## Installation

```bash
pip install earthengine-api opencv-python-headless scikit-image "transformers>=5.10.1" accelerate bitsandbytes
pip install "blurlabs @ git+https://github.com/BlurLabsAI/blurlabs.git"   # RAM 2
```

Gemma 4 and RAM 2 weights are gated on Hugging Face. Request access at
[google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) and
[blurlabs/ram2](https://huggingface.co/blurlabs/ram2), then set an `HF_TOKEN` secret in Colab.

## Quick Start

```python
# Initialize Google Earth Engine
PROJECT_ID = 'your-gee-project-id'
init_gee()

# Run the notebook cells that load Gemma 4 E2B and RAM 2, then:

# Analyze maritime area
results = analyze_with_gemma(
    lon=103.85,  # Singapore Strait
    lat=1.28,
    size_km=5,
    use_ram2=True,   # RAM 2 despeckling (falls back to raw image if unavailable)
    sr_factor=2,     # 1 = despeckle only, 2 = also 2x super-resolution
    enl=4.4          # speckle looks of Sentinel-1 IW GRD
)

# Export results
export_csv(results, 'singapore')

# Or let the agent plan the analysis
agent.run("Check the Strait of Hormuz, list the three largest ships, "
          "describe the largest one and export the results.")
```

## Technical Specifications

**Image Processing:**
- Resolution: 512x512 pixels
- Pixel size: ~10 meters/pixel
- Coverage: 5x5 km default
- Preprocessing: Gaussian blur (7x7 kernel), or RAM 2 when enabled

**Enhancement (RAM 2):**
- Model: Blur Labs RAM 2 (Reconstruct Anything Model 2), 359M params, physics-conditioned restoration
- Despeckling: speckle modelled as multiplicative Gamma noise (`deepinv.physics.GammaNoise(l=ENL)`)
- Super-resolution: bicubic `Downsampling(factor=2)` operator, ~10 m → ~5 m/pixel (1024x1024 output)
- Tiled inference (256 px tiles, 32 px overlap) to fit the Colab free tier
- Detector size limits and pixel size in meters are rescaled automatically

**Detection:**
- Method: Percentile-based thresholding (99.5th percentile)
- Morphological filtering for noise removal
- Size filtering: 10-300 pixels (100-30,000 m²)

**Classification:**
- Model: Gemma 4 E2B instruction-tuned (`google/gemma-4-E2B-it`)
- Input: 192×192 SAR image chip around the ship + length, width, aspect ratio, brightness, solidity
- Output: JSON with type, confidence and a one-sentence description
- Confidence: the model's own estimate, not a calibrated probability
- Fallback: size rules if Gemma is unavailable or its answer can't be parsed

**Agent:**
- Gemma 4 native function calling; tool schemas generated from Python type hints and docstrings
- Tools: `lookup_location`, `analyze_area`, `list_ships`, `describe_ship`, `export_results`
- Tool results are returned in Gemma 4's `tool_responses` format; stops after `max_steps` rounds

**Performance:**
- Memory: Gemma 4 E2B ~10GB in bfloat16 (less with `LOAD_IN_4BIT = True`); RAM 2 adds ~1.5GB
- RAM 2: a few seconds on GPU, ~1 minute on CPU per 512x512 scene (2x SR)
- GPU: recommended (Colab T4). CPU works but is slow

## Example Results

**Singapore Strait (103.85°E, 1.28°N), earlier Gemma 3 270M version without RAM 2:**
- Total ships: 17
- Medium Cargo: 16
- Bulk Carrier: 1
- Processing time: ~20 seconds

*Results for the Gemma 4 + RAM 2 pipeline have not been recorded yet.*

**Rotterdam Port (4.0°E, 51.95°N):**
- Total ships: Varies by date
- Mix of container ships, bulk carriers, tankers

## Applications

**Maritime Security:**
- Unauthorized vessel detection
- Restricted zone monitoring
- Dark vessel identification

**Port Operations:**
- Traffic density analysis
- Congestion monitoring
- Vessel classification statistics

**Environmental:**
- Oil spill source identification
- Offshore platform monitoring
- Maritime traffic patterns

**Research:**
- Ship detection algorithm validation
- Maritime activity trends
- Coastal surveillance

## Validation

Compare results with AIS (Automatic Identification System) ground truth:

**Free AIS Sources:**
- MarineTraffic (marinetraffic.com)
- VesselFinder (vesselfinder.com)
- AISHub (aishub.net)

**Academic Datasets:**
- NASA ShipDetectionNet
- Sentinel-1 Ship Detection Dataset (xView3)

## File Structure

```
SAR_Ship_Detection_Gemma4_RAM2.ipynb  # Main notebook
├── Authentication (Google Earth Engine)
├── Model Loading (Gemma 4 E2B, RAM 2)
├── SAR Data Loader (512x512)
├── RAM 2 Enhancement (despeckling + super-resolution)
├── Ship Detector (threshold-based)
├── Gemma 4 Classifier (image chip + measurements)
├── Visualization (3-panel display)
├── Export Functions (CSV)
└── Agent Mode (Gemma 4 function calling)
```

## Limitations

- Resolution: 512x512 limits detection of vessels 
- Small ships may be missed due to threshold settings
- Weather conditions affect SAR quality
- RAM 2 is a general-purpose restoration model, not trained specifically on SAR. Check the before/after
  comparison and tune `enl` (lower = stronger despeckling). It works on the 8-bit, percentile-stretched
  thumbnail, so very bright targets are clipped before enhancement.
- RAM 2 is licensed GPL-3.0-only (a commercial license is available from Blur Labs)
- Gemma 4 is not trained on SAR ship classification; treat types and descriptions as estimates and validate with AIS
- The agent can pick wrong tools or arguments; check the printed tool steps

## Future Improvements

- Higher resolution option (1024x1024)
- CFAR (Constant False Alarm Rate) detector
- Rotated bounding boxes for orientation
- Temporal tracking across multiple acquisitions
- Fine-tune Gemma 4 on labelled SAR ship chips (e.g. xView3)
- AIS integration for automatic validation

## Model Details

**Gemma 4 E2B (`google/gemma-4-E2B-it`):**
- Parameters: 2.3B effective (5.1B including embeddings)
- Inputs: text, images, audio
- Context window: 128K tokens
- Native function calling
- Quantization: 4-bit option available

**Why Gemma 4 E2B instead of Gemma 3 270M:**
- Gemma 3 270M is text-only, so it could only classify from five numbers, which a size rule does equally well
- Gemma 4 E2B sees the ship's image chip and can drive tools as an agent
- Cost: ~10× more parameters, so a GPU is recommended

**RAM 2 (`blurlabs/ram2`):**
- Reconstruct Anything Model 2, 359M parameters, physics-conditioned image restoration
- License: GPL-3.0-only (commercial license available from Blur Labs)

## Data Sources

**Sentinel-1:**
- Provider: European Space Agency (ESA) / Copernicus
- Instrument: C-band Synthetic Aperture Radar
- Mode: IW (Interferometric Wide Swath)
- Polarization: VV
- Revisit time: 6-12 days
- Resolution: 10m (ground range)

## Requirements

- Python 3.8+
- Google Earth Engine account (free)
- GPU with ~12GB+ memory recommended (Colab T4), or 4-bit loading
- Hugging Face account with access to the gated Gemma 4 and RAM 2 weights
- Internet connection for Sentinel-1 download

## License

MIT License - Free for non profit use (check Google's License)

## Citation

```
SAR Ship Detection with Gemma 4 E2B + RAM 2
Uses Sentinel-1 data from ESA/Copernicus
Uses Gemma 4 E2B from Google DeepMind
Uses RAM 2 from Blur Labs
```

## Contributing

Pull requests welcome for:
- Detection algorithm improvements
- Additional ship type categories
- AIS integration
- Performance optimizations
- Validation metrics

## Acknowledgments

- ESA/Copernicus for Sentinel-1 data
- Google DeepMind for Gemma 4
- Blur Labs for RAM 2
- Google Earth Engine for data infrastructure
