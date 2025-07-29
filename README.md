# Fall Detection System – fall1_intergrate

This project integrates human detection, pose tracking, fall detection, and MQTT-based alerting.

## Structure

- `main.py` – Main entry
- `detection/` – Human/pose detection
- `fall/` – Fall logic
- `comm/` – Communication (MQTT, emergency triggers)
- `config/` – Central config
- `utils/` – Visualization
- `models/` – YOLO models
- `tests/` – Unit testing

## Setup

```bash
pip install -r requirements.txt
python main.py
```
