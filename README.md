# studychat-automatic-labeling

This repository contains the code and data used in the paper:
"When Humans and LLMs Disagree: Understanding Human-LLM Disagreement in Learner Intent Annotation for Educational Chatbot Dialogues"

## How to use

#### 0. Requirements

```bash
python3 -m venv venv
source venv/bin/activate # venv\Scripts\Activate.ps1 for Windows
pip install -r requirements.txt
```

#### 1. Setup

```bash
python -m automatic_labeling.build_index
```

#### 2. Run automatic annotation
```bash
python -m automatic_labeling.run

# OR if you have an ID file:

python -m automatic_labeling.run --use-id-file
```

## Dataset format

The code expects a JSON (or JSONL) file with at least the following structure:
```json
{
  "e16b9530-e031-70ad-d225-033f2f45a27d": {
    "be5e27c0-caf2-4ec9-98e9-19a7923790d4": [
      {
        "userId": "e16b9530-e031-70ad-d225-033f2f45a27d",
        "chatId": "be5e27c0-caf2-4ec9-98e9-19a7923790d4",
        "turnId": 0,
        "prompt": "...",
        "response": "...",
        "topic": "a4",
      },
      ...
    ],
    ...
  },
  ...
}
```

## Configuration notes
You may want to configure several parts of the code. Look for TODO markers in the codebase.

## Citations

Coming soon.