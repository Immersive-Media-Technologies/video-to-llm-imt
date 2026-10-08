# Third-Party Notices

V2L-IMT by Immersive Media Technologies contains no third-party source code. It downloads
models from this project's own release `models-v1` (redistributed unchanged except where noted) and
calls libraries and tools the user installs separately. Their notices are reproduced or listed here.
The IMT Non-Commercial License (see LICENSE) covers V2L-IMT itself, not these components.

---

## Models redistributed in the release `models-v1`

### Whisper — `faster-whisper-small.zip`, `faster-whisper-base.zip`

- Model weights: Whisper by OpenAI. Copyright (c) 2022 OpenAI. MIT License.
- CTranslate2 conversion (the files `model.bin`, `config.json`, `tokenizer.json`, `vocabulary.txt`):
  by SYSTRAN for faster-whisper. Copyright (c) 2023 SYSTRAN. MIT License.
- Redistributed unchanged, one zip per model size.

### PANNs CNN14-DecisionLevelMax — `panns_cnn14_dlmax.onnx` (used only for sounds on request)

- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley — *PANNs: Large-Scale
  Pretrained Audio Neural Networks for Audio Pattern Recognition*, arXiv:1912.10211 (2019).
- Pretrained weights `Cnn14_DecisionLevelMax_mAP=0.385`, source: Zenodo record 3987831.
  License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Changes made by Immersive Media Technologies: converted to ONNX (opset 17) with the log-mel
  spectrogram front end included in the graph; the weights are unchanged (largest difference from the
  original on real audio: 4.5e-7). The 527 class names of Google's AudioSet (dataset licensed CC BY 4.0)
  are stored in the file's metadata.

---

## Libraries and tools installed by the user (not bundled)

| Component | License | Used for |
|---|---|---|
| faster-whisper (SYSTRAN) | MIT | speech recognition; it bundles the Silero VAD model (MIT) |
| CTranslate2 (OpenNMT) | MIT | runs the Whisper model (installed with faster-whisper) |
| ONNX Runtime (Microsoft) | MIT | runs the sound classifier (installed with faster-whisper) |
| NumPy | BSD-3-Clause | audio arrays |
| fpdf2 | LGPL-3.0 | `--format pdf` (optional) |
| Pillow | MIT-CMU (HPND) | `--format sheets` and PDF images (optional, comes with fpdf2) |
| fontTools | MIT | PDF fonts (optional, comes with fpdf2) |
| FFmpeg / ffprobe | LGPL-2.1+ / GPL depending on the build | decoding video and audio |

---

## MIT License (applies to the Whisper weights and their CTranslate2 conversion)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

## CC BY 4.0 (applies to the PANNs weights)

Creative Commons Attribution 4.0 International — legal code: https://creativecommons.org/licenses/by/4.0/legalcode.
In short: you may share and adapt the material for any purpose,
provided you give appropriate credit, indicate the license and indicate if changes were made — which the
section above does.
