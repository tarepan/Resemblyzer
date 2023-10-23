Clone of official `resemblyzer`.  
We hack package install-related part of official repository. All functionality is kept as is.  
For usage/demo/info, please check official repository.  

Resemblyzer allows you to derive a **high-level representation of a voice** through a deep learning model (referred to as the voice encoder). Given an audio file of speech, it creates a summary vector of 256 values (an embedding, often shortened to "embed" in this repo) that summarizes the characteristics of the voice spoken. 

## Code example
```python
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

fpath = Path("<path/to/audio>")
wav = preprocess_wav(fpath)

encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
np.set_printoptions(precision=3, suppress=True)
print(embed)
```

## Semantic diff
We remove `webrtcvad>=2.0.10` & `torch>=1.0.1` from `requirements_package.txt`.  
You should install them manually.  
