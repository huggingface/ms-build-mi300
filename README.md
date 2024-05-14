## Deploying TGI on the VM

Access to the VM as:

```
ssh -L 8081:localhost:80 your_name@azure-mi300-vm
```

From within the VM, please run:

```
docker run --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 256g --net host -v $(pwd)/hf_cache:/data -v $(pwd):/tgi tgi-rocm:latest text-generation-launcher --model-id NousResearch/Meta-Llama-3-8B-Instruct --cuda-graphs 0 --num-shard 1
```

You should see a log as follow:
```
config.json [00:00:00] [████████████████████████████████████████████████████████████████████████████████████████████████████████] 654 B/654 B 2.90 KiB/s (0s)2024-05-07T12:39:07.078362Z  INFO text_generation_launcher: Model supports up to 8192 but tgi will now set its default to 4096 instead. This is to save VRAM by refusing large prompts in order to allow more users on the same hardware. You can increase that size using `--max-batch-prefill-tokens=8242 --max-total-tokens=8192 --max-input-tokens=8191`.
2024-05-07T12:39:07.078381Z  INFO text_generation_launcher: Default `max_input_tokens` to 4095
2024-05-07T12:39:07.078387Z  INFO text_generation_launcher: Default `max_total_tokens` to 4096
2024-05-07T12:39:07.078391Z  INFO text_generation_launcher: Default `max_batch_prefill_tokens` to 4145
2024-05-07T12:39:07.078395Z  INFO text_generation_launcher: Sharding model on 2 processes
2024-05-07T12:39:07.078529Z  INFO download: text_generation_launcher: Starting download process.
2024-05-07T12:39:09.685072Z  INFO text_generation_launcher: Files are already present on the host. Skipping download.

2024-05-07T12:39:10.282605Z  INFO download: text_generation_launcher: Successfully downloaded weights.
2024-05-07T12:39:10.282975Z  INFO shard-manager: text_generation_launcher: Starting shard rank=0
2024-05-07T12:39:10.283533Z  INFO shard-manager: text_generation_launcher: Starting shard rank=1
2024-05-07T12:39:12.772701Z  INFO text_generation_launcher: ROCm: using Flash Attention 2 Composable Kernel implementation.

2024-05-07T12:39:12.773025Z  INFO text_generation_launcher: ROCm: using Flash Attention 2 Composable Kernel implementation.

2024-05-07T12:39:12.966280Z  WARN text_generation_launcher: Could not import Mamba: No module named 'mamba_ssm'

2024-05-07T12:39:12.967123Z  WARN text_generation_launcher: Could not import Mamba: No module named 'mamba_ssm'

2024-05-07T12:39:20.293016Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-05-07T12:39:20.298136Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=1
2024-05-07T12:39:20.565722Z  INFO text_generation_launcher: Server started at unix:///tmp/text-generation-server-0

2024-05-07T12:39:20.593317Z  INFO shard-manager: text_generation_launcher: Shard ready in 10.308899699s rank=0
2024-05-07T12:39:20.645430Z  INFO text_generation_launcher: Server started at unix:///tmp/text-generation-server-1

2024-05-07T12:39:20.698556Z  INFO shard-manager: text_generation_launcher: Shard ready in 10.413843187s rank=1
2024-05-07T12:39:20.796331Z  INFO text_generation_launcher: Starting Webserver
2024-05-07T12:39:20.807178Z  INFO text_generation_router: router/src/main.rs:195: Using the Hugging Face API
...
2024-05-07T12:39:21.272761Z  INFO text_generation_router: router/src/main.rs:289: Using config Some(Llama)
2024-05-07T12:39:21.272766Z  WARN text_generation_router: router/src/main.rs:298: no pipeline tag found for model meta-llama/Meta-Llama-3-8B-Instruct
2024-05-07T12:39:21.278450Z  INFO text_generation_router: router/src/main.rs:317: Warming up model
2024-05-07T12:39:22.121126Z  INFO text_generation_launcher: Cuda Graphs are disabled (CUDA_GRAPHS=None).

2024-05-07T12:39:22.141155Z  INFO text_generation_launcher: Cuda Graphs are disabled (CUDA_GRAPHS=None).

2024-05-07T12:39:22.141566Z  INFO text_generation_router: router/src/main.rs:354: Setting max batch total tokens to 801568
2024-05-07T12:39:22.141574Z  INFO text_generation_router: router/src/main.rs:355: Connected
2024-05-07T12:39:22.141577Z  WARN text_generation_router: router/src/main.rs:369: Invalid hostname, defaulting to 0.0.0.0
```

Locally on the laptop, install jupyter notebook and launch Jupyter Notebook:
```
pip install -U notebook
jupyter-notebook
```

And `local_chatbot.ipynb` can be used to query the model on your desktop through the SSH tunnel!

### Options to try
TGI's `text-generation-launcher` has many options, you can explore `text-generation-launcher --help`

Example of options to try **from the server** (arguments to specifiy to `text-generation-launcher`):
* `--quantize gptq` for [quantized models](https://huggingface.co/docs/text-generation-inference/conceptual/quantization) (beware: needs to use a GPTQ model (e.g. one from https://hf.co/models?search=gptq, for example `TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ`)
* `--dtype bfloat16`
* `--num-shard X`: how many GPUs to use to deploy the model, using [tensor parallelism](https://huggingface.co/docs/text-generation-inference/conceptual/tensor_parallelism)? (here only `--num-shard 1` as one user has only one GPU)
* `--speculate`: use [speculative decoding](https://huggingface.co/docs/text-generation-inference/conceptual/speculation). This argument specifies the number of input_ids to speculate on if using a medusa model, or using n-gram speculation
* `--cuda-graphs 1,2,3,4`: customize the batch sizes to capture cuda graphs for
* `--cuda-memory-fraction`: Limit the CUDA available memory. Useful to host several models on a single GPU (TGI otherwise reserves all the available memory).
* Vision-Language models: https://huggingface.co/docs/text-generation-inference/basic_tutorials/visual_language_models#inference-through-sending-curl-requests

Example of options to try **from the client**:
* [Grammar contrained generation](https://huggingface.co/docs/text-generation-inference/basic_tutorials/using_guidance#guidance): e.g. to contraint the generation to a specific format (JSON). Reference: [Guidance conceptual guide](https://huggingface.co/docs/text-generation-inference/conceptual/guidance).

```
curl localhost:3000/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
    "inputs": "I saw a puppy a cat and a raccoon during my bike ride in the park",
    "parameters": {
        "repetition_penalty": 1.3,
        "grammar": {
            "type": "json",
            "value": {
                "properties": {
                    "location": {
                        "type": "string"
                    },
                    "activity": {
                        "type": "string"
                    },
                    "animals_seen": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5
                    },
                    "animals": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["location", "activity", "animals_seen", "animals"]
            }
        }
    }
}'
```



## Model fine-tuning

Please run:
```
accelerate run test_peft.py
```
