# Table of content

[1. Deploying TGI on the VM](#deploying-tgi-on-the-vm)
	[1. Options to try](#options-to-try)
        [1. Quantization](#quantization)
        [2. Tensor parallelism](#tensor-parallelism)
        [3. Speculative decoding](#speculative-decoding)
        [4. Customize HIP Graph, TunableOp warmup](#customize-hip-graph-tunableop-warmup)
        [5. Deploy several models on a single GPU](#deploy-several-models-on-a-single-gpu)
        [6. Grammar contrained generation](#grammar-contrained-generation)
        [7. Benchmarking](#benchmarking)
        [8. Vision-Language models (VLM)](#vision-language-models-vlm)
[2. Model fine-tuning](#model-fine-tuning-with-transformers-and-peft)

# Deploying TGI on the VM

Access to the VM as:

```
ssh -L 8081:localhost:80 your_name@azure-mi300-vm
```

From within the VM, please run:

```
docker run --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 256g \
    --net host -v $(pwd)/hf_cache:/data \
    -e HUGGING_FACE_HUB_TOKEN=$HF_READ_TOKEN \
    ghcr.io/huggingface/text-generation-inference:sha-293b8125-rocm \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --num-shard 1 --port 8081
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

# Options to try

TGI's `text-generation-launcher` has many options, you can explore `text-generation-launcher --help`.

TGI's documentation can also be used as a reference: https://huggingface.co/docs/text-generation-inference

For the workshop, a few models have already been cached on the machines, and we recommend to use them:
* `meta-llama/Meta-Llama-3-8B-Instruct`
* `meta-llama/Meta-Llama-3-70B-Instruct`
* `TheBloke/Llama-2-70B-Chat-GPTQ` (GPTQ model)
* `casperhansen/llama-3-70b-instruct-awq` (AWQ model)
* `mistralai/Mistral-7B-Instruct-v0.2`
* `bigcode/starcoder2-15b-instruct-v0.1`
* `text-generation-inference/Mistral-7B-Instruct-v0.2-medusa` (with Medusa speculative decoding)

## Quantization

TGI can be used with quantized models (GPTQ, AWQ) with the option `--quantize gptq` for [quantized models](https://huggingface.co/docs/text-generation-inference/conceptual/quantization) (beware: needs to use a GPTQ model (e.g. one from https://hf.co/models?search=gptq, for example `TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ`)

Example, adding to the docker command:
```
--model-id TheBloke/Llama-2-70B-Chat-GPTQ --quantize gptq
```
or
```
--model-id casperhansen/llama-3-70b-instruct-awq --quantize awq
```

## Tensor parallelism

Here we use only `--num-shard 1`, as only one GPU is available per person. But on a full node, one may use `--num-shard X` to decide how many GPUs are best to deploy a model given for example latency constraints.

Read more about tensor parallelism in TGI: https://huggingface.co/docs/text-generation-inference/conceptual/tensor_parallelism

## Speculative decoding

TGI supports **n-gram** specumation, as well as [**Medusa**](https://arxiv.org/pdf/2401.10774) speculative decoding.

In the launcher, the argument `--speculate X` allows to use speculative decoding. This argument specifies the number of input_ids to speculate on if using a medusa model, or using n-gram speculation.

Example, adding to the docker command:
```
--model-id mistralai/Mistral-7B-Instruct-v0.2 --speculate 3
```
or with Medusa:

```
--model-id text-generation-inference/Mistral-7B-Instruct-v0.2-medusa --speculate 3
```
(see its config: https://huggingface.co/text-generation-inference/Mistral-7B-Instruct-v0.2-medusa/blob/main/config.json)

Read more at: https://huggingface.co/docs/text-generation-inference/conceptual/speculation

Medusa implementation: https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/layers/medusa.py

## Customize HIP Graph, TunableOp warmup

[HIP Graphs](https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.1/how-to/programming_manual.html#hip-graph) and [TunableOp](https://huggingface.co/docs/text-generation-inference/installation_amd#tunableop) are used in the warmup step to statically capture compute graphs in the decoding, and to select the best performing GEMM available implementation (from rocBLAS, hipBLASlt).

The sequence lengths for which HIP Graphs are captured can be specified with e.g. `--cuda-graphs 1,2,4,8`. `--cuda-graphs 0` can be used to disable HIP Graphs.

If necessary, TunableOp can be disabled with by passing `--env PYTORCH_TUNABLEOP_ENABLED="0"` when launcher TGI’s docker container.

## Deploy several models on a single GPU

Several models can be deployed on a single GPU. By default TGI reserves all the free GPU memory to pre-allocate the KV cache.

One can use the option `--cuda-memory-fraction` to limit the CUDA available memory used by TGI. Example: `--cuda-memory-fraction 0.5`

This is useful to deploy several different models on a single GPU.

## Vision-Language models (VLM)

Refer to: https://huggingface.co/docs/text-generation-inference/basic_tutorials/visual_language_models

## Grammar contrained generation

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

## Benchmarking

Text Generation Inference comes with its own benchmarking tool, `text-generation-benchmark`.

Usage: `text-generation-benchmark --help`

Example:
1. Launch a container with `--model-id meta-llama/Meta-Llama-3-8B-Instruct`
2. Open an other terminal in the container (`docker container ls` and then `docker exec -it container_name /bin/bash`
3. Then, run for example:
```
text-generation-benchmark --tokenizer-name meta-llama/Meta-Llama-3-8B-Instruct --sequence-length 2048 --decode-length 128 --warmups 2 --runs 10 -b 1 -b 2 -b 4 -b 8 -b 16 -b 32 -b 64
```

`text-generation-benchmark` can give results tables as:

| Parameter          | Value                                |
|--------------------|--------------------------------------|
| Model              | meta-llama/Meta-Llama-3-70B-Instruct |
| Sequence Length    | 2048                                 |
| Decode Length      | 128                                  |
| Top N Tokens       | None                                 |
| N Runs             | 10                                   |
| Warmups            | 2                                    |
| Temperature        | None                                 |
| Top K              | None                                 |
| Top P              | None                                 |
| Typical P          | None                                 |
| Repetition Penalty | None                                 |
| Frequency Penalty  | None                                 |
| Watermark          | false                                |
| Do Sample          | false                                |


| Step           | Batch Size | Average    | Lowest     | Highest    | p50        | p90        | p99        |
|----------------|------------|------------|------------|------------|------------|------------|------------|
| Prefill        | 1          | 345.72 ms  | 342.55 ms  | 348.42 ms  | 345.88 ms  | 348.42 ms  | 348.42 ms  |
|                | 2          | 455.36 ms  | 452.29 ms  | 458.80 ms  | 454.97 ms  | 458.80 ms  | 458.80 ms  |
|                | 4          | 673.80 ms  | 666.73 ms  | 678.06 ms  | 675.55 ms  | 678.06 ms  | 678.06 ms  |
|                | 8          | 1179.98 ms | 1176.53 ms | 1185.13 ms | 1180.36 ms | 1185.13 ms | 1185.13 ms |
|                | 16         | 2046.73 ms | 2036.32 ms | 2061.69 ms | 2045.36 ms | 2061.69 ms | 2061.69 ms |
|                | 32         | 4313.01 ms | 4273.01 ms | 4603.97 ms | 4282.30 ms | 4603.97 ms | 4603.97 ms |
| Decode (token) | 1          | 12.38 ms   | 12.02 ms   | 15.06 ms   | 12.08 ms   | 12.12 ms   | 12.12 ms   |
|                | 2          | 16.75 ms   | 16.02 ms   | 19.79 ms   | 16.06 ms   | 16.11 ms   | 16.11 ms   |
|                | 4          | 17.57 ms   | 16.28 ms   | 19.94 ms   | 18.84 ms   | 16.34 ms   | 16.34 ms   |
|                | 8          | 18.63 ms   | 16.75 ms   | 22.28 ms   | 19.55 ms   | 16.87 ms   | 16.87 ms   |
|                | 16         | 21.83 ms   | 18.94 ms   | 25.53 ms   | 21.99 ms   | 21.98 ms   | 21.98 ms   |
|                | 32         | 27.76 ms   | 24.49 ms   | 33.47 ms   | 27.84 ms   | 29.67 ms   | 29.67 ms   |
| Decode (total) | 1          | 1571.76 ms | 1526.99 ms | 1912.55 ms | 1534.09 ms | 1538.85 ms | 1538.85 ms |
|                | 2          | 2127.04 ms | 2034.91 ms | 2513.82 ms | 2039.47 ms | 2046.08 ms | 2046.08 ms |
|                | 4          | 2231.84 ms | 2067.17 ms | 2532.21 ms | 2393.08 ms | 2074.70 ms | 2074.70 ms |
|                | 8          | 2366.38 ms | 2127.92 ms | 2829.20 ms | 2483.24 ms | 2142.88 ms | 2142.88 ms |
|                | 16         | 2772.09 ms | 2405.33 ms | 3242.91 ms | 2792.36 ms | 2791.81 ms | 2791.81 ms |
|                | 32         | 3525.13 ms | 3110.67 ms | 4251.15 ms | 3535.48 ms | 3767.61 ms | 3767.61 ms |


| Step    | Batch Size | Average             | Lowest             | Highest             |
|---------|------------|---------------------|--------------------|---------------------|
| Prefill | 1          | 2.89 tokens/secs    | 2.87 tokens/secs   | 2.92 tokens/secs    |
|         | 2          | 4.39 tokens/secs    | 4.36 tokens/secs   | 4.42 tokens/secs    |
|         | 4          | 5.94 tokens/secs    | 5.90 tokens/secs   | 6.00 tokens/secs    |
|         | 8          | 6.78 tokens/secs    | 6.75 tokens/secs   | 6.80 tokens/secs    |
|         | 16         | 7.82 tokens/secs    | 7.76 tokens/secs   | 7.86 tokens/secs    |
|         | 32         | 7.42 tokens/secs    | 6.95 tokens/secs   | 7.49 tokens/secs    |
| Decode  | 1          | 81.16 tokens/secs   | 66.40 tokens/secs  | 83.17 tokens/secs   |
|         | 2          | 120.14 tokens/secs  | 101.04 tokens/secs | 124.82 tokens/secs  |
|         | 4          | 229.31 tokens/secs  | 200.62 tokens/secs | 245.75 tokens/secs  |
|         | 8          | 433.91 tokens/secs  | 359.11 tokens/secs | 477.46 tokens/secs  |
|         | 16         | 743.16 tokens/secs  | 626.60 tokens/secs | 844.79 tokens/secs  |
|         | 32         | 1164.14 tokens/secs | 955.98 tokens/secs | 1306.47 tokens/secs |

## Vision-Language models (VLM)

Refer to: https://huggingface.co/docs/text-generation-inference/basic_tutorials/visual_language_models


# Model fine-tuning with Transformers and PEFT

Run TGI's container in interactive model, adding the following to `docker run`:
```
--entrypoint "/bin/bash" -v $(pwd)/ms-build-mi300:/ms-build-mi300
```

Please run:
```
pip install datasets==2.19.1 deepspeed==0.14.2 transformers==4.40.2 peft==0.10.0
apt update && apt install libaio-dev -y
```
and then:
```
HF_CACHE="/data" accelerate launch --config_file deepspeed_zero3.yml peft_fine_tuning.py
```
