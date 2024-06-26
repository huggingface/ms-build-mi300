# Table of content

1. [Deploying TGI on the VM](#deploying-tgi-on-the-vm)
	1. [Options to try](#options-to-try)
        1. [Quantization](#quantization)
        2. [Tensor parallelism](#tensor-parallelism)
        3. [Speculative decoding](#speculative-decoding)
        4. [Customize HIP Graph, TunableOp warmup](#customize-hip-graph-tunableop-warmup)
        5. [Deploy several models on a single GPU](#deploy-several-models-on-a-single-gpu)
        6. [Grammar contrained generation](#grammar-contrained-generation)
        7. [Benchmarking](#benchmarking)
        8. [Vision-Language models (VLM)](#vision-language-models-vlm)
2. [Model fine-tuning](#model-fine-tuning-with-transformers-and-peft)

# Deploying TGI on the VM

Access the VM through SSH using any terminal application on your system.
 - IMPORTANT: Replace `<placeholders>` in the command according to printed setup instructions.
```
ssh \
    -L <300#>:localhost:<300#> \
    -L <888#>:localhost:<888#> \
    -L <786#>:localhost:<786#> \
    buildusere@<azure-vm-ip-address>
```

**Important: there are three ports to forward through ssh:**
* `300x`: TGI port.
* `888x`: jupyter notebook port.
* `786x`: gradio port.

From within the VM, please use the following Docker run command while taking note to first set the following variables according to your printout:
  - For `--device=/dev/dri/renderD###` set `GPUID`
  - For `--name <your-name>_tgi` set `NAME` to help identify your Docker container
```
GPUID=###
NAME=your_name
docker run --name ${NAME}_tgi --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri/renderD$GPUID --group-add video --ipc=host --shm-size 256g \
    --net host -v $(pwd)/hf/hf_cache:/data \
    --entrypoint "/bin/bash" \
    --env PYTORCH_TUNABLEOP_ENABLED=0 \
    --env HUGGING_FACE_HUB_TOKEN=$HF_READ_TOKEN \
    ghcr.io/huggingface/text-generation-inference:sha-293b8125-rocm
```
From within the container in interactive mode, make sure that you have one MI300 visible:

```
rocm-smi
```

giving
```
================================================== ROCm System Management Interface ==================================================
============================================================ Concise Info ============================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf              PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
======================================================================================================================================
0       2     0x74b5,   23674  35.0°C      132.0W    NPS1, N/A, 0        132Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
======================================================================================================================================
======================================================== End of ROCm SMI Log =========================================================
```

Then, from within the container in interactive mode, TGI can be launched with:
```
text-generation-launcher \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --num-shard 1 --port ####
```

with **the port being the one indicated on your individual instruction sheet**.

You should see a log as follow:
```
2024-05-20T17:32:40.790474Z  INFO text_generation_launcher: Default `max_input_tokens` to 4095
2024-05-20T17:32:40.790512Z  INFO text_generation_launcher: Default `max_total_tokens` to 4096
2024-05-20T17:32:40.790516Z  INFO text_generation_launcher: Default `max_batch_prefill_tokens` to 4145
2024-05-20T17:32:40.790521Z  INFO text_generation_launcher: Using default cuda graphs [1, 2, 4, 8, 16, 32]
2024-05-20T17:32:40.790658Z  INFO download: text_generation_launcher: Starting download process.
2024-05-20T17:32:43.060786Z  INFO text_generation_launcher: Files are already present on the host. Skipping download.

2024-05-20T17:32:43.794782Z  INFO download: text_generation_launcher: Successfully downloaded weights.
2024-05-20T17:32:43.795177Z  INFO shard-manager: text_generation_launcher: Starting shard rank=0
2024-05-20T17:32:46.044820Z  INFO text_generation_launcher: ROCm: using Flash Attention 2 Composable Kernel implementation.

2024-05-20T17:32:46.211469Z  WARN text_generation_launcher: Could not import Mamba: No module named 'mamba_ssm'

2024-05-20T17:32:49.436913Z  INFO text_generation_launcher: Server started at unix:///tmp/text-generation-server-0

2024-05-20T17:32:49.502406Z  INFO shard-manager: text_generation_launcher: Shard ready in 5.706223525s rank=0
2024-05-20T17:32:49.600606Z  INFO text_generation_launcher: Starting Webserver
2024-05-20T17:32:49.614745Z  INFO text_generation_router: router/src/main.rs:195: Using the Hugging Face API
2024-05-20T17:32:49.614784Z  INFO hf_hub: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/hf-hub-0.3.2/src/lib.rs:55: Token file not found "/root/.cache/huggingface/token"
2024-05-20T17:32:49.871020Z  INFO text_generation_router: router/src/main.rs:474: Serving revision c4a54320a52ed5f88b7a2f84496903ea4ff07b45 of model meta-llama/Meta-Llama-3-8B-Instruct
2024-05-20T17:32:50.068073Z  INFO text_generation_router: router/src/main.rs:289: Using config Some(Llama)
2024-05-20T17:32:50.071589Z  INFO text_generation_router: router/src/main.rs:317: Warming up model
2024-05-20T17:32:50.592906Z  INFO text_generation_launcher: PyTorch TunableOp (https://github.com/fxmarty/pytorch/tree/2.3-patched/aten/src/ATen/cuda/tunable) is enabled. The warmup may take several minutes, picking the ROCm optimal matrix multiplication kernel for the target lengths 1, 2, 4, 8, 16, 32, with typical 5-8% latency improvement for small sequence lengths. The picked GEMMs are saved in the file /data/tunableop_meta-llama-Meta-Llama-3-8B-Instruct_tp1_rank0.csv. To disable TunableOp, please launch TGI with `PYTORCH_TUNABLEOP_ENABLED=0`.

2024-05-20T17:32:50.593041Z  INFO text_generation_launcher: The file /data/tunableop_meta-llama-Meta-Llama-3-8B-Instruct_tp1_rank0.csv already exists and will be reused.

2024-05-20T17:32:50.593225Z  INFO text_generation_launcher: Warming up TunableOp for seqlen=1

2024-05-20T17:32:50.694955Z  INFO text_generation_launcher: Warming up TunableOp for seqlen=2

2024-05-20T17:32:50.707031Z  INFO text_generation_launcher: Warming up TunableOp for seqlen=4

2024-05-20T17:32:50.719015Z  INFO text_generation_launcher: Warming up TunableOp for seqlen=8

2024-05-20T17:32:50.731009Z  INFO text_generation_launcher: Warming up TunableOp for seqlen=16

2024-05-20T17:32:50.742969Z  INFO text_generation_launcher: Warming up TunableOp for seqlen=32

2024-05-20T17:32:50.755226Z  INFO text_generation_launcher: Cuda Graphs are enabled for sizes [1, 2, 4, 8, 16, 32]

2024-05-20T17:32:51.276651Z  INFO text_generation_router: router/src/main.rs:354: Setting max batch total tokens to 1346240
2024-05-20T17:32:51.276675Z  INFO text_generation_router: router/src/main.rs:355: Connected
2024-05-20T17:32:51.276679Z  WARN text_generation_router: router/src/main.rs:369: Invalid hostname, defaulting to 0.0.0.0
```

Now, in an other terminal, ssh again into the VM **with your individual TGI, jupyter & gradio ports**:
```
ssh \
    -L <300#>:localhost:<300#> \
    -L <888#>:localhost:<888#> \
    -L <786#>:localhost:<786#> \
    buildusere@<azure-vm-ip-address>
```

Then, from within the VM, launch the jupyter container as follow, replacing `<your-name>` in the command below with your name to help identify your Docker container:
  - The NAME variable will once again be used to help identify your Docker container.
```
NAME=your_name
docker run -it -u root --rm --entrypoint /bin/bash --net host \
    --env HUGGING_FACE_HUB_TOKEN=$HF_READ_TOKEN \
    --name ${NAME}_jnb \
    jupyter/base-notebook
```

Once inside this 2nd Docker container clone the repo for this workshop
```
apt update
apt install git
git clone https://github.com/huggingface/ms-build-mi300.git
```

Finally, launch the Notebooks server while taking note to replace the `<888#>` placeholder according to your printout (should be one of 8881, 8882, ..., 8888).
  - Take note of the URL supplied so can connect to Notebooks after.
```
jupyter-notebook --allow-root --port <888#>
```

You should see output that ends with something similar to:
```
[I 2024-05-19 00:38:53.523 ServerApp] Jupyter Server 2.8.0 is running at:
[I 2024-05-19 00:38:53.523 ServerApp] http://build-mi300x-vm1:8882/tree?token=9cffeac33839ab1e89e81f57dfe3be1739f4fd98729da0ad
[I 2024-05-19 00:38:53.523 ServerApp]     http://127.0.0.1:8882/tree?token=9cffeac33839ab1e89e81f57dfe3be1739f4fd98729da0ad
[I 2024-05-19 00:38:53.523 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2024-05-19 00:38:53.525 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/jpserver-121-open.html
    Or copy and paste one of these URLs:
        http://build-mi300x-vm1:8882/tree?token=9cffeac33839ab1e89e81f57dfe3be1739f4fd98729da0ad
        http://127.0.0.1:8882/tree?token=9cffeac33839ab1e89e81f57dfe3be1739f4fd98729da0ad
```

Now `local_chatbot.ipynb` can be used to query the model from your system through the SSH tunnel!  To do so...
  - Just copy and paste the provided URL into your browser (see ouput from Notebooks server)
  - The format of the URL is as such: `http://127.0.0.1:####/tree?token=<unique-value-from-output>`

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
text-generation-launcher --model-id TheBloke/Llama-2-70B-Chat-GPTQ --quantize gptq --port #####
```
or
```
text-generation-launcher --model-id casperhansen/llama-3-70b-instruct-awq --quantize awq  --port #####
```

## Tensor parallelism

Here we use only `--num-shard 1`, as only one GPU is available per person. But on a full node, one may use `--num-shard X` to decide how many GPUs are best to deploy a model given for example latency constraints.

Read more about tensor parallelism in TGI: https://huggingface.co/docs/text-generation-inference/conceptual/tensor_parallelism

## Speculative decoding

TGI supports **n-gram** speculation, as well as [**Medusa**](https://arxiv.org/pdf/2401.10774) speculative decoding.

In the launcher, the argument `--speculate X` allows to use speculative decoding. This argument specifies the number of input_ids to speculate on if using a medusa model, or using n-gram speculation.

Example, adding to the docker command:
```
text-generation-launcher --model-id mistralai/Mistral-7B-Instruct-v0.2 --speculate 3 --port #####
```
or with Medusa:

```
text-generation-launcher --model-id text-generation-inference/Mistral-7B-Instruct-v0.2-medusa --speculate 3  --port #####
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
HF_CACHE="/data" accelerate launch --config_file deepspeed_zero3.yaml peft_fine_tuning.py
```
