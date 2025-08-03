# Implementation-of-AutoDAN-Turbo



## 1. Download code

1.1 Download original code from: https://github.com/SaFoLab-WISC/AutoDAN-Turbo

```shell
git clone https://github.com/SaFoLab-WISC/AutoDAN-Turbo.git
```

1.2 Download LLM Chat Templates\:

```shell
cd AutoDAN-Turbo
cd llm
git clone https://github.com/chujiezheng/chat_templates.git
cd ..
```


<img width="366" height="300" alt="1754119599469" src="https://github.com/user-attachments/assets/ef178e3a-6411-4858-988b-8fe5a1c0bfeb" />



### 1.3 Download my new code:

```shell
git clone https://github.com/Wenxin-LU0/Implementation-of-AutoDAN-Turbo.git
```

### 1.4 Replace original code:

 I modified some code, so this requires to remove the original code first and then download the new code to replace it.

Specifically, the files that need to be changed are as follows: \
-- ./AutoDAN-Turbo/data/ \
-- ./AutoDAN-Turbo/test_r.py \
-- ./AutoDAN-Turbo/llm/openai_models.py

```shell
rm -rf data
rm -f test_r.py
mv -f Implementation-of-AutoDAN-Turbo/test_r.py .
mv -f Implementation-of-AutoDAN-Turbo/data .
rm -rf Implementation-of-AutoDAN-Turbo
```

### 1.5 Manually replace code:

Due to the presence of API keys, this file cannot be uploaded to GitHub. The source code needs to be modified manually. (Since my environment has been updated with the newer versions of vllm and OpenAI, which are incompatible with the older OpenAI functions, I've made the necessary modifications). My apologies for the extra work this has generated, thank you for your support.

**File Directory:** `AutoDAN-Turbo/llm/openai_models.py`

```py
import openai
import logging
from openai import AzureOpenAI
import json
from openai import OpenAI


class OpenAIEmbeddingModel:
    def __init__(
        self,
        azure: bool = False,
        azure_endpoint: str = None,
        azure_api_version: str = "2024-02-01",
        azure_deployment_name: str = None,
        azure_api_key: str = None,
        openai_api_key: str = None,
        embedding_model: str = "text-embedding-ada-002",
        logger: logging.Logger = None
    ):
        self.azure = azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment_name = azure_deployment_name
        self.embedding_model = embedding_model
        self.logger = logger

        if self.azure:
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint
            )
        else:
            openai.api_key = openai_api_key

    def encode(self, text):
        try:
            single_input = False
            if isinstance(text, str):
                text = [text]
                single_input = True
            if self.azure:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.azure_deployment_name
                )
                data_json = response.model_dump_json(indent=2)
                data = json.loads(data_json)["data"]
                data = sorted(data, key=lambda x: x["index"])
                embeddings = [d["embedding"] for d in data]
            else:
                #response = openai.Embedding.create(
                #    input=text,
                #    model=self.embedding_model)
                #embeddings = [item["embedding"] for item in response["data"]]
                
                client = OpenAI(
                    base_url='https://xiaoai.plus/v1',
                    api_key='') # put your openai api here
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002",
                    encoding_format="float"
                )
                embeddings = response.data[0].embedding
            
            if single_input and len(embeddings) == 1:
                return embeddings[0]
            return embeddings
        except Exception as e:
            self.logger.error(f"Embedding error: {e}", exc_info=True)
            return None

```



## 2. Build environment

### 2.1 Create an environment:

```shell
cd AutoDAN-Turbo
conda create -n autodanturbo python==3.12
conda activate autodanturbo
```

### 2.2 Download requirements:

Even after downloading the original requirements, some packages were still missing, so my requirements are as follows:

```shell
pip install vllm==0.10.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple faiss-cpu
pip install numpy==2.3.2
pip install openai==1.98.0
pip install Requests==2.32.4
pip install torch==2.7.1
pip install transformers==4.54.0
pip install google-generativeai
pip install google-cloud-aiplatform
pip install flashinfer-python
pip install spacy
pip install confection
pip install fschat
pip install ray
pip install anthropic
pip install mistralai
pip install torchvision
pip install sentence_transformers
pip install matplotlib
pip install accelerate
pip install datasketch
pip install pandas
pip install art
pip install tenacity
pip install boto3
pip install bpe
pip install fbgemm-gpu
pip install wandb
```



## 3. Run AutoDAN-Turbo

### 3.1 Open venv:

```shell
conda init
conda activate autodanturbo
cd AutoDAN-Turbo
```

### 3.2 Run AutoDAN-Turbo

--openai_api_key "openai api key" \
--embedding_model "text-embedding-ada-002" \
--hf_token "huggingface token" \
--deepseek_api_key "deepseek api key" \
--deepseek_model "deepseek-reasoner" \
--epochs 150"

```shell
python test_r.py --openai_api_key "" \
                 --embedding_model "text-embedding-ada-002" \
                 --hf_token "" \
                 --deepseek_api_key "" \
                 --deepseek_model "deepseek-reasoner" \
                 --epochs 150
```




Each prompt requires 150 epochs, and running 100 prompts may take several days to complete. See `/AutoDAN-Turbo/data/AutoDAN-Turbo_JailbreakBench.json` for the results.

Thank you again for your help :)



