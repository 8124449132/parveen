{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "machine_shape": "hm",
      "gpuType": "T4",
      "provenance": []
    },
    "accelerator": "GPU",
    "kaggle": {
      "accelerator": "gpu"
    },
    "language_info": {
      "name": "python"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!pip install transformers sentence_transformers fastapi uvicorn streamlit colab_everything\n"
      ],
      "metadata": {
        "id": "AUYGC8A_9CQY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Local Inference on GPU\n",
        "Model page: https://huggingface.co/ibm-granite/granite-3.3-2b-instruct\n",
        "\n",
        "⚠️ If the generated code snippets do not work, please open an issue on either the [model repo](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)\n",
        "\t\t\tand/or on [huggingface.js](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts) 🙏"
      ],
      "metadata": {
        "id": "JpC21L7w9CQa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Use a pipeline as a high-level helper\n",
        "from transformers import pipeline\n",
        "\n",
        "pipe = pipeline(\"text-generation\", model=\"ibm-granite/granite-3.3-2b-instruct\")\n",
        "messages = [\n",
        "    {\"role\": \"user\", \"content\": \"Who are you?\"},\n",
        "]\n",
        "pipe(messages)"
      ],
      "metadata": {
        "id": "ZI5NSFC39CQb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Load model directly\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
        "\n",
        "tokenizer = AutoTokenizer.from_pretrained(\"ibm-granite/granite-3.3-2b-instruct\")\n",
        "model = AutoModelForCausalLM.from_pretrained(\"ibm-granite/granite-3.3-2b-instruct\")\n",
        "messages = [\n",
        "    {\"role\": \"user\", \"content\": \"Who are you?\"},\n",
        "]\n",
        "inputs = tokenizer.apply_chat_template(\n",
        "\tmessages,\n",
        "\tadd_generation_prompt=True,\n",
        "\ttokenize=True,\n",
        "\treturn_dict=True,\n",
        "\treturn_tensors=\"pt\",\n",
        ").to(model.device)\n",
        "\n",
        "outputs = model.generate(**inputs, max_new_tokens=40)\n",
        "print(tokenizer.decode(outputs[0][inputs[\"input_ids\"].shape[-1]:]))"
      ],
      "metadata": {
        "id": "OA8uiii89CQb"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}