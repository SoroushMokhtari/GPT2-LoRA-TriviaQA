# A Cinephile GPT

> A GPT-2 model fine-tuned with Low-Rank Adaptation (LoRA) to become a movie trivia expert.

## Description

Cinephile-GPT is a project that demonstrates how to fine-tune a large language model (LLM) like GPT-2 on a specific domain—in this case, movie trivia. It uses a Retrieval-Augmented Generation (RAG) approach to find relevant context and **Low-Rank Adaptation (LoRA)** for efficient fine-tuning.

The core challenge addressed here is fitting a large knowledge base into a model with a fixed context window (1024 tokens for GPT-2). The project implements a full pipeline from data processing and retrieval to training and evaluation, providing a clear example of how to build a specialized question-answering system.

## Core Technologies

- **Model**: GPT-2
- **Fine-Tuning**: Low-Rank Adaptation (LoRA) implemented from scratch in PyTorch.
- **Retrieval**: `sentence-transformers` for finding the most relevant context for a given question.
- **Framework**: PyTorch
- **Core Libraries**: Hugging Face `transformers`, `pandas`, `joblib`

## How It Works

The project follows a "Retrieve-then-Train" pipeline, which is implemented in `main.ipynb`:

1.  **Data Processing & Chunking**: The initial movie trivia dataset (`Data/movie_trivia_qa.csv`) contains large context documents. These are first split into smaller, overlapping text chunks. This step is crucial to ensure that the context, question, and answer can fit within the model's 1024-token limit.

2.  **Retrieval with Sentence-Transformers**: For each question in the dataset, a retriever model (`all-MiniLM-L6-v2`) is used to find the single "best" context chunk from the list of chunks generated in the previous step. This is done by comparing the cosine similarity of the question and chunk embeddings.

3.  **Data Preparation**: The (question, best_chunk, answer) triplets are saved into `training_data.jsonl` and `validation_data.jsonl`. These files serve as the final source for training.

4.  **LoRA Fine-Tuning**:
    - The base GPT-2 model is loaded, and its original weights are frozen.
    - Small, trainable LoRA layers are injected into the linear layers of the model.
    - The model is then fine-tuned on the prepared dataset. Only the LoRA weights are updated, which constitutes less than 1% of the total model parameters. This makes training much faster and more memory-efficient.

5.  **Evaluation**: After training, the model is evaluated on the validation set to check its ability to generate correct answers from a given context and question.

## How to Use

### 1. Setup

First, clone the repository and set up the environment. The `conda_env.yml` file is provided for easy setup with Conda.

```bash
git clone <your-repo-url>
cd LORA
conda env create -f conda_env.yml
conda activate torch_env
```

### 2. Prepare the Data

The project uses the `movie_trivia_qa.csv` dataset, which should be placed in a `Data/` directory.

Run the initial cells in the `main.ipynb` notebook to perform the chunking and retrieval steps. This will generate the following files:

- `Data/processed_chunks_with_qa.jsonl`
- `Data/training_data.jsonl`
- `Data/validation_data.jsonl`

### 3. Train the Model

Continue running the cells in the **"Training"** section of `main.ipynb`. This will:
1.  Inject the LoRA layers into the GPT-2 model.
2.  Start the training and validation loop.
3.  Save the trained LoRA weights to `lora_gpt2_triviaqa_weights.pth` upon completion.

### 4. Evaluate and Generate Answers

The final cells in the notebook demonstrate how to:
1.  Load the base GPT-2 model.
2.  Apply the LoRA structure.
3.  Load the fine-tuned `lora_gpt2_triviaqa_weights.pth` weights.
4.  Run an evaluation on sample questions from the validation set to see the model in action.

## Future Work

This project provides a solid foundation for several advanced topics:

- **Advanced RAG**: Instead of retrieving only the single best chunk, one could implement a "Sentence Windowing" approach, where the chunks before and after the best one are also included to provide richer context.
- **End-to-End Training**: A more complex setup could involve training the retriever and the generator models jointly, allowing the retriever to learn what context is most useful for the generator.
- **Experiment with Different Models**: The same LoRA fine-tuning technique can be applied to more advanced models like Llama or Mistral.